#!/usr/bin/env python

# Copyright 2025 Kemal Kurniawan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, cast

import numpy as np
from flair.data import Corpus, Dictionary, Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from flair.embeddings.base import register_embeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.trainers.plugins import MetricRecord, TrainerPlugin
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds

from ann_agg import MajVoteAgg, SoftLabelAgg
from classifier import TextSoftClassifier
from corpus import SBICorpus
from fair_eval import compute_for_multilabel_group as compute_fair_score
from hlv_loss import JSDLoss, SoftMicroF1Loss
from perf_eval import poJSD, soft_accuracy
from perf_eval import soft_precision_recall_fscore_support as soft_prfs

ex = Experiment("hlv-fairness")
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore[assignment]
if "SACRED_MONGO_URL" in os.environ:
    ex.observers.append(
        MongoObserver(
            url=os.environ["SACRED_MONGO_URL"],
            db_name=os.getenv("SACRED_DB_NAME", "sacred"),
        )
    )


class SacredLogMetricsPlugin(TrainerPlugin):
    def __init__(self, run: Run, log_every: int = 10) -> None:
        super().__init__()
        self.__run = run
        self.__log_every = log_every

    @TrainerPlugin.hook
    def metric_recorded(self, record: MetricRecord) -> None:
        if self.__should_log(record):
            try:
                value = record.value.item()
            except AttributeError:
                value = record.value
            self.__run.log_scalar(str(record.name), value, step=record.global_step)

    def __should_log(self, record: MetricRecord) -> bool:
        is_batch_metric = len(record.name.parts) >= 2 and record.name.parts[1] in (
            "batch_loss",
            "gradient_norm",
        )
        return record.is_scalar and (
            not is_batch_metric or record.global_step % self.__log_every == 0
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__} | log_every: {self.__log_every}"


class EvaluateOnDevPlugin(TrainerPlugin):
    def __init__(
        self,
        datapoints: list[Sentence],
        run: Run,
        batch_size: int = 32,
        show_pbar: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__()
        if logger is None:
            logger = logging.getLogger(__name__)
        self.__datapoints = datapoints
        self.__batch_size = batch_size
        self.__show_pbar = show_pbar
        self.__run = run
        self.__logger = logger
        self.last_predictions = None

    @TrainerPlugin.hook
    def after_training_epoch(self, epoch):
        datapoints = self.__datapoints
        model = self.model
        batch_size = self.__batch_size
        show_pbar = self.__show_pbar
        run = self.__run
        log = self.__logger

        log.info("Evaluating on dev at the end of epoch %d", epoch)
        pred_label_type = f"pred {model.label_type}"
        model.predict(
            datapoints,
            batch_size,
            return_probabilities_for_all_classes=True,
            label_name=pred_label_type,
            verbose=show_pbar,
        )
        y_true, y_pred = np.zeros(
            [2, len(datapoints), len(model.label_dictionary)], dtype=float
        )
        for i, dp in enumerate(datapoints):
            for label in dp.get_labels(model.label_type):
                y_true[
                    i, model.label_dictionary.get_idx_for_item(label.value)
                ] = label.score
            for label in dp.get_labels(pred_label_type):
                y_pred[
                    i, model.label_dictionary.get_idx_for_item(label.value)
                ] = label.score
        self.last_predictions = (y_true, y_pred)
        pojsd = poJSD(y_true, y_pred)
        soft_acc = soft_accuracy(y_true, y_pred)
        pojsd = cast(float, pojsd)
        run.log_scalar("dev/poJSD", pojsd)
        log.info("dev/poJSD: %.4f", pojsd)
        run.log_scalar("dev/soft_acc", soft_acc)
        log.info("dev/soft_acc: %.4f", soft_acc)


@ex.config
def default():
    # data directory
    data_dir = ""
    # pretrained model name (see https://huggingface.co/models)
    model_name = "roberta-base"
    # whether to use LoRA
    use_lora = False
    # modules to apply LoRA to (ignored if use_lora=False)
    lora_target = None
    # training method [ReL, MV, SL, JSD, SmF1]
    method = "ReL"
    # scalar regulating entropy of soft labels (unused by ReL and MV)
    alpha = 1.0
    # learning rate
    lr = 5e-5
    # batch size for training
    batch_size = 32
    # batch size for evaluation
    eval_batch_size = 32
    # chunk batches into this size for grad accumulation
    batch_chunk_size = None
    # max epochs
    max_epochs = 10
    # log train statistics every this number of batches
    log_every = 10
    # metadata key to get group names
    group_names_metadata_key = "target_cats"
    # root directory for temp artifacts dir
    temp_root_dir = ""
    # save training artifacts in this directory
    artifacts_dir = ""
    # metadata to save if artifacts_dir is given
    metadata_to_save = ["annotator_ids", "target_cats"]
    # downsampling rate for train set
    train_sample = 1.0
    # downsampling rate for dev set
    dev_sample = 1.0
    # downsampling rate for test set
    test_sample = 1.0
    # whether to show evaluation progress bar
    show_eval_pbar = False


@ex.capture
def train_model(
    model: TextClassifier,
    corpus: Corpus,
    _run: Run,
    lr: float = 5e-5,
    batch_size: int = 32,
    batch_chunk_size: Optional[int] = None,
    max_epochs: int = 1,
    log_every: int = 10,
    group_names_metadata_key: str = "group_names",
    temp_root_dir: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
    metadata_to_save: Optional[list[str]] = None,
    eval_batch_size: int = 32,
    show_eval_pbar: bool = False,
    _log: Optional[logging.Logger] = None,
) -> None:
    if not temp_root_dir:
        temp_root_dir = None
    if metadata_to_save is None:
        metadata_to_save = []
    if _log is None:
        _log = logging.getLogger(__name__)
    dev_datapoints = None
    if corpus.dev:
        _log.info("Dev set has %d sentences", len(corpus.dev))  # type: ignore[arg-type]
        dev_datapoints = corpus.dev.datapoints  # type: ignore[attr-defined]
        corpus._dev = None
    test_datapoints = None
    if corpus.test:
        _log.info("Test set has %d sentences", len(corpus.test))  # type: ignore[arg-type]
        test_datapoints = corpus.test.datapoints  # type: ignore[attr-defined]
        corpus._test = None

    trainer = ModelTrainer(model, corpus)
    plugins: list[TrainerPlugin] = [SacredLogMetricsPlugin(_run, log_every)]
    eval_on_dev_plugin = None
    if dev_datapoints:
        eval_on_dev_plugin = EvaluateOnDevPlugin(
            dev_datapoints, _run, eval_batch_size, show_eval_pbar, _log
        )
        plugins.append(eval_on_dev_plugin)
    if artifacts_dir:
        artifacts_dir_ = Path(artifacts_dir)
        trainer.fine_tune(
            artifacts_dir,
            learning_rate=lr,
            mini_batch_size=batch_size,
            mini_batch_chunk_size=batch_chunk_size,
            max_epochs=max_epochs,
            plugins=plugins,
        )
    else:
        artifacts_dir_ = None
        with tempfile.TemporaryDirectory(dir=temp_root_dir) as tmpdir:
            trainer.fine_tune(
                tmpdir,
                learning_rate=lr,
                mini_batch_size=batch_size,
                mini_batch_chunk_size=batch_chunk_size,
                max_epochs=max_epochs,
                plugins=plugins,
            )

    pred_label_type = f"pred {model.label_type}"
    model.eval()
    if artifacts_dir_ and dev_datapoints:
        _log.info("Saving dev data to %s", artifacts_dir_ / "dev.jsonl")
        with (artifacts_dir_ / "dev.jsonl").open("w", encoding="utf8") as f:
            for dp in dev_datapoints:
                print(
                    json.dumps(
                        {
                            "text": dp.text,
                            "metadata": {
                                m: list(x)
                                if isinstance((x := dp.get_metadata(m)), set)
                                else x
                                for m in metadata_to_save
                            },
                        }
                    ),
                    file=f,
                )
        assert (
            eval_on_dev_plugin is not None
            and eval_on_dev_plugin.last_predictions is not None
        )
        y_true, y_pred = eval_on_dev_plugin.last_predictions
        _log.info("Saving dev predictions to %s", artifacts_dir_ / "dev.npz")
        np.savez_compressed(artifacts_dir_ / "dev.npz", true=y_true, pred=y_pred)
    if test_datapoints:
        if artifacts_dir_:
            _log.info("Saving test data to %s", artifacts_dir_ / "test.jsonl")
            with (artifacts_dir_ / "test.jsonl").open("w", encoding="utf8") as f:
                for dp in test_datapoints:
                    print(
                        json.dumps(
                            {
                                "text": dp.text,
                                "metadata": {
                                    m: list(x)
                                    if isinstance((x := dp.get_metadata(m)), set)
                                    else x
                                    for m in metadata_to_save
                                },
                            }
                        ),
                        file=f,
                    )
        _log.info("Making predictions on test")
        model.predict(
            test_datapoints,
            eval_batch_size,
            return_probabilities_for_all_classes=True,
            label_name=pred_label_type,
            verbose=show_eval_pbar,
        )
        y_true, y_pred = np.zeros(
            [2, len(test_datapoints), len(model.label_dictionary)], dtype=float
        )
        for i, dp in enumerate(test_datapoints):
            for label in dp.get_labels(model.label_type):
                y_true[
                    i, model.label_dictionary.get_idx_for_item(label.value)
                ] = label.score
            for label in dp.get_labels(pred_label_type):
                y_pred[
                    i, model.label_dictionary.get_idx_for_item(label.value)
                ] = label.score
        if artifacts_dir_:
            _log.info("Saving test predictions to %s", artifacts_dir_ / "test.npz")
            np.savez_compressed(artifacts_dir_ / "test.npz", true=y_true, pred=y_pred)

        _log.info("Evaluating performance on test")
        pojsd = poJSD(y_true, y_pred)
        soft_acc = soft_accuracy(y_true, y_pred)
        pojsd = cast(float, pojsd)
        _run.log_scalar("test/perf/poJSD", pojsd)
        _log.info("test/perf/poJSD: %.4f", pojsd)
        _run.log_scalar("test/perf/soft_acc", soft_acc)
        _log.info("test/perf/soft_acc: %.4f", soft_acc)

        can_eval_fair = True
        group_names = set()
        for dp in test_datapoints:
            try:
                group_names.update(dp.get_metadata(group_names_metadata_key))
            except KeyError:
                can_eval_fair = False
                break
        if can_eval_fair:
            group_dict = Dictionary(add_unk=False)
            for name in sorted(group_names):
                group_dict.add_item(name)
            if artifacts_dir_:
                _log.info("Saving group dict to %s", artifacts_dir_ / "group.dict")
                group_dict.save(artifacts_dir_ / "group.dict")
            group_membership = np.zeros(
                [len(test_datapoints), len(group_dict)], dtype=bool
            )
            for i, dp in enumerate(test_datapoints):
                for name in dp.get_metadata(group_names_metadata_key):
                    group_membership[i, group_dict.get_idx_for_item(name)] = True
            if artifacts_dir_:
                _log.info(
                    "Saving test group membership to %s",
                    artifacts_dir_ / "membership.npy",
                )
                np.save(artifacts_dir_ / "membership.npy", group_membership)
            _log.info("Evaluating fairness on test")
            fair_score = compute_fair_score(
                y_true,
                y_pred,
                group_membership,
                lambda y_true, y_pred: poJSD(y_true, y_pred, classwise=True),  # type: ignore[arg-type,return-value]
            )
            _run.log_scalar("test/fair/poJSD", fair_score)
            _log.info("test/fair/poJSD: %.4f", fair_score)
            fair_score = compute_fair_score(
                y_true,
                y_pred,
                group_membership,
                lambda y_true, y_pred: soft_prfs(y_true, y_pred)[2],
            )
            _run.log_scalar("test/fair/soft_f1", fair_score)
            _log.info("test/fair/soft_f1: %.4f", fair_score)


@register_embeddings
class TransformerDocumentEmbeddingsWithLoRA(TransformerDocumentEmbeddings):
    def __init__(
        self,
        lora_config: LoraConfig | dict,
        save_only_lora_parameters: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(lora_config, dict):
            lora_config = LoraConfig(**lora_config)
        self.model = get_peft_model(self.model, lora_config)
        self._lora_config = lora_config
        self._save_only_lora = save_only_lora_parameters
        if save_only_lora_parameters:
            self._register_state_dict_hook(self.__replace_model_state_dict)
            self._register_load_state_dict_pre_hook(self.__set_model_state_dict)
            self.register_load_state_dict_post_hook(self.__process_incompatible_keys)

    def to_params(self):
        params = super().to_params()
        return {
            **params,
            "lora_config": self._lora_config.to_dict(),
            "save_only_lora_parameters": self._save_only_lora,
        }

    def __replace_model_state_dict(self, module, state_dict, prefix, *args):
        assert self is module
        for name in list(state_dict.keys()):
            if name.startswith(f"{prefix}model."):
                state_dict.pop(name)
        for name, value in get_peft_model_state_dict(self.model).items():
            state_dict[f"{prefix}model.{name}"] = value

    def __set_model_state_dict(self, state_dict, prefix, *args):
        set_peft_model_state_dict(
            self.model,
            {
                name[len(f"{prefix}model.") :]: param
                for name, param in state_dict.items()
                if name.startswith(f"{prefix}model.")
            },
        )
        self.__prefix = prefix

    def __process_incompatible_keys(self, module, incompatible_keys):
        for key in list(incompatible_keys.missing_keys):
            if key.startswith(f"{self.__prefix}model."):
                incompatible_keys.missing_keys.remove(key)
        for key in list(incompatible_keys.unexpected_keys):
            if key.startswith(f"{self.__prefix}model."):
                incompatible_keys.unexpected_keys.remove(key)


@ex.automain
def train(
    data_dir,
    _run,
    _log,
    _seed,
    use_lora=False,
    model_name="bert-base-uncased",
    method="ReL",
    alpha=1.0,
    lora_target=None,
    train_sample=1.0,
    dev_sample=1.0,
    test_sample=1.0,
):
    """Train a model on the SBI corpus."""
    typename = "offensiveness"
    train_agg = None
    if method == "MV":
        train_agg = MajVoteAgg(typename)
    elif method in ("SL", "JSD", "SmF1"):
        train_agg = SoftLabelAgg(typename, SBICorpus.labelset, alpha)
    elif method != "ReL":
        raise ValueError(f"unknown method: {method}")
    _log.info("Reading SBI corpus from %s", data_dir)
    corpus = SBICorpus(
        Path(data_dir), train_agg, train_sample, dev_sample, test_sample, _seed
    )
    label_dict = Dictionary(add_unk=False)
    for l in SBICorpus.labelset:
        label_dict.add_item(l)

    _log.info("Creating classifier based on %s", model_name)
    if use_lora:
        _log.info("use_lora=True; LoRA will be used")
        embeddings = TransformerDocumentEmbeddingsWithLoRA(
            {"target_modules": lora_target}, model=model_name
        )
    else:
        embeddings = TransformerDocumentEmbeddings(model_name)
    if not embeddings.tokenizer.pad_token:
        _log.info("%s tokenizer has no pad_token; setting it to eos_token", model_name)
        embeddings.tokenizer.pad_token = embeddings.tokenizer.eos_token
    if method in ("ReL", "MV"):
        model = TextClassifier(embeddings, typename, label_dictionary=label_dict)
    else:
        model = TextSoftClassifier(embeddings, typename, label_dictionary=label_dict)
        if method == "JSD":
            model.loss_function = JSDLoss()
        elif method == "SmF1":
            model.loss_function = SoftMicroF1Loss()
        else:
            assert method == "SL"

    train_model(model, corpus, _run)
