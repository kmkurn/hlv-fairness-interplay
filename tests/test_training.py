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
from unittest.mock import Mock, call, patch

import numpy as np
import pytest
from flair.data import Corpus, Dictionary, Sentence
from flair.datasets import FlairDatapointDataset
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from perf_eval import poJSD
from perf_eval import soft_precision_recall_fscore_support as soft_prfs
from run_training import train_model
from sacred.run import Run


@pytest.fixture
def make_corpus():
    def _make_corpus(train, dev=None, test=None):
        if dev:
            dev = FlairDatapointDataset(dev)
        if test:
            test = FlairDatapointDataset(test)
        return Corpus(
            FlairDatapointDataset(train), dev, test, sample_missing_splits=False
        )

    return _make_corpus


@pytest.fixture
def make_label_dict():
    def _make_label_dict(labels):
        label_dict = Dictionary(add_unk=False)
        for l in labels:
            label_dict.add_item(l)
        return label_dict

    return _make_label_dict


def test_log_test_perf_scores(make_corpus, make_label_dict):
    label_type = "label"
    s1 = Sentence("foo")
    s1.add_label(label_type, value="A")
    s2 = Sentence("bar")
    s2.add_label(label_type, value="A", score=0.7)
    s2.add_label(label_type, value="B", score=0.3)
    corpus = make_corpus(s1, test=s2)
    label_dict = make_label_dict("AB")

    class MyTextClassifier(TextClassifier):
        def predict(self, sentences, *args, **kwargs):
            assert kwargs["return_probabilities_for_all_classes"]
            assert (label_name := kwargs["label_name"])
            s = sentences[0]
            s.add_label(label_name, value="A", score=0.1)
            s.add_label(label_name, value="B", score=0.9)
            return 0.0

    model = MyTextClassifier(
        TransformerDocumentEmbeddings(), label_type, label_dictionary=label_dict
    )
    mock_run = Mock(spec=Run)

    with patch("run_training.poJSD", return_value=0.654) as mock_pojsd, patch(
        "run_training.soft_accuracy", return_value=0.333
    ) as mock_sacc:
        train_model(model, corpus, mock_run)

    mock_pojsd.assert_called_once_with(
        pytest.approx(np.array([[0.7, 0.3]])),
        pytest.approx(np.array([[0.1, 0.9]])),
    )
    mock_sacc.assert_called_once_with(
        pytest.approx(np.array([[0.7, 0.3]])),
        pytest.approx(np.array([[0.1, 0.9]])),
    )
    assert [
        c
        for c in mock_run.log_scalar.call_args_list
        if c.args[0].startswith("test/perf/")
    ] == [
        call("test/perf/poJSD", 0.654),
        call("test/perf/soft_acc", 0.333),
    ]


def test_log_test_fair_scores(make_corpus, make_label_dict):
    label_type = "label"
    s1 = Sentence("foo")
    s1.add_label(label_type, value="A")
    s2 = Sentence("bar")
    s2.add_label(label_type, value="A", score=0.7)
    s2.add_label(label_type, value="B", score=0.3)
    s2.add_metadata("group_names", {"C1", "C3"})
    s3 = Sentence("baz")
    s3.add_label(label_type, value="A", score=0.5)
    s3.add_label(label_type, value="B", score=0.5)
    s3.add_metadata("group_names", {"C2"})
    corpus = make_corpus(s1, test=[s2, s3])
    label_dict = make_label_dict("AB")

    class MyTextClassifier(TextClassifier):
        def predict(self, sentences, *args, **kwargs):
            assert kwargs["return_probabilities_for_all_classes"]
            assert (label_name := kwargs["label_name"])
            s1, s2 = sentences
            s1.add_label(label_name, value="A", score=0.1)
            s1.add_label(label_name, value="B", score=0.9)
            s2.add_label(label_name, value="A", score=0.6)
            s2.add_label(label_name, value="B", score=0.4)
            return 0.0

    model = MyTextClassifier(
        TransformerDocumentEmbeddings(), label_type, label_dictionary=label_dict
    )
    mock_run = Mock(spec=Run)
    exp_y_true = np.array([[0.7, 0.3], [0.5, 0.5]])
    exp_y_pred = np.array([[0.1, 0.9], [0.6, 0.4]])

    with patch(
        "run_training.compute_fair_score", side_effect=[0.445, 0.554]
    ) as mock_fair:
        train_model(model, corpus, mock_run)

    assert len(mock_fair.call_args_list) == 2
    assert all(c.args[0] == pytest.approx(exp_y_true) for c in mock_fair.call_args_list)
    assert all(c.args[1] == pytest.approx(exp_y_pred) for c in mock_fair.call_args_list)
    assert all(
        (c.args[2] == np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)).all()
        for c in mock_fair.call_args_list
    )
    assert mock_fair.call_args_list[0].args[3](exp_y_true, exp_y_pred) == pytest.approx(
        poJSD(exp_y_true, exp_y_pred, classwise=True)
    )
    assert mock_fair.call_args_list[1].args[3](exp_y_true, exp_y_pred) == pytest.approx(
        soft_prfs(exp_y_true, exp_y_pred)[2]
    )
    assert [
        c
        for c in mock_run.log_scalar.call_args_list
        if c.args[0].startswith("test/fair/")
    ] == [call("test/fair/poJSD", 0.445), call("test/fair/soft_f1", 0.554)]


def test_eval_on_dev_each_epoch(make_corpus, make_label_dict):
    label_type = "label"
    s1 = Sentence("foo")
    s1.add_label(label_type, value="A")
    s2 = Sentence("bar")
    s2.add_label(label_type, value="A", score=0.7)
    s2.add_label(label_type, value="B", score=0.3)
    corpus = make_corpus(s1, dev=s2)
    label_dict = make_label_dict("AB")

    class MyTextClassifier(TextClassifier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__count = 0

        def predict(self, sentences, *args, **kwargs):
            assert kwargs["return_probabilities_for_all_classes"]
            assert (label_name := kwargs["label_name"])
            s = sentences[0]
            if self.__count == 0:
                s.add_label(label_name, value="A", score=0.1)
                s.add_label(label_name, value="B", score=0.9)
            else:
                s.add_label(label_name, value="A", score=0.6)
                s.add_label(label_name, value="B", score=0.4)
            self.__count += 1
            return 0.0

    model = MyTextClassifier(
        TransformerDocumentEmbeddings(), label_type, label_dictionary=label_dict
    )
    mock_run = Mock(spec=Run)

    with patch(
        "run_training.poJSD",
        side_effect=[0.654, 0.432],
    ) as mock_pojsd, patch(
        "run_training.soft_accuracy", side_effect=[0.333, 0.112]
    ) as mock_sacc:
        train_model(model, corpus, mock_run, max_epochs=2)

    assert mock_pojsd.call_args_list == [
        call(
            pytest.approx(np.array([[0.7, 0.3]])),
            pytest.approx(np.array([[0.1, 0.9]])),
        ),
        call(
            pytest.approx(np.array([[0.7, 0.3]])),
            pytest.approx(np.array([[0.6, 0.4]])),
        ),
    ]
    assert mock_sacc.call_args_list == [
        call(
            pytest.approx(np.array([[0.7, 0.3]])), pytest.approx(np.array([[0.1, 0.9]]))
        ),
        call(
            pytest.approx(np.array([[0.7, 0.3]])), pytest.approx(np.array([[0.6, 0.4]]))
        ),
    ]
    assert [
        c for c in mock_run.log_scalar.call_args_list if c.args[0].startswith("dev/")
    ] == [
        # Epoch 1
        call("dev/poJSD", 0.654),
        call("dev/soft_acc", 0.333),
        # Epoch 2
        call("dev/poJSD", 0.432),
        call("dev/soft_acc", 0.112),
    ]


def test_log_train_loss(make_corpus, make_label_dict):
    label_type = "label"
    s1 = Sentence("foo")
    s1.add_label(label_type, value="A")
    s2 = Sentence("bar")
    s2.add_label(label_type, value="B")
    s3 = Sentence("baz")
    s3.add_label(label_type, value="B")
    s4 = Sentence("quux")
    s4.add_label(label_type, value="B")
    corpus = make_corpus([s1, s2, s3, s4])
    label_dict = make_label_dict("AB")

    class MyTextClassifier(TextClassifier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_history = []

        def forward_loss(self, *args, **kwargs):
            res = super().forward_loss(*args, **kwargs)
            self.loss_history.append(res[0])
            return res

    model = MyTextClassifier(
        TransformerDocumentEmbeddings(), label_type, label_dictionary=label_dict
    )
    mock_run = Mock(spec=Run)

    train_model(model, corpus, mock_run, batch_size=1, max_epochs=3, log_every=2)

    assert [
        (c.args[1], c.kwargs)
        for c in mock_run.log_scalar.call_args_list
        if c.args[0] == "train/batch_loss"
    ] == [
        (pytest.approx(model.loss_history[i].item()), {"step": step})
        for i in range(12)
        if (step := (i + 1)) % 2 == 0
    ]


def test_model_is_saved(tmp_path, make_corpus, make_label_dict):
    s = Sentence("foo")
    s.add_label("label", value="A")
    corpus = make_corpus([s])
    label_dict = make_label_dict("AB")
    model = TextClassifier(
        TransformerDocumentEmbeddings(), label_type="label", label_dictionary=label_dict
    )

    train_model(model, corpus, Mock(spec=Run), artifacts_dir=tmp_path)

    assert (
        TextClassifier.load(tmp_path / "final-model.pt").label_dictionary.get_items()
        == label_dict.get_items()
    )


def test_predictions_are_saved(tmp_path, make_corpus, make_label_dict):
    label_type = "label"
    s1 = Sentence("foo")
    s1.add_label(label_type, value="A")
    s2 = Sentence("bar")
    s2.add_label(label_type, value="A", score=0.9)
    s2.add_label(label_type, value="B", score=0.1)
    s2.add_metadata("m1", {1, 15})
    s3 = Sentence("baz")
    s3.add_label(label_type, value="A", score=0.3)
    s3.add_label(label_type, value="B", score=0.7)
    s3.add_metadata("m1", {49, 25})
    s4 = Sentence("quux")
    s4.add_label(label_type, value="A", score=0.2)
    s4.add_label(label_type, value="B", score=0.8)
    s4.add_metadata("m1", {101, 11})
    s5 = Sentence("abc")
    s5.add_label(label_type, value="A", score=0.5)
    s5.add_label(label_type, value="B", score=0.5)
    s5.add_metadata("m1", {2, 22})
    corpus = make_corpus([s1], [s2, s3], [s4, s5])
    label_dict = make_label_dict("AB")

    class MyTextClassifier(TextClassifier):
        text2apred = {"bar": 0.15, "baz": 0.46, "quux": 0.33, "abc": 0.98}

        def predict(self, sentences, *args, **kwargs):
            label_name = kwargs["label_name"]
            for s in sentences:
                s.add_label(label_name, value="A", score=self.text2apred[s.text])
                s.add_label(label_name, value="B", score=1 - self.text2apred[s.text])

    model = MyTextClassifier(
        TransformerDocumentEmbeddings(), label_type, label_dictionary=label_dict
    )

    train_model(
        model, corpus, Mock(spec=Run), artifacts_dir=tmp_path, metadata_to_save=["m1"]
    )
    loaded_dev = np.load(tmp_path / "dev.npz")
    loaded_test = np.load(tmp_path / "test.npz")

    assert loaded_dev["true"] == pytest.approx(np.array([[0.9, 0.1], [0.3, 0.7]]))
    assert loaded_dev["pred"] == pytest.approx(np.array([[0.15, 0.85], [0.46, 0.54]]))
    assert loaded_test["true"] == pytest.approx(np.array([[0.2, 0.8], [0.5, 0.5]]))
    assert loaded_test["pred"] == pytest.approx(np.array([[0.33, 0.67], [0.98, 0.02]]))
    # NOTE cannot use pickle because it crashes with RecursionError when dumping real data
    with (tmp_path / "dev.jsonl").open(encoding="utf8") as f:
        assert [json.loads(l) for l in f] == [
            {"text": s2.text, "metadata": {"m1": list(s2.get_metadata("m1"))}},
            {"text": s3.text, "metadata": {"m1": list(s3.get_metadata("m1"))}},
        ]
    with (tmp_path / "test.jsonl").open(encoding="utf8") as f:
        assert [json.loads(l) for l in f] == [
            {"text": s4.text, "metadata": {"m1": list(s4.get_metadata("m1"))}},
            {"text": s5.text, "metadata": {"m1": list(s5.get_metadata("m1"))}},
        ]


def test_groups_are_saved(tmp_path, make_corpus, make_label_dict):
    label_type = "label"
    s1 = Sentence("foo")
    s1.add_label(label_type, value="A")
    s2 = Sentence("bar")
    s2.add_label(label_type, value="A", score=0.9)
    s2.add_label(label_type, value="B", score=0.1)
    s2.add_metadata("group_names", {"X", "Z"})
    s3 = Sentence("baz")
    s3.add_label(label_type, value="A", score=0.3)
    s3.add_label(label_type, value="B", score=0.7)
    s3.add_metadata("group_names", {"Y"})
    corpus = make_corpus([s1], test=[s2, s3])
    label_dict = make_label_dict("AB")
    model = TextClassifier(
        TransformerDocumentEmbeddings(), label_type, label_dictionary=label_dict
    )

    train_model(model, corpus, Mock(spec=Run), artifacts_dir=tmp_path)

    assert Dictionary.load_from_file(tmp_path / "group.dict").get_items() == list("XYZ")
    assert (
        np.load(tmp_path / "membership.npy")
        == np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    ).all()
