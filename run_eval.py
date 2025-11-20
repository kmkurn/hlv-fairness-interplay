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

import logging
import os
from pathlib import Path
from typing import Literal, Optional, cast

import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from scipy.stats import pmean

from fair_eval import compute_for_multilabel_group as compute_fair_score
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


@ex.config
def default():
    # training artifacts directory
    artifacts_dir = ""
    # exponent value for group aggregation of fairness scores
    p_group = 0


@ex.capture
def eval_pred(
    dev: dict[Literal["true", "pred"], np.ndarray],
    test: dict[Literal["true", "pred"], np.ndarray],
    membership: Optional[np.ndarray] = None,
    p_group: float = 0.0,
) -> dict[
    Literal["perf", "fair"],
    dict[Literal["dev", "test"], dict[Literal["poJSD", "soft_acc", "soft_f1"], float]],
]:
    fair_kwargs = {}
    if abs(p_group) > 1e-7:
        fair_kwargs["group_agg"] = lambda x: pmean(x, p_group, axis=-1)
    result = {
        "perf": {
            "dev": {
                "poJSD": cast(float, poJSD(dev["true"], dev["pred"])),
                "soft_acc": soft_accuracy(dev["true"], dev["pred"]),
            },
            "test": {
                "poJSD": cast(float, poJSD(test["true"], test["pred"])),
                "soft_acc": soft_accuracy(test["true"], test["pred"]),
            },
        },
    }
    if membership is not None:
        result.update(
            {
                "fair": {
                    "test": {
                        "poJSD": compute_fair_score(
                            test["true"],
                            test["pred"],
                            membership,
                            lambda y_true, y_pred: cast(
                                np.ndarray, poJSD(y_true, y_pred, classwise=True)
                            ),
                            **fair_kwargs,
                        ),
                        "soft_f1": compute_fair_score(
                            test["true"],
                            test["pred"],
                            membership,
                            lambda y_true, y_pred: soft_prfs(y_true, y_pred)[2],
                            **fair_kwargs,
                        ),
                    }
                },
            }
        )
    return result  # type: ignore[return-value]


@ex.automain
def evaluate(artifacts_dir, _run, _log=None):
    """Evaluate model predictions."""
    if _log is None:
        _log = logging.getLogger(__name__)
    artifacts_dir = Path(artifacts_dir)
    dev = np.load(artifacts_dir / "dev.npz")
    test = np.load(artifacts_dir / "test.npz")
    try:
        membership = np.load(artifacts_dir / "membership.npy")
    except FileNotFoundError:
        membership = None
    result = eval_pred(
        {"true": dev["true"], "pred": dev["pred"]},
        {"true": test["true"], "pred": test["pred"]},
        membership,
    )
    for split in ("dev", "test"):
        for metric in ("poJSD", "soft_acc"):
            _run.log_scalar(f"{split}/perf/{metric}", result["perf"][split][metric])
            _log.info("%s/perf/%s: %.4f", split, metric, result["perf"][split][metric])
    if "fair" in result:
        for metric in ("poJSD", "soft_f1"):
            _run.log_scalar(f"test/fair/{metric}", result["fair"]["test"][metric])
            _log.info("test/fair/%s: %.4f", metric, result["fair"]["test"][metric])
