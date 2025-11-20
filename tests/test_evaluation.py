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

from unittest.mock import Mock, call, patch

import numpy as np
import pytest
from perf_eval import poJSD
from perf_eval import soft_precision_recall_fscore_support as soft_prfs
from run_eval import evaluate
from sacred.run import Run


def test_log_perf_scores(tmp_path):
    rng = np.random.default_rng(seed=0)
    dev_true, dev_pred = rng.random(size=[2, 3, 5])
    test_true, test_pred = rng.random(size=[2, 3, 5])
    np.savez_compressed(tmp_path / "dev.npz", true=dev_true, pred=dev_pred)
    np.savez_compressed(tmp_path / "test.npz", true=test_true, pred=test_pred)
    mock_run = Mock(spec=Run)

    with patch("run_eval.poJSD", side_effect=[0.654, 0.432]) as mock_pojsd, patch(
        "run_eval.soft_accuracy", side_effect=[0.333, 0.111]
    ) as mock_sacc:
        evaluate(str(tmp_path), mock_run)

    assert (
        call(pytest.approx(dev_true), pytest.approx(dev_pred))
        in mock_pojsd.call_args_list
    )
    assert (
        call(pytest.approx(test_true), pytest.approx(test_pred))
        in mock_pojsd.call_args_list
    )
    assert (
        call(pytest.approx(dev_true), pytest.approx(dev_pred))
        in mock_sacc.call_args_list
    )
    assert (
        call(pytest.approx(test_true), pytest.approx(test_pred))
        in mock_sacc.call_args_list
    )
    assert call("dev/perf/poJSD", pytest.approx(0.654)) in mock_run.log_scalar.call_args_list
    assert (
        call("dev/perf/soft_acc", pytest.approx(0.333)) in mock_run.log_scalar.call_args_list
    )
    assert (
        call("test/perf/poJSD", pytest.approx(0.432)) in mock_run.log_scalar.call_args_list
    )
    assert (
        call("test/perf/soft_acc", pytest.approx(0.111))
        in mock_run.log_scalar.call_args_list
    )


def test_log_fair_scores(tmp_path):
    rng = np.random.default_rng(seed=0)
    test_true, test_pred = rng.random(size=[2, 3, 5])
    membership = np.array([[1, 0], [0, 0], [0, 1]], dtype=bool)
    np.savez_compressed(
        tmp_path / "dev.npz", true=rng.random(size=[3, 5]), pred=rng.random(size=[3, 5])
    )
    np.savez_compressed(tmp_path / "test.npz", true=test_true, pred=test_pred)
    np.save(tmp_path / "membership.npy", membership)
    mock_run = Mock(spec=Run)

    with patch("run_eval.compute_fair_score", side_effect=[0.445, 0.554]) as mock_fair:
        evaluate(str(tmp_path), mock_run)

    assert len(mock_fair.call_args_list) == 2
    assert all(c.args[0] == pytest.approx(test_true) for c in mock_fair.call_args_list)
    assert all(c.args[1] == pytest.approx(test_pred) for c in mock_fair.call_args_list)
    assert all((c.args[2] == membership).all() for c in mock_fair.call_args_list)
    assert mock_fair.call_args_list[0].args[3](test_true, test_pred) == pytest.approx(
        poJSD(test_true, test_pred, classwise=True)
    )
    assert mock_fair.call_args_list[1].args[3](test_true, test_pred) == pytest.approx(
        soft_prfs(test_true, test_pred)[2]
    )
    assert [
        c
        for c in mock_run.log_scalar.call_args_list
        if c.args[0].startswith("test/fair/")
    ] == [call("test/fair/poJSD", 0.445), call("test/fair/soft_f1", 0.554)]
