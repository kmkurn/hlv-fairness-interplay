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

import numpy as np
import pytest
from fair_eval import compute_for_multilabel_group
from perf_eval import soft_precision_recall_fscore_support


def test_multilabel_group_ok():
    rng = np.random.default_rng(seed=0)
    y_true, y_pred = rng.random(size=[2, 3, 2])
    group_membership = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)

    class MyEvalFunc:
        results = np.array([[0.3, 0.1], [0.2, 0.9], [0.2, 0.2], [0.8, 0.7]])

        def __init__(self):
            self.call_count = 0

        def __call__(self, y_true_, y_pred_):
            if self.call_count == 0:
                assert np.allclose(y_true_, y_true[[0, 2]])
                assert np.allclose(y_pred_, y_pred[[0, 2]])
            elif self.call_count == 1:
                assert np.allclose(y_true_, y_true[[1]])
                assert np.allclose(y_pred_, y_pred[[1]])
            elif self.call_count == 2:
                assert np.allclose(y_true_, y_true[[1, 2]])
                assert np.allclose(y_pred_, y_pred[[1, 2]])
            else:
                assert self.call_count == 3
                assert np.allclose(y_true_, y_true[[0]])
                assert np.allclose(y_pred_, y_pred[[0]])

            res = self.results[self.call_count]
            self.call_count += 1
            return res

    expected = (0.2 / 0.3 + 0.2 / 0.8) / 2 * (0.1 / 0.9 + 0.2 / 0.7) / 2

    res = compute_for_multilabel_group(
        y_true,
        y_pred,
        group_membership,
        MyEvalFunc(),
        group_agg=lambda s: s.mean(axis=-1),
        class_agg=lambda s: s.prod(axis=-1),
    )

    assert res == pytest.approx(expected)


def test_multilabel_group_default_group_agg():
    rng = np.random.default_rng(seed=0)
    y_true, y_pred = rng.random(size=[2, 3, 2])
    group_membership = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)
    expected = compute_for_multilabel_group(
        y_true,
        y_pred,
        group_membership,
        lambda yt, yp: soft_precision_recall_fscore_support(yt, yp)[2],
        group_agg=lambda s: (s**0.5).prod(axis=-1),
        class_agg=lambda s: s.prod(axis=-1),
    )

    res = compute_for_multilabel_group(
        y_true,
        y_pred,
        group_membership,
        lambda yt, yp: soft_precision_recall_fscore_support(yt, yp)[2],
        class_agg=lambda s: s.prod(axis=-1),
    )

    assert res == pytest.approx(expected)


def test_multilabel_group_default_class_agg():
    rng = np.random.default_rng(seed=0)
    y_true, y_pred = rng.random(size=[2, 3, 2])
    group_membership = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)
    expected = compute_for_multilabel_group(
        y_true,
        y_pred,
        group_membership,
        lambda yt, yp: soft_precision_recall_fscore_support(yt, yp)[2],
        group_agg=lambda s: s.mean(axis=-1),
        class_agg=lambda s: s.mean(axis=-1),
    )

    res = compute_for_multilabel_group(
        y_true,
        y_pred,
        group_membership,
        lambda yt, yp: soft_precision_recall_fscore_support(yt, yp)[2],
        group_agg=lambda s: s.mean(axis=-1),
    )

    assert res == pytest.approx(expected)


def test_multilabel_group_zero_perf():
    rng = np.random.default_rng(seed=0)
    y_true, y_pred = rng.random(size=[2, 3, 5])
    group_membership = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)

    class MyEvalFunc:
        results = rng.random(size=[4, 5])

        def __init__(self):
            self.call_count = 0
            self.results[1] = 0

        def __call__(self, y_true_, y_pred_):
            res = self.results[self.call_count]
            self.call_count += 1
            return res

    res = compute_for_multilabel_group(y_true, y_pred, group_membership, MyEvalFunc())

    assert np.isfinite(res)
