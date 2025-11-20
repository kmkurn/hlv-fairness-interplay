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

import warnings

import numpy as np
import pytest
from perf_eval import poJSD, soft_accuracy, soft_precision_recall_fscore_support
from scipy.stats import dirichlet


def test_pojsd_correct():
    p, q = 0.2, 0.6
    res = poJSD(np.array([[p, 1 - p]]), np.array([[q, 1 - q]]))

    assert res == pytest.approx(0.8754887502)


def test_classwise_pojsd_correct():
    y_true = np.array([[0.3, 0.1, 0.6]])
    y_pred = np.array([[0.7, 0.1, 0.2]])
    expected = np.array(
        [
            poJSD(np.array([[0.3, 0.7]]), np.array([[0.7, 0.3]])),
            poJSD(np.array([[0.1, 0.9]]), np.array([[0.1, 0.9]])),
            poJSD(np.array([[0.6, 0.4]]), np.array([[0.2, 0.8]])),
        ]
    )
    res = poJSD(y_true, y_pred, classwise=True)

    assert res == pytest.approx(expected)


def test_soft_acc_correct():
    p, q = 0.2, 0.6
    res = soft_accuracy(np.array([[p, 1 - p]]), np.array([[q, 1 - q]]))

    assert res == pytest.approx(0.6)


def test_soft_prfs_correct():
    ps = np.array([0.2, 0.3, 0.3, 0.9])
    qs = np.array([0.4, 0.1, 0.7, 0.8])

    Ps, Rs, Fs, Ns = soft_precision_recall_fscore_support(
        make_bindist(ps), make_bindist(qs)
    )

    assert Ps == pytest.approx([1.4 / 2.0, 1.7 / 2.0])
    assert Rs == pytest.approx([1.4 / 1.7, 1.7 / 2.3])
    assert Fs == pytest.approx([2 * 1.4 / 3.7, 2 * 1.7 / 4.3])
    assert Ns == pytest.approx([1.7, 2.3])


num_repeats = 100
sprf_fns = [
    lambda p, q: soft_precision_recall_fscore_support(p, q)[i] for i in range(3)
]


@pytest.mark.repeat(num_repeats)
@pytest.mark.parametrize("n_samples", [2, 5000])
@pytest.mark.parametrize(
    "eval_fn,lower,upper",
    list(
        zip(
            [poJSD, soft_accuracy, lambda p, q: poJSD(p, q, classwise=True), *sprf_fns],
            [0] * 6,
            [1, poJSD, 1, 1, 1],
        )
    ),
)
def test_bounded(dirichlet_rv, n_samples, eval_fn, lower, upper):
    p, q = dirichlet_rv.rvs(size=[2, n_samples])
    if callable(upper):
        upper = upper(p, q)
    res = eval_fn(p, q)
    if isinstance(res, np.ndarray):
        assert ((lower <= res) & (res <= upper)).all()
    else:
        assert lower <= res <= upper


@pytest.mark.repeat(num_repeats)
@pytest.mark.parametrize("n_samples", [2, 5000])
@pytest.mark.parametrize(
    "eval_fn", [poJSD, soft_accuracy, lambda p, q: poJSD(p, q, classwise=True), *sprf_fns]
)
def test_oracle(dirichlet_rv, n_samples, eval_fn):
    p = dirichlet_rv.rvs(n_samples)
    assert eval_fn(p, p) == pytest.approx(1.0)


@pytest.mark.parametrize("classwise", [False, True])
def test_pojsd_zero_prob(classwise):
    p1, q1 = np.array([1.0, 0.3]), np.array([0.3, 0.4])
    p2, q2 = np.array([0.4, 0.9]), np.array([0.7, 1.0])
    p3, q3 = np.array([1.0, 0.9]), np.array([1.0, 0.3])

    def pojsd(p, q):
        p_, q_ = make_bindist(p), make_bindist(q)
        return poJSD(p_, q_, classwise)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res1 = pojsd(p1, q1)
        res2 = pojsd(p2, q2)
        res3 = pojsd(p3, q3)

    assert not np.isnan(res1).all()
    assert not np.isnan(res2).all()
    assert not np.isnan(res3).all()


@pytest.fixture(scope="module", params=[2, 3])
def dirichlet_rv(request):
    return dirichlet(np.full(request.param, fill_value=1.0), seed=0)


def make_bindist(ps: np.ndarray) -> np.ndarray:
    return np.stack([ps, 1 - ps], axis=1)
