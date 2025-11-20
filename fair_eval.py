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

from typing import Callable, Optional

import numpy as np

Arr = np.ndarray


def compute_for_multilabel_group(
    y_true: Arr,
    y_pred: Arr,
    group_membership: Arr,
    classwise_perf_eval_fn: Callable[[Arr, Arr], Arr],
    group_agg: Optional[Callable[[Arr], Arr]] = None,
    class_agg: Optional[Callable[[Arr], Arr]] = None,
) -> float:
    if group_agg is None:
        group_agg = _default_group_agg
    if class_agg is None:
        class_agg = _default_class_agg
    scores = compute_scores_for_multilabel_group(
        y_true, y_pred, group_membership, classwise_perf_eval_fn
    )
    return float(class_agg(group_agg(scores)))


def compute_scores_for_multilabel_group(
    y_true: Arr,
    y_pred: Arr,
    group_membership: Arr,
    classwise_perf_eval_fn: Callable[[Arr, Arr], Arr],
) -> Arr:
    _, n_groups = group_membership.shape
    scores = []
    for gid in range(n_groups):
        is_member = group_membership[:, gid]
        res1 = classwise_perf_eval_fn(y_true[is_member], y_pred[is_member])
        res2 = classwise_perf_eval_fn(y_true[~is_member], y_pred[~is_member])
        scores.append(np.where(res1 < res2, res1 / res2, res2 / res1))
    return np.stack(scores, axis=-1)


def _default_group_agg(s: Arr) -> Arr:
    return (s ** (1 / s.shape[-1])).prod(axis=-1)


def _default_class_agg(s: Arr) -> Arr:
    return s.mean(axis=-1)
