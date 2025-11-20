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

Arr = np.ndarray


def poJSD(y_true: Arr, y_pred: Arr, classwise: bool = False) -> float | Arr:
    if classwise:  # classwise is equal to multilabel
        y_true = np.stack([y_true, 1 - y_true], axis=-1)
        y_pred = np.stack([y_pred, 1 - y_pred], axis=-1)
    y_mean = 0.5 * (y_true + y_pred)
    y_true_div_mean = np.divide(
        y_true, y_mean, out=np.zeros_like(y_true), where=~np.isclose(y_mean, 0)
    )
    y_pred_div_mean = np.divide(
        y_pred, y_mean, out=np.zeros_like(y_pred), where=~np.isclose(y_mean, 0)
    )
    kl_true_mean = (y_true * _log_or_zero(y_true_div_mean, np.log2)).sum(axis=-1)
    kl_pred_mean = (y_pred * _log_or_zero(y_pred_div_mean, np.log2)).sum(axis=-1)
    return 1 - (0.5 * (kl_true_mean + kl_pred_mean)).mean(axis=0 if classwise else None)


def soft_accuracy(y_true: Arr, y_pred: Arr) -> float:
    return np.where(y_true < y_pred, y_true, y_pred).sum(axis=-1).mean()


def soft_precision_recall_fscore_support(
    y_true: Arr, y_pred: Arr
) -> tuple[Arr, Arr, Arr, Arr]:
    min_sum = np.where(y_true < y_pred, y_true, y_pred).sum(axis=0)
    pred_sum = y_pred.sum(axis=0)
    true_sum = y_true.sum(axis=0)
    Ps = min_sum / pred_sum
    Rs = min_sum / true_sum
    Fs = 2 * min_sum / (pred_sum + true_sum)
    return Ps, Rs, Fs, true_sum


def _log_or_zero(x: Arr, /, log_fn=np.log) -> Arr:
    atol = 1e-8
    return log_fn(x, out=np.zeros_like(x), where=x >= atol)
