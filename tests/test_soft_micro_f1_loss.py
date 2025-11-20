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

from unittest.mock import patch

import pytest
import torch
from hlv_loss import SoftMicroF1Loss


def test_with_softmax_activation():
    torch.manual_seed(0)
    logits = torch.rand([2, 3])
    preds = torch.tensor([[0.1, 0.5, 0.4], [0.3, 0.6, 0.1]])
    targets = torch.tensor([[0.2, 0.4, 0.4], [0.1, 0.1, 0.8]])
    soft_accuracy = 0.5 * (0.1 + 0.4 + 0.4 + 0.1 + 0.1 + 0.1)

    with patch.object(
        logits, "softmax", autospec=True, return_value=preds
    ) as fake_softmax:
        loss = SoftMicroF1Loss()(logits, targets)

    torch.testing.assert_close(loss, torch.tensor(1 - soft_accuracy))
    fake_softmax.assert_called_once_with(dim=-1)


def test_warn_invalid_targets():
    loss_fn = SoftMicroF1Loss()
    with pytest.warns(UserWarning):
        loss_fn(torch.tensor([-0.03]), torch.tensor([1.2]))
    with pytest.warns(UserWarning):
        loss_fn(torch.tensor([-0.03]), torch.tensor([-0.2]))
