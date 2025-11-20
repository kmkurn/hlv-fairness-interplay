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
import math

import torch
from hlv_loss import JSDLoss


def test_multiclass_correct():
    torch.manual_seed(0)
    p0, p1 = 0.2, 0.3
    p2 = 1 - (p0 + p1)
    q0, q1 = 0.1, 0.7
    q2 = 1 - (q0 + q1)
    targets = torch.tensor([[p0, p1, p2]])
    preds = torch.tensor([[q0, q1, q2]])
    m0 = (p0 + q0) / 2
    m1 = (p1 + q1) / 2
    m2 = (p2 + q2) / 2
    kl_p_m = p0 * math.log2(p0 / m0) + p1 * math.log2(p1 / m1) + p2 * math.log2(p2 / m2)
    kl_q_m = q0 * math.log2(q0 / m0) + q1 * math.log2(q1 / m1) + q2 * math.log2(q2 / m2)
    expected = (kl_p_m + kl_q_m) / 2
    logits = torch.rand([1, 3])

    with patch.object(
        logits, "softmax", autospec=True, return_value=preds
    ) as fake_softmax:
        loss = JSDLoss()(logits, targets)

    torch.testing.assert_close(loss, torch.tensor(expected))
    fake_softmax.assert_called_once_with(dim=-1)
