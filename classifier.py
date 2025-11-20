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

import flair
import torch
from flair.data import Sentence
from flair.models import TextClassifier


class TextSoftClassifier(TextClassifier):
    def _prepare_label_tensor(
        self, prediction_data_points: list[Sentence]
    ) -> torch.Tensor:
        dps = prediction_data_points
        label_tensor = torch.empty(
            [len(dps), len(self.label_dictionary)], device=flair.device
        )
        for i, dp in enumerate(dps):
            for label in dp.get_labels(self.label_type):
                j = self.label_dictionary.get_idx_for_item(label.value)
                label_tensor[i, j] = label.score
        return label_tensor
