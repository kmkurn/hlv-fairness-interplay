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

import pytest
import torch.nn as nn
from classifier import TextSoftClassifier
from flair.data import Dictionary, Sentence
from flair.embeddings import TransformerDocumentEmbeddings


def test_correct_loss():
    label_type = "label"
    s = Sentence("foo")
    s.add_label(label_type, value="pos", score=1 / 3)
    s.add_label(label_type, value="neg", score=2 / 3)
    s.add_label(label_type, value="neu", score=0)
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("pos")
    label_dict.add_item("neg")
    label_dict.add_item("neu")
    clf = TextSoftClassifier(TransformerDocumentEmbeddings(), label_type, label_dictionary=label_dict)
    for p in clf.parameters():
        nn.init.zeros_(p)
    for i in range(len(label_dict)):
        nn.init.constant_(clf.decoder.bias[i], (i + 1) / 10)

    loss, _ = clf.forward_loss([s])

    assert loss.item() == pytest.approx(
        clf.decoder.bias.exp().sum().log().item() - 1 / 3 * 0.1 - 2 / 3 * 0.2
    )


def test_correct_count():
    label_type = "label"
    s = Sentence("foo")
    s.add_label(label_type, value="pos", score=1 / 3)
    s.add_label(label_type, value="neg", score=2 / 3)
    s.add_label(label_type, value="neu", score=0)
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("pos")
    label_dict.add_item("neg")
    label_dict.add_item("neu")
    clf = TextSoftClassifier(TransformerDocumentEmbeddings(), label_type, label_dictionary=label_dict)

    _, count = clf.forward_loss([s])

    assert count == 1
