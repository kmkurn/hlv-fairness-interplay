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

from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
from flair.data import Sentence


class MajVoteAgg:
    def __init__(self, typename: str) -> None:
        self.typename = typename

    def __call__(self, ss: list[Sentence], /) -> list[Sentence]:
        text2sents = defaultdict(list)
        for s in ss:
            text2sents[s.text].append(s)
        new_ss = []
        for sents in text2sents.values():
            label2count = Counter(
                {
                    s.get_metadata("annotator_id"): s.get_label(self.typename).value
                    for s in sents
                }.values()
            )
            assert sents
            s = sents[0]
            s.remove_labels(self.typename)
            assert label2count
            mv_label, _ = label2count.most_common(n=1)[0]
            s.add_label(self.typename, mv_label)
            new_ss.append(s)
        return new_ss


class SoftLabelAgg:
    def __init__(
        self, typename: str, labels: Sequence[str], alpha: float = 1.0
    ) -> None:
        """Aggregation into soft labels, i.e. probability distribution over labels.

        The probability of a label is proportional to its count raised to the power of `alpha`.

        Args:
            typename: Label type name.
            labels: Sequence of labels, i.e. label vocabulary.
            alpha: Non-negative scalar regulating the entropy of the label distribution. Small
                values mean higher entropy and vice versa.
        """
        self.typename = typename
        self.labels = labels
        self.alpha = alpha

    def __call__(self, ss: list[Sentence], /) -> list[Sentence]:
        text2sents = defaultdict(list)
        for s in ss:
            text2sents[s.text].append(s)
        new_ss = []
        for sents in text2sents.values():
            label2count = Counter(
                {
                    s.get_metadata("annotator_id"): s.get_label(self.typename).value
                    for s in sents
                }.values()
            )
            assert sents
            s = sents[0]
            s.remove_labels(self.typename)
            ps = np.array(
                [label2count[self.labels[i]] for i in range(len(self.labels))],
                dtype=float,
            )
            ps **= self.alpha
            ps /= ps.sum()
            for i in range(len(self.labels)):
                s.add_label(self.typename, self.labels[i], float(ps[i]))
            new_ss.append(s)
        return new_ss
