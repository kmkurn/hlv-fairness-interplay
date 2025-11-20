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

import random
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from flair.data import Corpus, Sentence
from flair.datasets import FlairDatapointDataset


class SBICorpus(Corpus):
    labelset = "yes no maybe".split()
    _eps = 1e-7

    def __init__(
        self,
        data_dir: Path,
        train_agg: Optional[Callable[[list[Sentence]], list[Sentence]]] = None,
        train_sample: float = 1.0,
        dev_sample: float = 1.0,
        test_sample: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """Social bias inference corpus.

        Args:
            data_dir: Corpus data directory containing the CSV files.
            train_agg: Aggregation function to be applied to the training data.
            train_sample: Downsample the training data to this rate. Must be less than one to take effect.
            dev_sample: Downsample the development data to this rate. Must be less than one to take effect.
            test_sample: Downsample the test data to this rate. Must be less than one to take effect.
            seed: Random seed for the downsampling. Ignored if neither `train_sample`, `dev_sample`, nor
                `test_sample` is less than one.
        """
        if train_agg is None:
            train_agg = self._disagg

        self._train_agg = train_agg
        self._train_sample = train_sample
        self._dev_sample = dev_sample
        self._test_sample = test_sample
        if (
            train_sample < 1 - self._eps
            or dev_sample < 1 - self._eps
            or test_sample < 1 - self._eps
        ):
            random.seed(seed)

        kwargs = {}

        if (data_dir / "SBIC.v2.trn.csv").exists():
            df_trn = pd.read_csv(data_dir / "SBIC.v2.trn.csv", index_col=False)
            trn_dat = []
            for row in df_trn.itertuples():
                s = Sentence(row.post)
                s.add_label("offensiveness", self._to_label(row.offensiveYN))
                s.add_metadata("annotator_id", row.WorkerId)
                trn_dat.append(s)
            kwargs["train"] = self._preprocess_train(trn_dat)

        if (data_dir / "SBIC.v2.tst.csv").exists():
            df_tst = pd.read_csv(data_dir / "SBIC.v2.tst.csv", index_col=False)
            df_tst["targetCategory"] = df_tst["targetCategory"].fillna("")
            tst_dat = []
            for row in (
                df_tst.groupby("post")[["offensiveYN", "WorkerId", "targetCategory"]]
                .agg(lambda xs: xs.tolist())
                .itertuples()
            ):
                s = Sentence(row.Index)
                label2count = Counter(
                    self._to_label(x)
                    for x in dict(zip(row.WorkerId, row.offensiveYN)).values()
                )
                total = sum(label2count.values())
                for label in self.labelset:
                    s.add_label("offensiveness", label, label2count[label] / total)
                s.add_metadata("annotator_ids", set(row.WorkerId))
                s.add_metadata("target_cats", {x for x in row.targetCategory if x})
                tst_dat.append(s)
            kwargs["test"] = self._preprocess_test(tst_dat)

        if (data_dir / "SBIC.v2.dev.csv").exists():
            df_dev = pd.read_csv(data_dir / "SBIC.v2.dev.csv", index_col=False)
            df_dev["targetCategory"] = df_dev["targetCategory"].fillna("")
            dev_dat = []
            for row in (
                df_dev.groupby("post")[["offensiveYN", "WorkerId", "targetCategory"]]
                .agg(lambda xs: xs.tolist())
                .itertuples()
            ):
                s = Sentence(row.Index)
                label2count = Counter(
                    self._to_label(x)
                    for x in dict(zip(row.WorkerId, row.offensiveYN)).values()
                )
                total = sum(label2count.values())
                for label in self.labelset:
                    s.add_label("offensiveness", label, label2count[label] / total)
                s.add_metadata("annotator_ids", set(row.WorkerId))
                s.add_metadata("target_cats", {x for x in row.targetCategory if x})
                dev_dat.append(s)
            kwargs["dev"] = self._preprocess_dev(dev_dat)

        super().__init__(
            **{split: FlairDatapointDataset(dat) for split, dat in kwargs.items()},  # type: ignore[arg-type]
            sample_missing_splits=False,
        )

    # TODO extract common code
    def _preprocess_train(self, dat):
        dat = self._train_agg(dat)
        if self._train_sample < 1.0 - self._eps:
            dat = random.sample(dat, round(len(dat) * self._train_sample))
        return dat

    def _preprocess_dev(self, dat):
        if self._dev_sample < 1.0 - self._eps:
            dat = random.sample(dat, round(len(dat) * self._dev_sample))
        return dat

    def _preprocess_test(self, dat):
        if self._test_sample < 1.0 - self._eps:
            dat = random.sample(dat, round(len(dat) * self._test_sample))
        return dat

    @staticmethod
    def _to_label(x):
        if x > 0.5:
            return "yes"
        if x < 0.5:
            return "no"
        return "maybe"

    @staticmethod
    def _disagg(ss):
        return ss
