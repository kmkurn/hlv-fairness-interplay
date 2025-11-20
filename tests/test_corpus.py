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

import csv
import random
import textwrap

import pytest
from corpus import SBICorpus


@pytest.fixture
def write_sbic_data(tmp_path):
    # Columns are separated by tabs
    content = textwrap.dedent(
        """
    offensiveYN	annotatorGender	annotatorMinority	WorkerId	annotatorPolitics	annotatorRace	annotatorAge	post	targetMinority	targetCategory	dataSource
    1.0			women			women				0			P1					R1				35				foo		M1				C1				t/xxx
    1.0			women			women				1			P2					R2				40				foo		M1,M2			C1				t/xxx
    1.0			men				men					2			P1					R2				27				foo		M3				C2				t/xxx
    1.0			men				men					2			P1					R2				27				foo		M3				C2				t/xxx
    0.0			women			women				3			P2					R1				20				foo		M4				C2				t/xxx
    0.0			men				men					4			P1					R1				37				foo		M1,M2			C1				t/xxx
    0.5			men				men					5			P2					R2				47				foo		M2				C1				t/xxx
    1.0			women			women				1			P2					R2				40				bar		M3				C2				t/xxx
    1.0			men				men					2			P1					R2				27				bar		M4				C2				t/xxx
    0.0			women			women				3			P2					R1				20				bar		M1				C2				t/xxx
    """
    ).strip()
    lines = content.splitlines()
    header = lines.pop(0)  # prevent header from being shuffled
    random.seed(0)
    random.shuffle(lines)
    lines.insert(0, header)

    def _write_sbic_data(split):
        suffix = {"train": "trn", "dev": "dev", "test": "tst"}[split]
        sbic_dir = tmp_path / "sbic"
        sbic_dir.mkdir()
        with (sbic_dir / f"SBIC.v2.{suffix}.csv").open("w", encoding="utf8") as f:
            csv.writer(f).writerows(l.split() for l in lines)
        return sbic_dir

    return _write_sbic_data


@pytest.mark.parametrize("split", ["dev", "test"])
def test_correct_eval_sets(tmp_path, write_sbic_data, split):
    sbic_dir = write_sbic_data(split)
    corpus = SBICorpus(sbic_dir)

    assert {
        s.text: (
            {l.value: l.score for l in s.get_labels("offensiveness")},
            s.get_metadata("annotator_ids"),
            s.get_metadata("target_cats"),
        )
        for s in (corpus.dev if split == "dev" else corpus.test)
    } == {
        "foo": (
            pytest.approx(
                {
                    "yes": 3 / 6,
                    "no": 2 / 6,
                    "maybe": 1 / 6,
                }
            ),
            set(range(6)),
            {"C1", "C2"},
        ),
        "bar": (
            pytest.approx(
                {
                    "yes": 2 / 3,
                    "no": 1 / 3,
                    "maybe": 0,
                }
            ),
            {1, 2, 3},
            {"C2"},
        ),
    }


def test_disagg_train_set(tmp_path, write_sbic_data):
    sbic_dir = write_sbic_data(split="train")
    corpus = SBICorpus(sbic_dir)

    assert {
        (s.text, s.get_label("offensiveness").value, s.get_metadata("annotator_id"))
        for s in corpus.train
    } == {
        ("foo", "yes", 0),
        ("foo", "yes", 1),
        ("foo", "yes", 2),
        ("foo", "no", 3),
        ("foo", "no", 4),
        ("foo", "maybe", 5),
        ("bar", "yes", 1),
        ("bar", "yes", 2),
        ("bar", "no", 3),
    }


def test_agg_train_set(tmp_path, write_sbic_data):
    def dummy_agg(ss):
        new_ss = []
        for s in ss:
            s.set_label("offensiveness", "xx")
            new_ss.append(s)
        return new_ss

    sbic_dir = write_sbic_data(split="train")
    corpus = SBICorpus(sbic_dir, train_agg=dummy_agg)

    assert {
        (s.text, s.get_label("offensiveness").value, s.get_metadata("annotator_id"))
        for s in corpus.train
    } == {
        ("foo", "xx", 0),
        ("foo", "xx", 1),
        ("foo", "xx", 2),
        ("foo", "xx", 3),
        ("foo", "xx", 4),
        ("foo", "xx", 5),
        ("bar", "xx", 1),
        ("bar", "xx", 2),
        ("bar", "xx", 3),
    }


def test_missing_target_cat_in_dev_set(tmp_path):
    content = textwrap.dedent(
        """
    offensiveYN,annotatorGender,annotatorMinority,WorkerId,annotatorPolitics,annotatorRace,annotatorAge,post,targetMinority,targetCategory,dataSource
    1.0,women,women,0,P1,R1,35,foo,M1,,t/xxx
        """
    )
    sbic_dir = tmp_path / "sbic"
    sbic_dir.mkdir()
    (sbic_dir / "SBIC.v2.dev.csv").write_text(content, encoding="utf8")

    corpus = SBICorpus(sbic_dir)

    assert corpus.dev[0].get_metadata("target_cats") == set()


def test_missing_target_cat_in_test_set(tmp_path):
    content = textwrap.dedent(
        """
    offensiveYN,annotatorGender,annotatorMinority,WorkerId,annotatorPolitics,annotatorRace,annotatorAge,post,targetMinority,targetCategory,dataSource
    1.0,women,women,0,P1,R1,35,foo,M1,,t/xxx
        """
    )
    sbic_dir = tmp_path / "sbic"
    sbic_dir.mkdir()
    (sbic_dir / "SBIC.v2.tst.csv").write_text(content, encoding="utf8")

    corpus = SBICorpus(sbic_dir)

    assert corpus.test[0].get_metadata("target_cats") == set()


# TODO reduce duplication in downsample tests below
def test_downsample_train_set(tmp_path):
    content = textwrap.dedent(
        """
    offensiveYN,annotatorGender,annotatorMinority,WorkerId,annotatorPolitics,annotatorRace,annotatorAge,post,targetMinority,targetCategory,dataSource
    1.0,women,women,0,P1,R1,35,foo,M1,,t/xxx
    1.0,women,women,0,P1,R1,35,foo,M1,,t/xxx
    1.0,women,women,0,P1,R1,35,foo,M1,,t/xxx
    1.0,women,women,0,P1,R1,35,foo,M1,,t/xxx
    1.0,women,women,0,P1,R1,35,foo,M1,,t/xxx
        """
    )
    sbic_dir = tmp_path / "sbic"
    sbic_dir.mkdir()
    (sbic_dir / "SBIC.v2.trn.csv").write_text(content, encoding="utf8")

    corpus = SBICorpus(sbic_dir, train_sample=0.2, seed=0)

    assert len(corpus.train) == 1


def test_downsample_dev_set(tmp_path):
    content = textwrap.dedent(
        """
    offensiveYN,annotatorGender,annotatorMinority,WorkerId,annotatorPolitics,annotatorRace,annotatorAge,post,targetMinority,targetCategory,dataSource
    1.0,women,women,0,P1,R1,35,foo1,M1,,t/xxx
    1.0,women,women,0,P1,R1,35,foo2,M1,,t/xxx
    1.0,women,women,0,P1,R1,35,foo3,M1,,t/xxx
    1.0,women,women,0,P1,R1,35,foo4,M1,,t/xxx
    1.0,women,women,0,P1,R1,35,foo5,M1,,t/xxx
        """
    )
    sbic_dir = tmp_path / "sbic"
    sbic_dir.mkdir()
    (sbic_dir / "SBIC.v2.dev.csv").write_text(content, encoding="utf8")

    corpus = SBICorpus(sbic_dir, dev_sample=0.2, seed=0)

    assert len(corpus.dev) == 1


def test_downsample_test_set(tmp_path):
    content = textwrap.dedent(
        """
    offensiveYN,annotatorGender,annotatorMinority,WorkerId,annotatorPolitics,annotatorRace,annotatorAge,post,targetMinority,targetCategory,dataSource
    1.0,women,women,0,P1,R1,35,foo1,M1,,t/xxx
    1.0,women,women,0,P1,R1,35,foo2,M1,,t/xxx
    1.0,women,women,0,P1,R1,35,foo3,M1,,t/xxx
    1.0,women,women,0,P1,R1,35,foo4,M1,,t/xxx
    1.0,women,women,0,P1,R1,35,foo5,M1,,t/xxx
        """
    )
    sbic_dir = tmp_path / "sbic"
    sbic_dir.mkdir()
    (sbic_dir / "SBIC.v2.tst.csv").write_text(content, encoding="utf8")

    corpus = SBICorpus(sbic_dir, test_sample=0.2, seed=0)

    assert len(corpus.test) == 1
