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

from flair.data import Sentence

from ann_agg import MajVoteAgg


def test_correct():
    agg = MajVoteAgg(typename="label")
    s1 = Sentence("foo")
    s1.add_label(typename="label", value="yes")
    s1.add_metadata("annotator_id", "A")
    s2 = Sentence("foo")
    s2.add_label(typename="label", value="yes")
    s2.add_metadata("annotator_id", "B")
    s3 = Sentence("foo")
    s3.add_label(typename="label", value="no")
    s3.add_metadata("annotator_id", "C")
    s3a = Sentence("foo")
    s3a.add_label(typename="label", value="no")
    s3a.add_metadata("annotator_id", "C")
    s3b = Sentence("foo")
    s3b.add_label(typename="label", value="no")
    s3b.add_metadata("annotator_id", "C")
    s4 = Sentence("bar")
    s4.add_label(typename="label", value="no")
    s4.add_metadata("annotator_id", "A")
    s5 = Sentence("bar")
    s5.add_label(typename="label", value="yes")
    s5.add_metadata("annotator_id", "B")
    s6 = Sentence("bar")
    s6.add_label(typename="label", value="no")
    s6.add_metadata("annotator_id", "C")
    ss = [s6, s4, s3, s1, s5, s3b, s2, s3a]

    agg_ss = agg(ss)

    assert {s.text: s.get_label("label").value for s in agg_ss} == {
        "foo": "yes",
        "bar": "no",
    }
