#!/usr/bin/env python

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

import argparse
from pathlib import Path

import numpy as np

from fair_eval import compute_scores_for_multilabel_group
from perf_eval import poJSD
from perf_eval import soft_precision_recall_fscore_support as soft_prfs


def main(artifacts_dir: Path) -> None:
    loaded = np.load(artifacts_dir / "test.npz")
    membership = np.load(artifacts_dir / "membership.npy")
    scores_pojsd = compute_scores_for_multilabel_group(loaded["true"], loaded["pred"], membership, lambda yt, yp: poJSD(yt, yp, classwise=True))  # type: ignore[arg-type,return-value]
    scores_sf1 = compute_scores_for_multilabel_group(
        loaded["true"], loaded["pred"], membership, lambda yt, yp: soft_prfs(yt, yp)[2]
    )
    np.savez_compressed(
        artifacts_dir / "fair-scores.npz", poJSD=scores_pojsd, soft_f1=scores_sf1
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compute class-wise fairness scores for multilabel groups using saved predictions"
    )
    p.add_argument(
        "artifacts_dir",
        type=Path,
        help="load predictions and save results in this directory",
    )
    args = p.parse_args()
    main(args.artifacts_dir)
