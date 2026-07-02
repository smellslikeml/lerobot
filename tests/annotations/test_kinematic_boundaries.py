#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Kinematic boundary anchoring (InSight Stage-1) — unit + integration tests.

The integration tests drive the *existing* plan module
(:class:`PlanSubtasksMemoryModule`) end-to-end and assert that enabling
``snap_subtasks_to_kinematics`` moves VLM subtask boundaries onto the recorded
gripper/end-effector kinematic event, while the default (off) path is byte-for
-byte the prior purely-visual behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")

import pandas as pd  # noqa: E402

from lerobot.annotations.steerable_pipeline.config import PlanConfig  # noqa: E402
from lerobot.annotations.steerable_pipeline.kinematic_boundaries import (  # noqa: E402
    detect_kinematic_boundaries,
    snap_spans_to_kinematic_boundaries,
)
from lerobot.annotations.steerable_pipeline.modules import PlanSubtasksMemoryModule  # noqa: E402
from lerobot.annotations.steerable_pipeline.reader import iter_episodes  # noqa: E402
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging  # noqa: E402

from ._helpers import make_canned_responder  # noqa: E402

# A canned 3-subtask plan whose middle boundary (0.5s) sits just after the
# gripper closes (0.4s) and whose last boundary (0.9s) is far from any event.
_SUBTASKS = {
    "atomic subtasks": {
        "subtasks": [
            {"text": "move the gripper to the bottle", "start": 0.0, "end": 0.5},
            {"text": "grasp the bottle", "start": 0.5, "end": 0.9},
            {"text": "pour into the cup", "start": 0.9, "end": 1.1},
        ]
    },
    "compressed semantic memory": {"memory": "poured once"},
}


def _dataset_with_gripper(root: Path) -> Path:
    """Build a 12-frame episode whose ``observation.state`` gripper closes at 0.4s."""
    from tests.fixtures.dataset_factories import build_annotation_dataset

    build_annotation_dataset(
        root, episode_specs=[(0, 12, "Pour water from the bottle into the cup.")], fps=10
    )
    parquet = root / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(parquet)
    # Single-channel gripper: open (0.0) for frames 0-3, closed (1.0) from frame
    # 4 (=0.4s) on. The open->closed flip is the only kinematic boundary after
    # min-separation thinning.
    df["observation.state"] = [[0.0]] * 4 + [[1.0]] * 8
    df.to_parquet(parquet, index=False)
    return root


def _subtask_timestamps(rows: list[dict]) -> list[float]:
    return sorted(r["timestamp"] for r in rows if r["style"] == "subtask")


def _run_plan(root: Path, tmp_path: Path, config: PlanConfig) -> list[dict]:
    module = PlanSubtasksMemoryModule(vlm=make_canned_responder(_SUBTASKS), config=config)
    record = next(iter_episodes(root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    return staging.read("plan")


def test_detect_boundaries_from_gripper_transition(tmp_path: Path) -> None:
    root = _dataset_with_gripper(tmp_path / "ds")
    record = next(iter_episodes(root))
    boundaries = detect_kinematic_boundaries(
        record.frames_df(), record.frame_timestamps, state_keys=("observation.state",)
    )
    assert boundaries == [pytest.approx(0.4)]


def test_detect_boundaries_no_state_column_is_empty(fixture_dataset_root: Path) -> None:
    """The default annotation fixture has no state column -> safe no-op."""
    record = next(iter_episodes(fixture_dataset_root))
    assert detect_kinematic_boundaries(record.frames_df(), record.frame_timestamps) == []


def test_snap_pins_first_and_respects_tolerance() -> None:
    spans = [
        {"text": "a", "start": 0.0, "end": 0.5},
        {"text": "b", "start": 0.5, "end": 0.9},
        {"text": "c", "start": 0.9, "end": 1.1},
    ]
    snapped = snap_spans_to_kinematic_boundaries(spans, [0.4], tolerance_s=0.2)
    starts = [s["start"] for s in snapped]
    assert starts[0] == 0.0  # first pinned
    assert starts[1] == pytest.approx(0.4)  # within tolerance -> snapped
    assert starts[2] == 0.9  # 0.5 away from 0.4 -> left alone


def test_plan_module_snaps_subtasks_to_kinematics(tmp_path: Path) -> None:
    root = _dataset_with_gripper(tmp_path / "ds")
    rows = _run_plan(
        root,
        tmp_path,
        PlanConfig(snap_subtasks_to_kinematics=True, kinematic_snap_tolerance_s=0.2),
    )
    starts = _subtask_timestamps(rows)
    # boundary moved from the VLM's 0.5 onto the gripper-close frame 0.4
    assert any(s == pytest.approx(0.4) for s in starts)
    assert all(s != pytest.approx(0.5) for s in starts)
    # all subtask rows still land on exact source frames
    frame_set = set(next(iter_episodes(root)).frame_timestamps)
    assert all(s in frame_set for s in starts)


def test_plan_module_default_leaves_subtasks_unsnapped(tmp_path: Path) -> None:
    root = _dataset_with_gripper(tmp_path / "ds")
    rows = _run_plan(root, tmp_path, PlanConfig())  # snapping off by default
    starts = _subtask_timestamps(rows)
    # purely-visual behaviour: the VLM's 0.5 boundary survives, 0.4 is absent
    assert any(s == pytest.approx(0.5) for s in starts)
    assert all(s != pytest.approx(0.4) for s in starts)
