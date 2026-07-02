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
"""Kinematic anchoring of primitive-action boundaries.

Adapted from InSight (*Self-Guided Skill Acquisition via Steerable VLAs*,
https://arxiv.org/abs/2606.24884): InSight's Stage-1 segmentation partitions
demonstrations into labelled primitives by combining VLM plan decomposition
with **end-effector poses** — the kinematic trace is what makes the boundaries
land on real grasp/release/motion transitions rather than on the VLM's visual
guesswork.

The steerable pipeline already produces VLM subtask spans from contact sheets.
This module supplies the missing kinematic half: it reads the recorded
``observation.state`` / ``action`` trace for an episode and proposes boundary
timestamps where the world state actually changes — gripper open/close
transitions and end-effector motion pauses (velocity minima). The plan module
then snaps its VLM span starts to the nearest such boundary, so a primitive
like "grasp the handle" begins on the exact frame the gripper closes.

Only Stage-1 segmentation grounding is implemented here. InSight's Stage-2
online data flywheel (autonomous attempt + collection of missing primitives)
needs a policy-in-the-loop simulator this data-pipeline repo does not host and
is intentionally out of scope.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _stack_state_column(frames_df: Any, key: str, n_frames: int) -> np.ndarray | None:
    """Stack a per-frame vector column into a dense ``[T, D]`` float array.

    Returns ``None`` when the column is absent, ragged, or not numeric — the
    caller treats that as "no kinematic signal" and leaves the spans untouched.
    """
    if key not in getattr(frames_df, "columns", []):
        return None
    try:
        rows = [np.atleast_1d(np.asarray(v, dtype=float)).ravel() for v in frames_df[key].to_list()]
    except (TypeError, ValueError):
        return None
    if len(rows) != n_frames or n_frames == 0:
        return None
    width = rows[0].shape[0]
    if width == 0 or any(r.shape[0] != width for r in rows):
        return None
    arr = np.vstack(rows)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _gripper_transition_times(gripper: np.ndarray, frame_timestamps: Sequence[float]) -> list[float]:
    """Timestamps where the gripper crosses the middle of its observed range.

    The gripper channel is normalised to ``[0, 1]`` over the episode and
    binarised at ``0.5``; every open<->closed flip is a primitive boundary
    (grasp begins / release begins). A gripper that never moves yields nothing.
    """
    lo, hi = float(np.min(gripper)), float(np.max(gripper))
    if hi - lo < 1e-6:
        return []
    binary = ((gripper - lo) / (hi - lo) >= 0.5).astype(int)
    flips = np.nonzero(np.diff(binary) != 0)[0] + 1
    return [float(frame_timestamps[i]) for i in flips]


def _motion_pause_times(
    pose: np.ndarray, frame_timestamps: Sequence[float], speed_quantile: float
) -> list[float]:
    """Timestamps of end-effector motion pauses (local speed minima).

    Per-frame speed is the norm of the central difference of ``pose``. A frame
    is a pause boundary when its speed dips below its predecessor yet stays at
    or below its successor *and* sits in the lowest ``speed_quantile`` of the
    episode's speeds — i.e. the arm slowed to switch primitives, not just a
    mid-stroke wobble. The strict left comparison keeps a flat low-speed
    plateau from spawning a boundary at every frame.
    """
    if pose.shape[0] < 3:
        return []
    vel = np.gradient(pose, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    if not np.any(speed > 0):
        return []
    threshold = float(np.quantile(speed, speed_quantile))
    times: list[float] = []
    for i in range(1, len(speed) - 1):
        if speed[i] <= threshold and speed[i] < speed[i - 1] and speed[i] <= speed[i + 1]:
            times.append(float(frame_timestamps[i]))
    return times


def _thin(times: Sequence[float], min_separation_s: float) -> list[float]:
    """Drop boundaries closer than ``min_separation_s`` to a kept earlier one."""
    kept: list[float] = []
    for t in sorted(set(times)):
        if not kept or t - kept[-1] >= min_separation_s:
            kept.append(t)
    return kept


def detect_kinematic_boundaries(
    frames_df: Any,
    frame_timestamps: Sequence[float],
    *,
    state_keys: Sequence[str] = ("observation.state", "action"),
    gripper_index: int = -1,
    speed_quantile: float = 0.3,
    min_separation_s: float = 0.4,
) -> list[float]:
    """Propose primitive-action boundary timestamps from the kinematic trace.

    Walks ``state_keys`` in order and uses the first column that decodes into a
    dense ``[T, D]`` array matching ``frame_timestamps``. Boundaries are the
    union of gripper open/close transitions and end-effector motion pauses,
    thinned so none are closer than ``min_separation_s``. Returns ``[]`` (a
    safe no-op for the caller) when no usable state column is present.
    """
    ts = [float(t) for t in frame_timestamps]
    if len(ts) < 3:
        return []
    feats: np.ndarray | None = None
    for key in state_keys:
        feats = _stack_state_column(frames_df, key, len(ts))
        if feats is not None:
            break
    if feats is None:
        return []

    boundaries = list(_gripper_transition_times(feats[:, gripper_index], ts))
    boundaries += _motion_pause_times(feats, ts, speed_quantile)
    return _thin(boundaries, min_separation_s)


def snap_spans_to_kinematic_boundaries(
    spans: Sequence[dict[str, Any]],
    boundaries: Sequence[float],
    *,
    tolerance_s: float,
    pin_first: bool = True,
) -> list[dict[str, Any]]:
    """Snap each span ``start`` to the nearest kinematic boundary in range.

    A span start is moved to its nearest boundary only when that boundary is
    within ``tolerance_s`` — far-away starts keep the VLM's value, so the
    kinematics refine the cut points without overriding the VLM's segmentation.
    The first span's start is left at the episode origin when ``pin_first`` is
    set (the plan module always pins the first subtask to ``t0``). The returned
    spans are sorted by start; collisions are resolved downstream by the plan
    module's frame-dedupe + full-coverage stitch.
    """
    if not spans or not boundaries:
        return [dict(s) for s in spans]
    ordered = sorted(float(b) for b in boundaries)
    out: list[dict[str, Any]] = []
    for i, span in enumerate(spans):
        new = dict(span)
        if not (i == 0 and pin_first):
            start = float(span["start"])
            nearest = min(ordered, key=lambda b: abs(b - start))
            if abs(nearest - start) <= tolerance_s:
                new["start"] = nearest
        out.append(new)
    out.sort(key=lambda s: float(s["start"]))
    return out
