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
"""Policy-paced online/offline data mixing for real-robot RL.

Adapts the *Policy-Paced Learning* (PPL) idea from WorldSample ("WorldSample:
Closed-loop Real-robot RL with World Modelling") to LeRobot's distributed
HIL-SERL training loop.

WorldSample closes a real/synthetic loop: physical rollouts are augmented with
synthetic transitions produced by a world model, and PPL *regulates* that mix
through sample selection and scheduling so that useful augmentation is balanced
against value overestimation. LeRobot has no world-model rollout generator, but
its :class:`OnlineOfflineMixer` already mixes two structurally analogous sources
-- live online rollouts (the "real" side) and a fixed demonstration buffer (a
secondary, augmenting dataset that extends coverage beyond what online
collection provides) -- at a *fixed* ratio.

:class:`PolicyPacedMixer` keeps PPL's core mechanism -- a *policy-paced*
schedule that moves the online fraction over training instead of holding it
constant -- while substituting the two components LeRobot cannot host:

* World-model synthetic transitions -> the offline demonstration buffer (the
  target-native secondary/augmentation source).
* PPL's learned value-overestimation regulator -> a parameter-free progress
  schedule (linear/cosine), with an optional ``ratio_modulator`` callable hook
  that an algorithm with a critic can use to down-weight augmentation when
  overestimation rises. The mixer itself stays algorithm-agnostic.

Qualitatively the policy sees more demonstration data early (when it is immature
and online coverage is low) and weans toward online data as it matures -- the
regulation PPL prescribes.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from lerobot.types import BatchType

from ..buffer import ReplayBuffer
from .data_mixer import DataMixer, OnlineOfflineMixer

# A schedule maps training progress in [0, 1] to a shape value in [0, 1]; the
# mixer interpolates the online ratio between ``start_ratio`` and ``end_ratio``
# along that shape. A ratio modulator optionally adjusts the scheduled ratio
# (e.g. to suppress augmentation when a critic overestimates value).
RatioSchedule = Callable[[float], float]
RatioModulator = Callable[[float, int], float]


def _linear_shape(progress: float) -> float:
    return progress


def _cosine_shape(progress: float) -> float:
    return 0.5 * (1.0 - math.cos(math.pi * progress))


_SCHEDULES: dict[str, RatioSchedule] = {
    "linear": _linear_shape,
    "cosine": _cosine_shape,
}


def _resolve_schedule(schedule: str | RatioSchedule) -> RatioSchedule:
    if callable(schedule):
        return schedule
    try:
        return _SCHEDULES[schedule]
    except KeyError as error:
        raise ValueError(
            f"Unknown online_ratio_schedule {schedule!r}; expected one of {sorted(_SCHEDULES)} or a callable."
        ) from error


def _clamp_ratio(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


class PolicyPacedMixer(DataMixer):
    """Online/offline mixer whose online fraction is paced across training.

    Wraps an :class:`OnlineOfflineMixer` and rewrites its ``online_ratio`` on
    every draw according to a schedule, instead of holding it fixed.

    Args:
        online_buffer: Buffer of live (online) rollouts.
        offline_buffer: Buffer of demonstration / augmenting transitions. When
            ``None``, sampling is online-only and the schedule has no effect.
        schedule: Named schedule (``"linear"`` or ``"cosine"``) or a callable
            mapping progress in ``[0, 1]`` to a shape in ``[0, 1]``.
        total_steps: Number of draws over which the ratio anneals from
            ``start_ratio`` to ``end_ratio``. The ratio is clamped at
            ``end_ratio`` afterwards.
        start_ratio: Online fraction at progress 0.
        end_ratio: Online fraction at progress >= 1.
        ratio_modulator: Optional ``modulator(ratio, step) -> ratio`` to adjust
            the scheduled ratio -- e.g. from a value-overestimation signal.
            ``None`` keeps a pure parameter-free progress schedule.

    Note:
        Batches are drawn serially through :meth:`sample` (inheriting the base
        :class:`DataMixer` iterator) so the ratio can be updated every draw;
        the async-prefetch path of :class:`OnlineOfflineMixer` is not used.
    """

    def __init__(
        self,
        online_buffer: ReplayBuffer,
        offline_buffer: ReplayBuffer | None = None,
        *,
        schedule: str | RatioSchedule = "linear",
        total_steps: int = 1,
        start_ratio: float = 0.5,
        end_ratio: float = 1.0,
        ratio_modulator: RatioModulator | None = None,
    ) -> None:
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        self._schedule = _resolve_schedule(schedule)
        self._total_steps = int(total_steps)
        self._start_ratio = _clamp_ratio(start_ratio)
        self._end_ratio = _clamp_ratio(end_ratio)
        self._modulator = ratio_modulator
        self._step = 0
        self._inner = OnlineOfflineMixer(
            online_buffer=online_buffer,
            offline_buffer=offline_buffer,
            online_ratio=self._current_ratio(),
        )

    @property
    def online_ratio(self) -> float:
        """Effective online fraction used for the most recent draw."""
        return self._inner.online_ratio

    @property
    def step(self) -> int:
        """Number of batches drawn so far (the policy-update counter)."""
        return self._step

    def _current_ratio(self) -> float:
        progress = min(self._step / self._total_steps, 1.0)
        ratio = self._start_ratio + (self._end_ratio - self._start_ratio) * self._schedule(progress)
        ratio = _clamp_ratio(ratio)
        if self._modulator is not None:
            ratio = _clamp_ratio(self._modulator(ratio, self._step))
        return ratio

    def sample(self, batch_size: int) -> BatchType:
        self._inner.online_ratio = self._current_ratio()
        batch = self._inner.sample(batch_size)
        self._step += 1
        return batch
