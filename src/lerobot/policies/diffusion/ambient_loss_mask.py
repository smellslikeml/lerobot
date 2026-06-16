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
"""Noise-dependent loss masking for diffusion policies.

Adapted from "Ambient Diffusion Policy: Imitation Learning from Suboptimal
Data in Robotics" (https://arxiv.org/abs/2606.12365).

The paper observes that robot action data follows a spectral power law, which
gives the optimal Diffusion Policy a global-to-local hierarchy: high diffusion
times (large noise) carry coarse/global structure, low diffusion times (small
noise) carry fine/local detail. Suboptimal or out-of-distribution
demonstrations corrupt the *mid-frequency* band; their useful signal lives at
the extremes. Instead of co-training on every diffusion time for every sample,
Ambient Diffusion Policy introduces **noise-dependent data usage**: each sample
is only allowed to contribute to the denoising loss at the diffusion times
where its signal is trustworthy.

This module turns a per-sample scalar *quality* score in ``[0, 1]`` (1.0 =
clean/optimal, lower = more suboptimal) plus the per-sample diffusion
``timesteps`` into a per-sample weight in ``{0, 1}`` that multiplies into the
existing loss mask. A quality of ``1.0`` always yields a weight of ``1.0`` (the
sample participates at every diffusion time), so a batch of all-clean samples —
or no quality column at all — reproduces the standard Diffusion Policy loss
exactly.

Two masking modes are supported:

``"band"`` (default, faithful to the paper)
    A sample contributes only at the high *and* low extremes of the diffusion
    schedule, with the excluded mid-noise band widening as quality drops. This
    keeps a suboptimal sample's global structure (high noise) and its local
    detail (low noise) while discarding the harmful mid-frequency content.

``"high"``
    A sample contributes only at noise magnitudes above a per-sample threshold,
    suppressing its loss at low-noise (small-timestep) levels. This is the
    simpler one-sided variant for OOD demonstrations whose fine detail is
    untrustworthy but whose coarse structure is still useful.
"""

from __future__ import annotations

import torch
from torch import Tensor

AMBIENT_MASK_MODES = ("band", "high")


def normalize_quality(
    quality: Tensor | None,
    batch_size: int,
    device: torch.device | str,
) -> Tensor:
    """Coerce a per-sample quality input into a clean ``(batch_size,)`` float tensor.

    ``None`` (no quality column in the batch) is treated as "all samples are
    fully trustworthy" and maps to a tensor of ones, making the mask a no-op.
    Scalars are broadcast to the batch. Values are clamped to ``[0, 1]``.
    """
    if quality is None:
        return torch.ones(batch_size, device=device, dtype=torch.float32)

    if not isinstance(quality, Tensor):
        quality = torch.as_tensor(quality)
    quality = quality.to(device=device, dtype=torch.float32).reshape(-1)

    if quality.numel() == 1:
        quality = quality.expand(batch_size)
    elif quality.numel() != batch_size:
        raise ValueError(
            f"Ambient quality must be a scalar or have one entry per sample. "
            f"Got {quality.numel()} entries for a batch of {batch_size}."
        )
    return quality.clamp(0.0, 1.0)


def compute_ambient_loss_mask(
    timesteps: Tensor,
    quality: Tensor | None,
    num_train_timesteps: int,
    *,
    mode: str = "band",
) -> Tensor:
    """Per-sample noise-dependent loss weight for Ambient Diffusion Policy.

    Args:
        timesteps: ``(batch_size,)`` integer diffusion timesteps sampled for the
            batch (the same tensor passed to ``add_noise``).
        quality: ``(batch_size,)`` (or scalar, or ``None``) per-sample quality in
            ``[0, 1]``. Lower means more suboptimal. ``None`` disables masking.
        num_train_timesteps: Size of the forward diffusion schedule.
        mode: ``"band"`` (suppress the mid-noise band, the paper's mechanism) or
            ``"high"`` (suppress low-noise levels only).

    Returns:
        ``(batch_size,)`` float weight in ``{0.0, 1.0}``: ``1.0`` where the
        sample is allowed to contribute to the denoising loss at its timestep.
    """
    if mode not in AMBIENT_MASK_MODES:
        raise ValueError(f"`mode` must be one of {AMBIENT_MASK_MODES}. Got {mode!r}.")

    batch_size = timesteps.shape[0]
    quality = normalize_quality(quality, batch_size, timesteps.device)

    # Normalize timesteps to [0, 1] along the diffusion schedule.
    denom = max(num_train_timesteps - 1, 1)
    t_norm = timesteps.to(device=quality.device, dtype=torch.float32) / denom

    if mode == "high":
        # Contribute only above a per-sample noise threshold. quality=1 -> threshold 0
        # (everywhere); quality=0 -> threshold 1 (only the highest noise level).
        threshold = 1.0 - quality
        mask = (t_norm >= threshold).to(torch.float32)
    else:  # "band"
        # Distance from the middle of the schedule, in [0, 1] (0 at mid-noise, 1 at
        # the extremes). Contribute only in the outer region; the excluded band
        # widens as quality drops. quality=1 -> contribute everywhere.
        dist_from_mid = (2.0 * (t_norm - 0.5)).abs()
        mask = (dist_from_mid >= (1.0 - quality)).to(torch.float32)

    return mask
