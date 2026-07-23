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
"""Tests for Ambient Diffusion Policy noise-dependent loss masking.

Covers both the standalone mask (`compute_ambient_loss_mask`) and its wiring
into the existing `DiffusionModel.compute_loss` call site, including the
backward-compat guarantee that a batch of fully clean samples reproduces the
vanilla Diffusion Policy loss exactly.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.ambient_loss_mask import compute_ambient_loss_mask
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

# ---------------------------------------------------------------------------
# Standalone mask behavior
# ---------------------------------------------------------------------------


def test_none_quality_is_noop():
    timesteps = torch.tensor([0, 25, 50, 75, 99])
    mask = compute_ambient_loss_mask(timesteps, None, num_train_timesteps=100)
    assert torch.equal(mask, torch.ones(5))


def test_full_quality_contributes_at_every_timestep():
    timesteps = torch.arange(100)
    quality = torch.ones(100)
    for mode in ("band", "high"):
        mask = compute_ambient_loss_mask(timesteps, quality, num_train_timesteps=100, mode=mode)
        assert torch.equal(mask, torch.ones(100)), mode


def test_high_mode_suppresses_low_noise():
    # quality=0.5 -> threshold 0.5 -> only timesteps in the upper half contribute.
    timesteps = torch.tensor([0, 49, 50, 99])
    quality = torch.full((4,), 0.5)
    mask = compute_ambient_loss_mask(timesteps, quality, num_train_timesteps=100, mode="high")
    assert torch.equal(mask, torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_band_mode_suppresses_mid_noise_keeps_extremes():
    # quality=0.5 -> excluded band is the inner half; only the outer quartiles contribute.
    timesteps = torch.tensor([0, 24, 50, 75, 99])
    quality = torch.full((5,), 0.5)
    mask = compute_ambient_loss_mask(timesteps, quality, num_train_timesteps=100, mode="band")
    # Low (0) and high (99) extremes survive; the mid-noise samples are suppressed.
    assert mask[0] == 1.0
    assert mask[-1] == 1.0
    assert mask[2] == 0.0


def test_scalar_quality_is_broadcast():
    timesteps = torch.tensor([10, 90])
    mask = compute_ambient_loss_mask(timesteps, 1.0, num_train_timesteps=100)
    assert torch.equal(mask, torch.ones(2))


def test_quality_is_clamped():
    timesteps = torch.tensor([0, 50])
    # An out-of-range quality > 1 must not produce a threshold below zero / weird mask.
    mask = compute_ambient_loss_mask(timesteps, torch.tensor([5.0, 5.0]), num_train_timesteps=100)
    assert torch.equal(mask, torch.ones(2))


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        compute_ambient_loss_mask(torch.tensor([0]), None, num_train_timesteps=100, mode="nope")


def test_mismatched_quality_size_raises():
    with pytest.raises(ValueError):
        compute_ambient_loss_mask(torch.tensor([0, 1, 2]), torch.tensor([1.0, 1.0]), num_train_timesteps=100)


# ---------------------------------------------------------------------------
# Integration with the existing DiffusionModel.compute_loss call site
# ---------------------------------------------------------------------------


def _tiny_config(**overrides) -> DiffusionConfig:
    """A minimal state-only DiffusionConfig (no vision backbone) for fast CPU tests."""
    kwargs = {
        "n_obs_steps": 2,
        "horizon": 16,
        "n_action_steps": 8,
        "down_dims": (64, 128),
        "diffusion_step_embed_dim": 32,
        "num_train_timesteps": 100,
        "crop_shape": None,
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
    }
    kwargs.update(overrides)
    return DiffusionConfig(**kwargs)


def _make_batch(batch_size=6, config=None):
    return {
        OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 4),
        OBS_ENV_STATE: torch.randn(batch_size, config.n_obs_steps, 4),
        ACTION: torch.randn(batch_size, config.horizon, 4),
        "action_is_pad": torch.zeros(batch_size, config.horizon, dtype=torch.bool),
    }


def test_config_rejects_bad_ambient_mode():
    with pytest.raises(ValueError):
        _tiny_config(ambient_mask_mode="bogus")


def test_clean_quality_matches_vanilla_loss():
    """All-clean quality must reproduce the vanilla loss bit-for-bit (no-op guarantee)."""
    base = DiffusionModel(_tiny_config())
    batch = _make_batch(config=base.config)

    torch.manual_seed(0)
    vanilla = base.compute_loss({k: v.clone() for k, v in batch.items()})

    ambient_cfg = _tiny_config(use_ambient_loss_masking=True)
    ambient = DiffusionModel(ambient_cfg)
    ambient.load_state_dict(base.state_dict())
    clean_batch = {k: v.clone() for k, v in batch.items()}
    clean_batch[ambient_cfg.ambient_quality_key] = torch.ones(batch[ACTION].shape[0])

    torch.manual_seed(0)
    out = ambient.compute_loss(clean_batch)
    assert torch.allclose(vanilla, out, atol=1e-6)


def test_suboptimal_quality_changes_loss():
    """Marking samples as suboptimal must change the (masked) training loss."""
    cfg = _tiny_config(use_ambient_loss_masking=True, ambient_mask_mode="high")
    model = DiffusionModel(cfg)
    batch = _make_batch(config=cfg)

    torch.manual_seed(0)
    clean_batch = {k: v.clone() for k, v in batch.items()}
    clean_batch[cfg.ambient_quality_key] = torch.ones(batch[ACTION].shape[0])
    clean_loss = model.compute_loss(clean_batch)

    torch.manual_seed(0)
    noisy_batch = {k: v.clone() for k, v in batch.items()}
    # Heavily suboptimal: only the highest-noise timesteps may contribute.
    noisy_batch[cfg.ambient_quality_key] = torch.zeros(batch[ACTION].shape[0])
    noisy_loss = model.compute_loss(noisy_batch)

    assert not torch.allclose(clean_loss, noisy_loss)


def test_missing_quality_key_is_safe():
    """Enabling masking without a quality column in the batch is a no-op, not a crash."""
    cfg = _tiny_config(use_ambient_loss_masking=True)
    model = DiffusionModel(cfg)
    batch = _make_batch(config=cfg)  # no quality key present

    torch.manual_seed(0)
    loss = model.compute_loss(batch)
    assert torch.isfinite(loss)
