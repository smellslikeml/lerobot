#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Unit tests for the RP-LoRA AdamW optimizer (arxiv:2402.02347).

These exercise the integration with the existing ``OptimizerConfig``
registry and the per-step inverse-Gram preconditioner mechanics, without
running a full LoRA fine-tune.
"""

import pytest
import torch
from torch.nn import Parameter

# Importing the package runs the register_subclass("rp_lora_adamw") decorator
# in rp_lora.py — the same wiring path make_optimizer_and_scheduler relies on.
import lerobot.optim as optim_pkg
from lerobot.optim import RPLoRAAdamWConfig
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.optim.rp_lora import RPLoRAAdamW, pair_lora_factors


def _named_lora_pair(prefix: str, r: int = 4, n: int = 8, m: int = 6):
    a = Parameter(torch.randn(r, n))
    b = Parameter(torch.randn(m, r))
    return {f"{prefix}.lora_A.default.weight": a, f"{prefix}.lora_B.default.weight": b}, a, b


def test_rp_lora_registered_in_optimizer_registry():
    """Importing lerobot.optim must register the ``rp_lora_adamw`` choice.

    ``OptimizerConfig.type`` resolves the registered choice name via
    ``get_choice_name``, so this also asserts the wiring edit in
    ``optim/__init__.py`` took effect (the same registry
    ``make_optimizer_and_scheduler`` dispatches through).
    """
    assert hasattr(optim_pkg, "RPLoRAAdamWConfig")
    assert isinstance(RPLoRAAdamWConfig(), OptimizerConfig)
    assert RPLoRAAdamWConfig().type == "rp_lora_adamw"


def test_rp_lora_pairs_factors_by_name_and_ignores_non_lora():
    a1, b1 = Parameter(torch.zeros(4, 8)), Parameter(torch.zeros(6, 4))
    a2, b2 = Parameter(torch.zeros(4, 5)), Parameter(torch.zeros(7, 4))
    other = Parameter(torch.zeros(3, 3))
    named = {
        "base.q_proj.lora_A.default.weight": a1,
        "base.q_proj.lora_B.default.weight": b1,
        "base.norm.weight": other,  # non-LoRA -> ignored
        "base.v_proj.lora_A.default.weight": a2,
        "base.v_proj.lora_B.default.weight": b2,
        "base.k_proj.lora_A.default.weight": Parameter(torch.zeros(4, 9)),  # no B partner
    }
    pairs = pair_lora_factors(named)

    pair_ids = {(id(a), id(b)) for a, b in pairs}
    assert pair_ids == {(id(a1), id(b1)), (id(a2), id(b2))}


def test_rp_lora_step_preconditions_by_paired_gram():
    """A step must equal plain AdamW on the manually-scaled gradients."""
    named, a, b = _named_lora_pair("x", r=4, n=8, m=6)
    a.grad = torch.randn(4, 8)
    b.grad = torch.randn(6, 4)
    eps = 1e-6

    # Reference: clone the factors, scale their grads by the inverse Gram
    # exactly as eq. 1 prescribes, then run a vanilla AdamW step.
    a_ref = Parameter(a.detach().clone())
    b_ref = Parameter(b.detach().clone())
    eye = torch.eye(4)
    a_ref.grad = torch.linalg.inv(b.detach().t() @ b.detach() + eps * eye) @ a.grad
    b_ref.grad = b.grad @ torch.linalg.inv(a.detach() @ a.detach().t() + eps * eye)

    opt = RPLoRAAdamWConfig(lr=1e-2, gram_eps=eps).build(named)
    ref = torch.optim.AdamW([a_ref, b_ref], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    opt.step()
    ref.step()

    assert torch.allclose(a, a_ref, atol=1e-6)
    assert torch.allclose(b, b_ref, atol=1e-6)


def test_rp_lora_no_op_on_non_lora_params():
    """With no LoRA pairs, RP-LoRA AdamW must behave exactly like AdamW."""
    p = Parameter(torch.randn(4, 5))
    p.grad = torch.randn_like(p)
    p_ref = Parameter(p.detach().clone())
    p_ref.grad = p.grad.detach().clone()

    named = {"base.linear.weight": p}  # no lora_A/lora_B -> empty pairs
    opt = RPLoRAAdamWConfig(lr=1e-2).build(named)
    ref = torch.optim.AdamW([p_ref], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    opt.step()
    ref.step()

    assert torch.allclose(p, p_ref, atol=1e-7)


def test_rp_lora_skips_preconditioning_when_pair_grad_incomplete():
    """If either factor's grad is missing, the pair is left unscaled.

    A still gets a plain AdamW update from its own grad (B has no grad, so
    its preconditioner cannot fire); the result must equal vanilla AdamW.
    """
    named, a, b = _named_lora_pair("x")
    a.grad = torch.ones(4, 8)
    # b.grad stays None -> pair skipped, a's grad is not preconditioned.

    a_ref = Parameter(a.detach().clone())
    a_ref.grad = a.grad.detach().clone()

    RPLoRAAdamWConfig(lr=1e-3).build(named).step()
    torch.optim.AdamW([a_ref], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0).step()

    assert torch.allclose(a, a_ref, atol=1e-7)
    assert b.grad is None  # B untouched — no grad to update from


def test_rp_lora_requires_named_params():
    p = Parameter(torch.randn(4, 5))
    with pytest.raises(AssertionError):
        RPLoRAAdamWConfig().build([p])  # list, not a dict of named parameters


def test_rp_lora_optimizer_is_torch_adamw():
    """The optimizer subclasses torch.optim.AdamW (one-level pattern)."""
    named, _, _ = _named_lora_pair("x")
    opt = RPLoRAAdamWConfig().build(named)
    assert isinstance(opt, RPLoRAAdamW)
    assert isinstance(opt, torch.optim.AdamW)
