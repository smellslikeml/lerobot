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
"""Riemannian Preconditioned LoRA (RP-LoRA) AdamW optimizer.

Adapted from "Riemannian Preconditioned LoRA for Fine-Tuning Foundation
Models" (Zhang et al., ICML 2024, arxiv:2402.02347).

A LoRA layer replaces a frozen weight ``W`` with ``W + B A``, where
``A`` is ``(r, n)`` and ``B`` is ``(m, r)`` — exactly the parameter
layout HuggingFace PEFT exposes as ``lora_A`` / ``lora_B``. That
factorization lives on a low-rank quotient manifold, and training under
the Riemannian metric of Mishra & Sepulchre (2016) amounts to
preconditioning each gradient by an ``r x r`` inverse Gram matrix built
from the *paired* factor (eq. 1 / Algorithm 1 of the paper):

    grad_A <- (B^T B + eps I)^{-1} grad_A   # left-multiply  ((r,r) @ (r,n))
    grad_B <- grad_B (A A^T + eps I)^{-1}   # right-multiply ((m,r) @ (r,r))

This rebalances the A/B scale imbalance of vanilla LoRA training (the
motivation behind LoRA+, which instead splits the learning rate) and, per
Theorem 4.1 of the paper, yields stable feature learning with a single
learning rate. We follow the paper's "scaled AdamW" recipe: precondition
each *raw* gradient in place, then hand control to a standard AdamW
step. The inverse of an ``r x r`` matrix (``r`` is tiny, e.g. 4) is the
only added cost, so storage and runtime overhead are negligible.

Non-LoRA parameters have no paired factor and pass through unchanged, so
the optimizer is a drop-in for models that mix adapter and non-adapter
trainable parameters. Select with ``--policy.optimizer=rp_lora_adamw``
alongside ``--peft.method_type=LORA``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import Parameter

from .optimizers import OptimizerConfig, OptimizerParams


def pair_lora_factors(
    params: dict[str, Parameter],
) -> list[tuple[Parameter, Parameter]]:
    """Group PEFT ``lora_A`` / ``lora_B`` parameters into ``(A, B)`` pairs.

    PEFT names paired factors with a shared prefix, e.g.
    ``...q_proj.lora_A.default.weight`` and ``...q_proj.lora_B.default.weight``;
    we match on the substring preceding ``lora_A`` / ``lora_B``. Parameters
    without a LoRA partner are ignored and fall back to plain AdamW.
    """
    a_by_prefix: dict[str, Parameter] = {}
    b_by_prefix: dict[str, Parameter] = {}
    for name, param in params.items():
        if "lora_A" in name:
            a_by_prefix[name.split("lora_A", 1)[0]] = param
        elif "lora_B" in name:
            b_by_prefix[name.split("lora_B", 1)[0]] = param
    return [(a, b_by_prefix[prefix]) for prefix, a in a_by_prefix.items() if prefix in b_by_prefix]


class RPLoRAAdamW(torch.optim.AdamW):
    """AdamW that applies the RP-LoRA inverse-Gram preconditioner to LoRA grads.

    Each :meth:`step` rescales the ``.grad`` of every paired ``(A, B)``
    factor in place *before* delegating to :class:`torch.optim.AdamW`. The
    Gram matrices are built from the current (pre-update) factor values and
    detached, so no second-order autograd graph is constructed.

    Note:
        Closures are unsupported: preconditioning happens once on the
        already-populated ``.grad``, mirroring Algorithm 1 of the paper
        and the no-closure usage in lerobot's training loop.
    """

    def __init__(
        self,
        params: list[Parameter],
        pairs: list[tuple[Parameter, Parameter]],
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        gram_eps: float,
    ) -> None:
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self._lora_pairs = pairs
        self.gram_eps = gram_eps

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        # Scale each LoRA gradient by its paired factor's inverse Gram
        # before the AdamW update. A is (r, n); B is (m, r).
        for a, b in self._lora_pairs:
            if a.grad is None or b.grad is None:
                continue
            rank = a.shape[0]
            eye = torch.eye(rank, dtype=a.dtype, device=a.device)
            btb_inv = torch.linalg.inv(b.detach().t() @ b.detach() + self.gram_eps * eye)
            aat_inv = torch.linalg.inv(a.detach() @ a.detach().t() + self.gram_eps * eye)
            a.grad = btb_inv @ a.grad
            b.grad = b.grad @ aat_inv
        return super().step(closure)


@OptimizerConfig.register_subclass("rp_lora_adamw")
@dataclass
class RPLoRAAdamWConfig(OptimizerConfig):
    """Configuration for :class:`RPLoRAAdamW`.

    Attributes:
        gram_eps: Ridge term added to the Gram matrices before inversion,
            matching the ``delta`` regularizer of the paper (default 1e-6).
    """

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    gram_eps: float = 1e-6

    def build(self, params: OptimizerParams) -> torch.optim.Optimizer:
        """Build the RP-LoRA AdamW optimizer.

        Args:
            params: Must be a ``dict[str, Parameter]`` from
                ``dict(model.named_parameters())`` so paired LoRA factors
                can be matched by name (same contract as
                :class:`XVLAAdamWConfig`).

        Returns:
            The configured :class:`RPLoRAAdamW` optimizer.

        Raises:
            AssertionError: If ``params`` is not a ``dict`` (e.g. built from
                ``model.parameters()``), since named parameters are required
                to pair LoRA factors.
        """
        assert isinstance(params, dict), (
            "RP-LoRA requires named_parameters() (a dict[name, Parameter]) to pair LoRA "
            "factors — see XVLAAdamWConfig for the same contract."
        )
        pairs = pair_lora_factors(params)
        # Keep a stable, trainable-only param list for the underlying AdamW.
        trainable_params = [p for p in params.values() if p.requires_grad]
        return RPLoRAAdamW(
            trainable_params,
            pairs=pairs,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            gram_eps=self.gram_eps,
        )
