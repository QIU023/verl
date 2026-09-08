# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""A fake-quant QAT expert stack ships to the rollout as the actor computes with it, dequant(quant(w))."""

import types
import unittest

import pytest

torch = pytest.importorskip("torch")
mx_qat = pytest.importorskip("torchtitan.components.quantization.mx_qat")

import torch.nn as nn  # noqa: E402


class _QATExperts(nn.Module, mx_qat.MXQATExpertsBase):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(4, 8, 64))


class _PlainExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(4, 8, 64))


class TestQATStackSync(unittest.TestCase):
    def _engine(self, experts):
        from verl.workers.engine.torchtitan.transformer_impl import TorchTitanEngine

        model = nn.Module()
        model.layers = nn.ModuleDict({"1": nn.Module()})
        model.layers["1"].moe = nn.Module()
        model.layers["1"].moe.experts = experts
        fake = types.SimpleNamespace(module=[model])
        fake._owning_module = lambda fqn: TorchTitanEngine._owning_module(fake, fqn)
        fake._as_the_actor_computes = lambda fqn, full: TorchTitanEngine._as_the_actor_computes(fake, fqn, full)
        return fake

    def test_a_qat_stack_is_fake_quantized_and_a_plain_one_is_not(self):
        torch.manual_seed(0)
        qat = _QATExperts()
        fake = self._engine(qat)
        full = qat.w1.detach().clone()
        shipped = fake._as_the_actor_computes("layers.1.moe.experts.w1", full)
        expected = mx_qat._fake_quant_mx(full, mx_qat._WEIGHT_ELEM, mx_qat._BLOCK)
        self.assertTrue(torch.equal(shipped.to(expected.dtype), expected))
        self.assertFalse(torch.equal(shipped.float(), full.float()))
        plain = _PlainExperts()
        fake2 = self._engine(plain)
        self.assertTrue(torch.equal(fake2._as_the_actor_computes("layers.1.moe.experts.w1", plain.w1.detach()), plain.w1.detach()))

    def test_an_unknown_parameter_passes_through(self):
        fake = self._engine(_PlainExperts())
        t = torch.randn(2, 3)
        self.assertTrue(torch.equal(fake._as_the_actor_computes("nowhere.w1", t), t))
