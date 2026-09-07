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
"""The full-tensor weight sync ships EVERY expert of a sharded expert stack.

torchtitan's ``to_hf`` names only the local experts of a dim-0-sharded stack (the
DCP convention), so the engine takes such stacks out before the conversion, gathers
each one whole and lets ``to_hf`` split the full stack into all its experts.
"""

import os
import types
import unittest

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan.models.kimi_k3")

from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from torch.distributed.tensor import DTensor, Shard  # noqa: E402


class _Adapter:
    """The two adapter surfaces the engine touches: from_hf_map (stack detection) and to_hf (the split)."""

    from_hf_map = {
        "model.layers.{}.moe.experts.{}.w1.weight": "layers.{}.moe.experts.w1",
        "model.layers.{}.attn.wq.weight": "layers.{}.attn.wq",
    }

    def to_hf(self, sd):
        out = {}
        for k, v in sd.items():
            if k.endswith("moe.experts.w1"):
                layer = k.split(".")[1]
                for e, row in enumerate(v.unbind(0)):
                    out[f"model.layers.{layer}.moe.experts.{e}.w1.weight"] = row
            else:
                out[k.replace("layers.", "model.layers.")] = v
        return out


class TestExpertStacksGoWhole(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)
        cls.mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))

    @classmethod
    def tearDownClass(cls):
        torch.distributed.destroy_process_group()

    def _engine(self):
        from verl.workers.engine.torchtitan.transformer_impl import TorchTitanEngine

        fake = types.SimpleNamespace(checkpointer=types.SimpleNamespace(sd_adapter=_Adapter()))
        fake._expert_stack_slots = lambda name, param: TorchTitanEngine._expert_stack_slots(fake, name, param)
        return TorchTitanEngine, fake

    def test_a_sharded_stack_is_detected_and_a_plain_matrix_is_not(self):
        engine, fake = self._engine()
        stack = DTensor.from_local(torch.randn(4, 3, 2), self.mesh, [Shard(0)])
        wq = DTensor.from_local(torch.randn(6, 2), self.mesh, [Shard(0)])
        self.assertEqual(len(fake._expert_stack_slots("layers.1.moe.experts.w1", stack)), 4)
        self.assertIsNone(fake._expert_stack_slots("layers.1.attn.wq", wq))

    def test_every_expert_of_the_stack_is_yielded_from_the_gathered_tensor(self):
        engine, fake = self._engine()
        local = torch.randn(4, 3, 2)
        stack = DTensor.from_local(local, self.mesh, [Shard(0)])
        got = dict(engine._iter_expert_stacks({"layers.1.moe.experts.w1": stack}, fake.checkpointer.sd_adapter, "cpu"))
        self.assertEqual(sorted(got), [f"model.layers.1.moe.experts.{e}.w1.weight" for e in range(4)])
        for e in range(4):
            self.assertTrue(torch.equal(got[f"model.layers.1.moe.experts.{e}.w1.weight"], local[e].to(torch.bfloat16)))
            self.assertTrue(got[f"model.layers.1.moe.experts.{e}.w1.weight"].is_contiguous())
