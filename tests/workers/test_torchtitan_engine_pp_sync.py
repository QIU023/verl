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
"""Under a pipeline, the weight sync on every rank yields every stage's tensors, in stage order."""

import os
import types
import unittest

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan.models.kimi_k3")

import torch.multiprocessing as mp  # noqa: E402


def _worker(rank, world, port, out):
    from torch.distributed.device_mesh import init_device_mesh

    from verl.workers.engine.torchtitan.transformer_impl import TorchTitanEngine

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("pp",))
    fake = types.SimpleNamespace(parallel_dims=types.SimpleNamespace(pp_enabled=True, get_mesh=lambda name: mesh))
    # stage 0 holds two tensors, stage 1 one: different counts per rank are part of the contract
    mine = [(f"stage{rank}.w{i}", torch.full((2, 3), float(10 * rank + i))) for i in range(2 - rank)]
    got = list(TorchTitanEngine._iter_pp_gathered(fake, iter(mine), "cpu"))
    out.put((rank, [(n, t.tolist()) for n, t in got]))  # plain lists: tensor storage must not outlive the worker
    torch.distributed.destroy_process_group()


class TestPipelineSyncGathersEveryStage(unittest.TestCase):
    def test_both_ranks_yield_every_stage_in_order(self):
        ctx = mp.get_context("spawn")
        out = ctx.Queue()
        procs = [ctx.Process(target=_worker, args=(r, 2, 29531, out)) for r in range(2)]
        for p in procs:
            p.start()
        results = dict(out.get(timeout=120) for _ in procs)
        for p in procs:
            p.join(timeout=30)
        expected = [("stage0.w0", 0.0), ("stage0.w1", 1.0), ("stage1.w0", 10.0)]
        for rank in range(2):
            names = [n for n, _ in results[rank]]
            self.assertEqual(names, [n for n, _ in expected])
            for (n, t), (_, v) in zip(results[rank], expected):
                self.assertEqual(t, torch.full((2, 3), v).tolist(), (rank, n))
