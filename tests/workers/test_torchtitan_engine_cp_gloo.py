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
"""Context parallel on the torchtitan engine, live over a two-rank gloo group.

Two halves of the CP contract, both on CPU:

* the logit gather (``_finish_pred``). The engine gathers the token-sharded logits
  over the cp group and lets every cp rank compute the SAME full-sequence loss, so
  the gather's backward must hand each rank its own slice UNSCALED: FSDP reduces
  over ``dp_shard x cp`` and divides by ``dp_size`` only, so the cp axis contributes
  a plain sum. ``grad_scaler=True`` -- the Ulysses default, meant for a loss that is
  already averaged over the sp group -- would multiply that sum by the cp degree.
  The second half of the test pins exactly that factor, so the file documents why
  the engine passes ``grad_scaler=False``.

* the model's CP preprocessing (``preprocess_inputs``), which is what the engine
  hands the forward under CP: the masks built from the global positions, the
  contiguous rank-ordered token shards the rank-ordered logit gather above depends
  on, and the KDA routing.

No GPU and no compile: the K3 debug model builds on CPU, and its flex ``BlockMask``
is created on CPU by ``create_attention_mask``. ``preprocess_inputs`` needs no
outer SPMD mesh context -- ``annotate_input_spmd_types`` enters
``set_current_spmd_mesh(parallel_dims.spmd_dense_mesh())`` itself -- so the only
thing the test has to arrange is ``parallel_dims.device_type`` being "cpu" while
the mesh is built.
"""

import os
import traceback
import unittest
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
# The CP model path lives in the torchtitan fork; skip rather than fail when it is
# not on the path -- same guard as the sibling engine tests.
pytest.importorskip("torchtitan.models.kimi_k3")

import torch.multiprocessing as mp  # noqa: E402

from verl.utils.net_utils import get_free_port  # noqa: E402

WORLD = 2
_TIMEOUT = 900


def _entry(worker, rank, world, port, out):
    """One gloo rank: init, run ``worker``, ship back plain python (no tensor storage)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world)
        out.put((rank, None, worker(rank, world)))
    except BaseException:  # noqa: BLE001 - the rank's traceback is the test failure
        out.put((rank, traceback.format_exc(), None))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _spawn(worker, world=WORLD):
    """Run ``worker`` on ``world`` gloo ranks; a rank's assertion fails the test here."""
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port, _ = get_free_port("127.0.0.1")
    procs = [ctx.Process(target=_entry, args=(worker, r, world, port, out)) for r in range(world)]
    for p in procs:
        p.start()
    try:
        results = [out.get(timeout=_TIMEOUT) for _ in procs]
    finally:
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
    failures = [(rank, tb) for rank, tb, _ in results if tb is not None]
    if failures:
        raise AssertionError("\n\n".join(f"rank {rank}:\n{tb}" for rank, tb in failures))
    return {rank: payload for rank, _, payload in results}


# --------------------------------------------------------------------------------------
# 1. the cp logit gather (_finish_pred)
# --------------------------------------------------------------------------------------

_T, _V = 8, 5


def _logit_gather_worker(rank, world):
    from verl.utils.ulysses import gather_outputs_and_unpad

    group = torch.distributed.group.WORLD
    shard = _T // world

    # Identical on every rank: the full logits the no-CP run would produce, and the
    # weights standing in for the loss's dL/dlogits. Every cp rank computes the same
    # full-sequence loss, so the weights must not depend on the rank.
    torch.manual_seed(1234)
    full = torch.randn(1, _T, _V)
    weights = torch.randn(1, _T, _V)

    # Reference: the same loss on the full sequence, no CP anywhere.
    reference = full.detach().clone().requires_grad_()
    (reference * weights).sum().backward()

    payload = {"reference_grad": reference.grad.tolist()}
    for grad_scaler in (False, True):
        local = full.detach()[:, rank * shard : (rank + 1) * shard].clone().requires_grad_()
        gathered = gather_outputs_and_unpad(local, gather_dim=1, grad_scaler=grad_scaler, group=group)
        # Forward: the gather is rank-ordered, which is why the engine pins the
        # contiguous (non-permuting) cp load balancer.
        assert gathered.shape == (1, _T, _V), gathered.shape
        assert torch.equal(gathered.detach(), full), f"rank {rank} gathered out of order"

        (gathered * weights).sum().backward()

        # FSDP reduces the weight gradients over dp_shard x cp and divides by dp_size
        # only, so the cp axis is a plain SUM: the gradient the optimizer sees is the
        # two ranks' shards concatenated in rank order.
        parts = [torch.empty_like(local.grad) for _ in range(world)]
        torch.distributed.all_gather(parts, local.grad.contiguous(), group=group)
        summed = torch.cat(parts, dim=1)

        if grad_scaler:
            # The Ulysses scaling multiplies every cp rank's slice by the cp degree.
            assert torch.allclose(summed, world * reference.grad), "grad_scaler=True is not the cp degree"
            assert not torch.allclose(summed, reference.grad), "world=2 must not be a no-op"
        else:
            assert torch.allclose(summed, reference.grad), "grad_scaler=False must reproduce the full-seq grad"
        payload["local_grad_scaler" if grad_scaler else "local_grad_plain"] = local.grad.tolist()
        payload["summed_scaler" if grad_scaler else "summed_plain"] = summed.tolist()
    return payload


class TestContextParallelLogitGather(unittest.TestCase):
    """``_finish_pred``'s cp gather: rank-ordered forward, unscaled backward."""

    def test_unscaled_gather_reproduces_the_full_sequence_gradient(self):
        results = _spawn(_logit_gather_worker)
        reference = torch.tensor(results[0]["reference_grad"])
        for rank in range(WORLD):
            # Every cp rank computes the same full-sequence loss, so every rank must
            # agree on the reference and on the summed gradient.
            self.assertEqual(results[rank]["reference_grad"], results[0]["reference_grad"])
            summed = torch.tensor(results[rank]["summed_plain"])
            self.assertTrue(torch.allclose(summed, reference), rank)
            # ... out of a slice that is only this rank's quarter of the sequence.
            local = torch.tensor(results[rank]["local_grad_plain"])
            self.assertEqual(list(local.shape), [1, _T // WORLD, _V])
            self.assertTrue(torch.allclose(local, reference[:, rank * (_T // WORLD) : (rank + 1) * (_T // WORLD)]))

    def test_grad_scaler_true_would_multiply_the_gradient_by_the_cp_degree(self):
        """Why the engine passes ``grad_scaler=False``: the default scales by cp."""
        results = _spawn(_logit_gather_worker)
        reference = torch.tensor(results[0]["reference_grad"])
        for rank in range(WORLD):
            summed = torch.tensor(results[rank]["summed_scaler"])
            self.assertTrue(torch.allclose(summed, WORLD * reference), rank)
            self.assertFalse(torch.allclose(summed, reference), rank)


# --------------------------------------------------------------------------------------
# 2. the model's CP preprocessing (what the engine hands preprocess_inputs)
# --------------------------------------------------------------------------------------

_SEQ = 32
_DOC_BOUNDARY = 20


def _packed_stream():
    """A packed two-document stream of ``_SEQ`` tokens; positions restart at 0 mid-stream."""
    tokens_T = torch.arange(_SEQ, dtype=torch.long) % 17
    positions_T = torch.cat(
        [torch.arange(_DOC_BOUNDARY), torch.arange(_SEQ - _DOC_BOUNDARY)]
    ).to(torch.long)
    labels_T = (tokens_T + 1) % 17
    return tokens_T, labels_T, positions_T


def _cp_config():
    """The K3 debug config staged as the engine stages one under CP."""
    from torchtitan.config.transform import ContextParallelTransform, apply_transforms
    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.common.cp_attention import UlyssesCPFlexInnerAttention
    from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
    from torchtitan.models.kimi_k3.cp_kda import ContextParallelInnerKDA
    from torchtitan.models.kimi_k3.kda import InnerKDA

    config = kimi_k3_debugmodel()
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = WORLD
    # The field is a ContextParallelLoadBalancerConfig and defaults to "headtail",
    # which permutes the stream; the engine pins it to None (contiguous) because
    # _finish_pred gathers the logits back in rank order.
    assert config.parallelism.context_parallel_load_balancer.load_balancer_type == "headtail"
    config.parallelism.context_parallel_load_balancer.load_balancer_type = None
    transform = ContextParallelTransform(
        inner_attention={
            FlexInnerAttention.Config: UlyssesCPFlexInnerAttention,
            InnerKDA.Config: ContextParallelInnerKDA,
        },
        exclude_fqn_prefixes=("vision_encoder",),
    )
    return apply_transforms(config, [transform])


def _preprocess_worker(rank, world):
    from torchtitan.distributed.parallel_dims import ParallelDims

    config = _cp_config()
    with patch("torchtitan.distributed.parallel_dims.device_type", "cpu"):
        parallel_dims = ParallelDims(
            dp_replicate=1, dp_shard=1, cp=world, tp=1, pp=1, ep=1, world_size=world
        )
        parallel_dims.build_mesh()
    assert parallel_dims.cp_enabled

    model = config.model_spec.model.build()
    tokens_T, labels_T, positions_T = _packed_stream()
    # The engine hands the model the folded [T] stream with the GLOBAL positions
    # (prepare_model_inputs) and takes back this rank's shard.
    inputs, labels, extra_kwargs = model.preprocess_inputs(
        {"input": tokens_T, "labels": labels_T, "positions": positions_T},
        parallel_dims=parallel_dims,
        parallelism=config.parallelism,
    )

    shard = _SEQ // world
    lo, hi = rank * shard, (rank + 1) * shard
    assert inputs.shape == (shard,), inputs.shape
    # Contiguous, rank-ordered shards: rank r holds tokens [shard*r, shard*(r+1)).
    assert torch.equal(inputs, tokens_T[lo:hi]), f"rank {rank} shard is not contiguous"
    assert labels.shape == (shard,), labels.shape
    assert torch.equal(labels, labels_T[lo:hi])

    positions = extra_kwargs["positions"]
    assert positions.shape == (shard,), positions.shape
    assert torch.equal(positions, positions_T[lo:hi]), f"rank {rank} positions do not follow the shard"

    masks = extra_kwargs["attention_masks"]
    assert set(masks) == {"quadratic_attention", "kda"}, sorted(masks)
    # Ulysses keeps the flex BlockMask GLOBAL (the all-gather-KV backend is the one
    # that shards it through the compiled create_block_mask).
    assert tuple(masks["quadratic_attention"].shape) == (1, 1, _SEQ, _SEQ)
    # The KDA metadata carries the packed stream's GLOBAL document offsets; the
    # rank-local boundaries live on the routing object below.
    global_cu_seqlens = masks["kda"].cu_seq_q.tolist()
    assert global_cu_seqlens == [0, _DOC_BOUNDARY, _SEQ], global_cu_seqlens

    routing = extra_kwargs["kda_cp_routing"]
    assert routing is not None
    assert int(routing.cp_rank) == rank, (routing.cp_rank, rank)
    assert int(routing.world) == world

    # All-gathering the shards in rank order reproduces the stream -- the property the
    # engine's rank-ordered logit gather relies on.
    parts = [torch.empty_like(inputs) for _ in range(world)]
    torch.distributed.all_gather(parts, inputs.contiguous(), group=torch.distributed.group.WORLD)
    assert torch.equal(torch.cat(parts), tokens_T), "rank-ordered shards do not rebuild the stream"

    return {
        "inputs": inputs.tolist(),
        "labels": labels.tolist(),
        "positions": positions.tolist(),
        "extra_keys": sorted(extra_kwargs),
        "gathered": torch.cat(parts).tolist(),
        "global_cu_seqlens": global_cu_seqlens,
        "blockmask_shape": list(masks["quadratic_attention"].shape),
        "routing_cu_seqlens": routing.cu_seqlens.tolist(),
        "routing_type": type(routing).__name__,
    }


class TestContextParallelPreprocessInputs(unittest.TestCase):
    """The K3 model's CP preprocessing on CPU: contiguous shards, masks, KDA routing."""

    def test_each_rank_gets_its_contiguous_half_of_the_stream(self):
        results = _spawn(_preprocess_worker)
        tokens_T, labels_T, positions_T = _packed_stream()
        shard = _SEQ // WORLD
        for rank in range(WORLD):
            got = results[rank]
            self.assertEqual(len(got["inputs"]), shard)
            self.assertEqual(got["inputs"], tokens_T[rank * shard : (rank + 1) * shard].tolist())
            self.assertEqual(got["labels"], labels_T[rank * shard : (rank + 1) * shard].tolist())
            self.assertEqual(got["positions"], positions_T[rank * shard : (rank + 1) * shard].tolist())
            self.assertEqual(got["gathered"], tokens_T.tolist())

    def test_extra_kwargs_carry_the_masks_and_the_kda_routing(self):
        results = _spawn(_preprocess_worker)
        for rank in range(WORLD):
            got = results[rank]
            self.assertEqual(got["extra_keys"], ["attention_masks", "kda_cp_routing", "positions"])
            # The flex mask stays global under the Ulysses backend ...
            self.assertEqual(got["blockmask_shape"], [1, 1, _SEQ, _SEQ])
            # ... and so do the KDA document offsets, while the routing holds the
            # rank-local segment boundaries: rank 1 starts inside the first document.
            self.assertEqual(got["global_cu_seqlens"], [0, _DOC_BOUNDARY, _SEQ])
            self.assertEqual(got["routing_type"], "ContextParallelRouting")
        self.assertEqual(results[0]["routing_cu_seqlens"], [0, _SEQ // WORLD])
        self.assertEqual(
            results[1]["routing_cu_seqlens"],
            [0, _DOC_BOUNDARY - _SEQ // WORLD, _SEQ // WORLD],
        )


if __name__ == "__main__":
    unittest.main()
