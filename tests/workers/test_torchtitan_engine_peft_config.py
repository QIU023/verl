# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Adapter-only weight sync for the torchtitan engine: peft_config and naming.

``engine_workers.update_weights`` gates its base-then-adapter sequence on
``peft_config`` being present. The engine used to return None unconditionally, so a run
configured with ``model.lora.merge=False`` still received merged full weights and LoRA
bought no sync bandwidth at all. These pin the three things that decision rests on:

* ``peft_config`` reports the rank and alpha the adapters were BUILT with, and the
  target-module names that were actually wrapped;
* the adapter half emits PEFT's ``lora_A`` / ``lora_B`` names and the raw, UNSCALED
  factors, because PEFT re-applies ``lora_alpha / r`` from the config;
* the base half is the FULL model with ``base_layer`` inserted only on wrapped
  projections -- shipping just the wrapped bases would leave the rollout without
  embeddings, norms or experts.

CPU only: these exercise the naming and config helpers directly, with no process group,
no GPU and no rollout engine.
"""

import math
import unittest

import pytest

torch = pytest.importorskip("torch")
# The engine module imports torchtitan at module scope, so skip rather than fail when
# the fork is not on the path -- same guard as the sibling weight-sync test.
pytest.importorskip("torchtitan.models.kimi_k3")

import torch.nn as nn  # noqa: E402


class _Wrapper(nn.Module):
    """Stands in for KimiLoRALinear: the engine detects the trio by attribute."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.lora_a = nn.Parameter(torch.randn(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self._lora_scaling = alpha / rank


class _Model(nn.Module):
    def __init__(self, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.q_proj = _Wrapper(4, 4, rank, alpha)
        self.o_proj = _Wrapper(4, 4, rank, alpha)
        self.norm = nn.RMSNorm(4)


class _FakeAdapter:
    """Minimal stand-in for the state-dict adapter's naming contract."""

    def _is_text_only(self, state_dict):
        return True

    def _tt_key_to_hf(self, key, text_only=False):
        return f"model.{key}"


def _helpers():
    from verl.workers.engine.torchtitan.transformer_impl import (
        _adapter_state_dict,
        _insert_base_layer_suffix,
        _peft_config_from_wrappers,
        _titan_lora_wrappers,
        _wrapped_hf_base_names,
    )

    return (
        _titan_lora_wrappers,
        _peft_config_from_wrappers,
        _wrapped_hf_base_names,
        _adapter_state_dict,
        _insert_base_layer_suffix,
    )


class TestTitanPeftConfig(unittest.TestCase):
    def test_wrappers_are_found_by_attribute_not_by_config(self):
        find, _, _, _, _ = _helpers()
        found = find(_Model())
        self.assertEqual(sorted(found), ["o_proj", "q_proj"])

    def test_no_wrappers_means_no_peft_config(self):
        _, build, _, _, _ = _helpers()
        self.assertIsNone(build({}))

    def test_peft_config_reports_the_built_rank_and_alpha(self):
        find, build, _, _, _ = _helpers()
        cfg = build(find(_Model(rank=8, alpha=16.0)))
        self.assertEqual(cfg["r"], 8)
        self.assertAlmostEqual(cfg["lora_alpha"], 16.0, places=5)
        self.assertEqual(cfg["target_modules"], ["o_proj", "q_proj"])
        self.assertEqual(cfg["bias"], "none")

    def test_mixed_ranks_raise_rather_than_reporting_one_of_them(self):
        _, build, _, _, _ = _helpers()
        model = _Model(rank=8, alpha=16.0)
        model.o_proj = _Wrapper(4, 4, rank=4, alpha=8.0)
        find, _, _, _, _ = _helpers()
        with self.assertRaises(ValueError):
            build(find(model))

    def test_adapter_half_uses_peft_names_and_unscaled_factors(self):
        find, _, names, adapters, _ = _helpers()
        model = _Model(rank=8, alpha=16.0)
        with torch.no_grad():
            model.q_proj.lora_b.fill_(0.5)
        wrappers = find(model)
        hf = names(_FakeAdapter(), wrappers, {})
        out = adapters(wrappers, hf)
        self.assertEqual(
            sorted(out),
            [
                "model.o_proj.lora_A.weight",
                "model.o_proj.lora_B.weight",
                "model.q_proj.lora_A.weight",
                "model.q_proj.lora_B.weight",
            ],
        )
        # UNSCALED: the wrapper's scaling is 2.0 here, so a pre-multiplied export
        # would read 1.0 and PEFT would then scale it again.
        self.assertAlmostEqual(
            float(out["model.q_proj.lora_B.weight"].detach().flatten()[0]), 0.5, places=6
        )
        self.assertEqual(out["model.q_proj.lora_A.weight"].shape, (8, 4))
        self.assertEqual(out["model.q_proj.lora_B.weight"].shape, (4, 8))

    def test_base_half_keeps_everything_and_renames_only_the_wrapped(self):
        find, _, names, _, insert = _helpers()
        wrappers = find(_Model())
        hf = names(_FakeAdapter(), wrappers, {})
        params = {
            "model.q_proj.weight": torch.zeros(1),
            "model.o_proj.weight": torch.zeros(1),
            "model.norm.weight": torch.zeros(1),
            "model.embed_tokens.weight": torch.zeros(1),
        }
        out = insert(params, hf)
        self.assertEqual(
            sorted(out),
            [
                "model.embed_tokens.weight",
                "model.norm.weight",
                "model.o_proj.base_layer.weight",
                "model.q_proj.base_layer.weight",
            ],
        )

    def test_a_missing_base_key_is_not_invented(self):
        """A wrapped projection absent from this rank's shard must not appear."""
        find, _, names, _, insert = _helpers()
        wrappers = find(_Model())
        hf = names(_FakeAdapter(), wrappers, {})
        out = insert({"model.q_proj.weight": torch.zeros(1)}, hf)
        self.assertEqual(sorted(out), ["model.q_proj.base_layer.weight"])

    def test_scaling_round_trip_matches_the_wrapper_math(self):
        """alpha recovered from the wrapper must reproduce its own scaling."""
        find, build, _, _, _ = _helpers()
        for rank, alpha in ((8, 16.0), (4, 4.0), (16, 32.0)):
            cfg = build(find(_Model(rank=rank, alpha=alpha)))
            self.assertAlmostEqual(
                cfg["lora_alpha"] / cfg["r"], alpha / rank, places=6
            )
            self.assertFalse(math.isnan(cfg["lora_alpha"]))


if __name__ == "__main__":
    unittest.main()
