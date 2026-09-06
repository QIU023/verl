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
* the base half is the FULL model under its plain HF names -- the vLLM receiver
  (``resolve_weight_name``) toggles ``.base_layer`` per name against its live
  namespace, so the trainer renames nothing; a wrapped projection whose base key
  is absent is named in a warning, never invented.

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
    """Stands in for torchtitan core's LoRA linear: the base ``weight`` on the module,
    ``lora_a`` / ``lora_b`` as Linear submodules, ``_lora_scaling`` = alpha / rank.
    The engine detects it as it detects the earlier wrapper (``base`` submodule,
    adapter parameters): by the trio of attributes."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.base = self  # the attribute trio the engine looks for
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        nn.init.zeros_(self.lora_b.weight)
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


def _sd(*fqns, wrapper: str | None = None):
    """A state dict shaped like the real one: LoRA modules contribute base + adapters.

    ``wrapper`` inserts an activation-checkpointing segment into the MODULE path only,
    which is the asymmetry that caused the bug: named_modules() keeps it, state_dict()
    strips it.
    """
    out = {}
    for fqn in fqns:
        out[f"{fqn}.weight"] = torch.zeros(1)
        out[f"{fqn}.lora_a.weight"] = torch.zeros(1)
        out[f"{fqn}.lora_b.weight"] = torch.zeros(1)
    _ = wrapper
    return out


def _engine_logger():
    from verl.workers.engine.torchtitan import transformer_impl

    return transformer_impl.logger


def _helpers():
    from verl.workers.engine.torchtitan.transformer_impl import (
        _adapter_state_dict,
        _peft_config_from_wrappers,
        _titan_lora_wrappers,
        _warn_wrapped_bases_missing,
        _wrapped_hf_base_names,
    )

    return (
        _titan_lora_wrappers,
        _peft_config_from_wrappers,
        _wrapped_hf_base_names,
        _adapter_state_dict,
        _warn_wrapped_bases_missing,
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
            model.q_proj.lora_b.weight.fill_(0.5)
        wrappers = find(model)
        hf = names(_FakeAdapter(), wrappers, _sd(*wrappers))
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

    def test_base_half_ships_plain_hf_names(self):
        """No ``.base_layer`` from the trainer: the receiver resolves it per name."""
        find, _, names, _, warn = _helpers()
        wrappers = find(_Model())
        hf = names(_FakeAdapter(), wrappers, _sd(*wrappers))
        params = {
            "model.q_proj.weight": torch.zeros(1),
            "model.o_proj.weight": torch.zeros(1),
            "model.norm.weight": torch.zeros(1),
            "model.embed_tokens.weight": torch.zeros(1),
        }
        with self.assertNoLogs(_engine_logger(), level="WARNING"):
            warn(params, hf)
        self.assertEqual(
            sorted(params),
            [
                "model.embed_tokens.weight",
                "model.norm.weight",
                "model.o_proj.weight",
                "model.q_proj.weight",
            ],
        )

    def test_a_missing_base_key_is_not_invented(self):
        """A wrapped projection absent from this rank's shard must not appear.

        It also must not raise. Two absences are legitimate -- a PP rank does not own
        every layer, and a graft-only target can have no HF destination because to_hf
        drops what the original architecture has no key for -- so this warns with both
        names instead. The silent version of this is what produced a KeyError from deep
        inside vLLM's loader, with no indication of which projection disagreed.
        """
        find, _, names, _, warn = _helpers()
        wrappers = find(_Model())
        hf = names(_FakeAdapter(), wrappers, _sd(*wrappers))
        params = {"model.q_proj.weight": torch.zeros(1)}
        with self.assertLogs(_engine_logger(), level="WARNING") as logs:
            warn(params, hf)
        self.assertEqual(sorted(params), ["model.q_proj.weight"])
        self.assertIn("o_proj", "\n".join(logs.output))


class TestShardExportLoraMode(unittest.TestCase):
    """The sharded delta export folds adapters into the bases and refuses adapter mode."""

    def _engine(self, lora_cfg):
        import types

        from verl.workers.engine.torchtitan.transformer_impl import TorchTitanEngine

        fake = types.SimpleNamespace(
            module=[_Model()],
            model_config=types.SimpleNamespace(lora=lora_cfg),
            parallel_dims=types.SimpleNamespace(pp_enabled=False),
        )
        # the two engine methods the shard path calls before it touches a device
        fake._assert_shard_export_supported = lambda: None
        fake._lora_mode = lambda: TorchTitanEngine._lora_mode(fake)
        return TorchTitanEngine, fake

    def test_an_unconfigured_lora_block_means_merged(self):
        engine, fake = self._engine({})
        wrappers, adapter_mode = engine._lora_mode(fake)
        self.assertEqual(sorted(wrappers), ["o_proj", "q_proj"])
        self.assertFalse(adapter_mode)

    def test_adapter_mode_is_refused_on_the_shard_path(self):
        engine, fake = self._engine({"merge": False})
        self.assertTrue(engine._lora_mode(fake)[1])
        with self.assertRaises(NotImplementedError):
            engine.get_per_tensor_param_shard(fake)

    def test_scaling_round_trip_matches_the_wrapper_math(self):
        """alpha recovered from the wrapper must reproduce its own scaling."""
        find, build, _, _, _ = _helpers()
        for rank, alpha in ((8, 16.0), (4, 4.0), (16, 32.0)):
            cfg = build(find(_Model(rank=rank, alpha=alpha)))
            self.assertAlmostEqual(
                cfg["lora_alpha"] / cfg["r"], alpha / rank, places=6
            )
            self.assertFalse(math.isnan(cfg["lora_alpha"]))


class TestWrapperSegments(unittest.TestCase):
    """named_modules() keeps wrapper segments that state_dict() strips.

    Under activation checkpointing a module reached at
    `layers.0._checkpoint_wrapped_module.ffn.gate_proj` appears in the state dict as
    `layers.0.ffn.gate_proj`. Composing an HF name from the MODULE path then hands to_hf
    something it cannot map:

        ValueError: Unmapped tt key:
        'layers.0._checkpoint_wrapped_module.ffn.gate_proj.weight'

    which is what a live GRPO adapter sync actually raised. merge_lora_state_dict already
    hit this from the same direction, so the naming goes through torchtitan's
    _state_dict_prefix rather than a second stripper here.
    """

    def test_a_checkpoint_wrapped_module_path_maps_to_the_stripped_name(self):
        _, _, names, _, _ = _helpers()
        module_path = "layers.0._checkpoint_wrapped_module.ffn.gate_proj"
        state_dict_path = "layers.0.ffn.gate_proj"
        hf = names(_FakeAdapter(), {module_path: object()}, _sd(state_dict_path))
        self.assertEqual(hf[module_path], f"model.{state_dict_path}.weight")

    def test_an_unrecognised_wrapper_raises_instead_of_guessing(self):
        _, _, names, _, _ = _helpers()
        with self.assertRaises(KeyError):
            names(_FakeAdapter(), {"layers.0._mystery_wrap.ffn.gate_proj": object()}, {})


if __name__ == "__main__":
    unittest.main()
