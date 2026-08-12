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
* the base half is the FULL model -- shipping just the wrapped bases would leave the
  rollout without embeddings, norms or experts -- with ``base_layer`` inserted on every
  projection the ROLLOUT wraps, which is a wider set than the one torchtitan wrapped.

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


def _sd(*fqns, wrapper: str | None = None):
    """A state dict shaped like the real one: LoRA modules contribute base + adapters.

    ``wrapper`` inserts an activation-checkpointing segment into the MODULE path only,
    which is the asymmetry that caused the bug: named_modules() keeps it, state_dict()
    strips it.
    """
    out = {}
    for fqn in fqns:
        out[f"{fqn}.base.weight"] = torch.zeros(1)
        out[f"{fqn}.lora_a"] = torch.zeros(1)
        out[f"{fqn}.lora_b"] = torch.zeros(1)
    _ = wrapper
    return out


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

    def test_base_half_keeps_everything_and_renames_the_projections(self):
        _, _, _, _, insert = _helpers()
        params = {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
            "model.layers.0.self_attn.o_proj.weight": torch.zeros(1),
            "model.norm.weight": torch.zeros(1),
            "model.embed_tokens.weight": torch.zeros(1),
        }
        out = insert(params, "kimi_k3")
        self.assertEqual(
            sorted(out),
            [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.o_proj.base_layer.weight",
                "model.layers.0.self_attn.q_proj.base_layer.weight",
                "model.norm.weight",
            ],
        )

    def test_projections_torchtitan_did_not_wrap_are_still_renamed(self):
        """The rollout decides this set, not the trainer, and the two differ.

        vLLM's ``get_supported_lora_modules`` collects leaf names by module TYPE -- "in
        vLLM, all linear layers support LoRA" -- so enabling LoRA wraps EVERY linear
        regardless of the adapter's target_modules. ``apply_lora`` skips K3's KDA subtree
        structurally, so a KDA ``q_proj`` is unwrapped on the trainer side and wrapped on
        the rollout side.

        Naming that key from the trainer's wrapper set shipped the plain name into a
        params_dict that only had ``base_layer``, and vLLM's stacked loop reads a missing
        destination as "packed projection not present on this layer" and falls through to
        the plain path -- which is how this surfaced as
        ``KeyError: 'layers.0.self_attn.q_proj.weight'`` with nothing pointing at LoRA.
        """
        _, _, _, _, insert = _helpers()
        # Nothing here is wrapped by torchtitan; every one is a linear vLLM wraps.
        params = {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
            "model.layers.0.self_attn.f_a_proj.weight": torch.zeros(1),
            "model.layers.0.self_attn.b_proj.weight": torch.zeros(1),
            "model.layers.0.self_attention_res_proj.weight": torch.zeros(1),
            "model.layers.0.block_sparse_moe.gate.weight": torch.zeros(1),
            "model.layers.0.block_sparse_moe.gate.e_score_correction_bias": torch.zeros(1),
        }
        out = insert(params, "kimi_k3")
        for name in params:
            stem, _, suffix = name.rpartition(".")
            self.assertIn(f"{stem}.base_layer.{suffix}", out, name)

    def test_the_ones_that_are_not_linear_keep_their_names(self):
        """Renaming a norm would point it at a base_layer that never exists.

        `q_conv1d` deliberately is NOT in here: vLLM builds the KDA short conv as a
        ColumnParallelLinear, so it IS wrapped. Reading module names instead of module
        types is what put it here in the first version of this test.
        """
        _, _, _, _, insert = _helpers()
        params = {
            "model.layers.0.self_attn.o_norm.weight": torch.zeros(1),
            "model.layers.0.self_attn.A_log": torch.zeros(1),
            "model.layers.0.input_layernorm.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(1),
        }
        self.assertEqual(sorted(insert(params, "kimi_k3")), sorted(params))

    def test_every_key_the_text_stack_ships_is_classified(self):
        """The whole HF key space at once, so this stops being found one KeyError at a time.

        Left column is every distinct leaf name in the k3mini export (1032 keys, 41
        distinct leaves); right column is whether its vLLM destination is a LinearBase
        subclass and therefore LoRA-wrapped. Two of these are not guessable from the name:
        `conv1d` IS a ColumnParallelLinear in vLLM, and `embed_tokens` / `lm_head` are NOT
        wrapped because K3 declares no `embedding_modules`.
        """
        _, _, _, _, insert = _helpers()
        renamed = {
            # MLA and KDA projections
            "q_proj", "k_proj", "v_proj", "o_proj", "b_proj", "g_proj",
            "f_a_proj", "f_b_proj", "q_a_proj", "q_b_proj",
            "kv_a_proj_with_mqa", "kv_b_proj",
            # short conv, modelled as a linear
            "q_conv1d", "k_conv1d", "v_conv1d",
            # dense FFN and latent MoE
            "gate_proj", "up_proj", "down_proj",
            "routed_expert_down_proj", "routed_expert_up_proj",
            # Block AttnRes graft
            "self_attention_res_proj", "output_attn_res_proj", "mlp_res_proj",
        }
        plain = {
            "input_layernorm", "post_attention_layernorm", "q_a_layernorm",
            "kv_a_layernorm", "o_norm", "mlp_res_norm", "output_attn_res_norm",
            "self_attention_res_norm", "routed_expert_norm",
            "embed_tokens", "lm_head", "norm",
        }
        params = {f"model.layers.0.{leaf}.weight": torch.zeros(1) for leaf in renamed | plain}
        # The router and its correction bias hang off block_sparse_moe. The per-expert
        # weights are renamed too, for a different reason than the projections: FusedMoE's
        # expert mapping builds its weight_name WITH the base_layer prefix whenever the
        # model has any base_layer parameter, so an un-suffixed expert key matches no
        # mapping entry and falls through to the plain-name lookup.
        params["model.layers.0.block_sparse_moe.gate.weight"] = torch.zeros(1)
        params["model.layers.0.block_sparse_moe.gate.e_score_correction_bias"] = torch.zeros(1)
        params["model.layers.0.self_attn.A_log"] = torch.zeros(1)
        params["model.layers.0.self_attn.dt_bias"] = torch.zeros(1)
        for w in ("w1", "w2", "w3"):
            params[f"model.layers.0.block_sparse_moe.experts.0.{w}.weight"] = torch.zeros(1)

        out = insert(params, "kimi_k3")
        for leaf in sorted(renamed):
            self.assertIn(f"model.layers.0.{leaf}.base_layer.weight", out, leaf)
        for leaf in sorted(plain):
            self.assertIn(f"model.layers.0.{leaf}.weight", out, leaf)
        self.assertIn("model.layers.0.block_sparse_moe.gate.base_layer.weight", out)
        self.assertIn(
            "model.layers.0.block_sparse_moe.gate.base_layer.e_score_correction_bias", out
        )
        self.assertIn("model.layers.0.self_attn.A_log", out)
        self.assertIn("model.layers.0.self_attn.dt_bias", out)
        for w in ("w1", "w2", "w3"):
            self.assertIn(
                f"model.layers.0.block_sparse_moe.experts.0.{w}.base_layer.weight", out, w
            )
        self.assertEqual(len(out), len(params))

    def test_the_rename_survives_vllms_stacked_substring_replace(self):
        """The suffix goes on the SOURCE name, so the mapping has to compose.

        vLLM maps a source projection onto its packed destination with
        ``name.replace(weight_name, param_name)``. That is a substring replace on the
        module segment, which is the only reason renaming the source is sound: the
        inserted ``base_layer`` sits after the segment being replaced and survives it.
        """
        _, _, _, _, insert = _helpers()
        renamed = insert(
            {"model.layers.0.self_attn.q_proj.weight": torch.zeros(1)}, "kimi_k3"
        )
        (name,) = renamed
        self.assertEqual(
            name.replace(".q_proj", ".in_proj_qkvgfab"),
            "model.layers.0.self_attn.in_proj_qkvgfab.base_layer.weight",
        )

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
