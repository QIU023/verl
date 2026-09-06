# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""The torchtitan engine's base half ships plain HF names; the vLLM receiver resolves them.

``resolve_weight_name`` toggles one ``.base_layer`` segment per name against the live vLLM
namespace, so the trainer renames nothing. These pin the three cases the base half meets:
a LoRA-wrapped receiver on released vLLM (suffix added), on newer vLLM whose LoRA layer
loads inner names (kept plain), and a receiver without LoRA (kept plain).
"""

import pytest

pytest.importorskip("vllm")

from verl.utils.vllm import utils as vu  # noqa: E402


class _Model:
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    hf_to_vllm_mapper = None

    def named_parameters(self, remove_duplicate=False):
        return iter(())


_WRAPPED = {
    "model.layers.0.self_attn.qkv_proj.base_layer.weight",
    "model.layers.0.mlp.down_proj.base_layer.weight",
    "model.norm.weight",
    "model.embed_tokens.weight",
}
_PLAIN = {"model.layers.0.self_attn.qkv_proj.weight", "model.norm.weight"}
_Q = "model.layers.0.self_attn.q_proj.weight"


def test_plain_base_names_reach_the_wrapped_receiver(monkeypatch):
    monkeypatch.setattr(vu, "_HAS_LORA_LOAD_WEIGHTS", False)
    assert vu.resolve_weight_name(_Model(), _Q, _WRAPPED) == "model.layers.0.self_attn.q_proj.base_layer.weight"
    assert vu.resolve_weight_name(_Model(), "model.norm.weight", _WRAPPED) == "model.norm.weight"


def test_plain_names_stay_plain_on_newer_vllm(monkeypatch):
    monkeypatch.setattr(vu, "_HAS_LORA_LOAD_WEIGHTS", True)
    assert vu.resolve_weight_name(_Model(), _Q, _WRAPPED) == _Q


@pytest.mark.parametrize("has_lora_load_weights", [False, True])
def test_plain_names_on_a_receiver_without_lora(monkeypatch, has_lora_load_weights):
    monkeypatch.setattr(vu, "_HAS_LORA_LOAD_WEIGHTS", has_lora_load_weights)
    assert vu.resolve_weight_name(_Model(), _Q, _PLAIN) == _Q
