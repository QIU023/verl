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

"""The torchtitan engine's weight sync has to fold LoRA adapters into the base.

A LoRA-wrapped projection stores ``base.weight``, ``lora_a`` and ``lora_b``. The
state-dict adapter maps ``base.weight`` onto the plain HF name and has no mapping for
the adapter tensors, so a raw ``state_dict()`` ships the unmerged base and drops
everything LoRA learned. Under LoRA the base is frozen, so the rollout would receive
identical weights at every step -- which looks exactly like a working sync unless you
check, and means the actor trains adapters the rollout never sees.

Unlike the FSDP engine, this one has no adapter-mode path: ``get_per_tensor_param``
returns ``peft_config=None``, and ``engine_workers`` gates its base-then-adapter
sequence on that being present. Merged is therefore the only correct mode here.

CPU only, no ray and no rollout: the branch under test is a state-dict transform.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchtitan.models.kimi_k3")


def _helper():
    """Load just the transform, without pulling in the engine's dependency chain."""
    import pathlib

    src = pathlib.Path(
        "verl/workers/engine/torchtitan/transformer_impl.py"
    ).read_text()
    start = src.index("def _merged_state_dict_if_lora")
    end = src.index("class EngineEvalModeCtx")
    ns = {
        "logger": types.SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None
        )
    }
    exec(compile(src[start:end], "transformer_impl_excerpt", "exec"), ns)
    return ns["_merged_state_dict_if_lora"]


def _lora_model():
    from torchtitan.models.kimi_k3 import config_registry as cr

    model = cr.kimi_k3_debugmodel_gated_lora().model_spec.model.build()
    model.init_weights()
    return model


class TestMergedWeightSync:
    def test_adapter_tensors_do_not_reach_the_rollout(self):
        merged, merged_keys = _helper()(_lora_model())
        assert not [k for k in merged if "lora" in k or ".base." in k]
        assert merged_keys, "the merge must report which keys it produced"
        assert "layers.0.ffn.gate_proj.weight" in merged

    def test_the_merged_output_tracks_the_adapter(self):
        """The differential that separates this from the raw path.

        With lora_b at its zero init the two agree -- LoRA is identity at step 0, so
        that is correct, and it is why the defect was invisible. Give lora_b a value
        and the merged output moves while a raw state_dict does not.
        """
        model = _lora_model()
        helper = _helper()
        key = "layers.0.ffn.gate_proj.weight"
        before = helper(model)[0][key].clone()
        raw_before = model.state_dict()["layers.0.ffn.gate_proj.base.weight"].clone()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith("lora_b"):
                    param.fill_(0.01)

        assert not torch.equal(helper(model)[0][key], before)
        raw_after = model.state_dict()["layers.0.ffn.gate_proj.base.weight"]
        assert torch.equal(raw_after, raw_before)

    def test_a_model_without_lora_is_untouched(self):
        from torchtitan.models.kimi_k3 import config_registry as cr

        model = cr.kimi_k3_debugmodel_report_arch().model_spec.model.build()
        plain, merged_keys = _helper()(model)
        assert set(plain) == set(model.state_dict())
        assert merged_keys == frozenset()
