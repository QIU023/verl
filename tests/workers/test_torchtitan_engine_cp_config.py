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
"""Context parallel on the torchtitan engine: backend mapping, config ordering, defaults.

The engine raises ``parallelism.context_parallel_degree`` on the BUILT Trainer.Config and
only then applies the CP transform, because torchtitan validates the degree against the
inner attentions every time a Trainer.Config is post-inited. These pin that ordering, the
backend table behind ``context_parallel_backend``, the load-balancer pin the rank-ordered
logit gather depends on, and the three engine-config keys the yamls carry.

CPU only: config objects and one meta-device model build, no process group and no GPU.
"""

import dataclasses
import pathlib
import unittest

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
# The engine module imports torchtitan at module scope, so skip rather than fail when
# the fork is not on the path -- same guard as the sibling engine tests.
pytest.importorskip("torchtitan.models.kimi_k3")

from torchtitan.config import ContextParallelLoadBalancerConfig  # noqa: E402
from torchtitan.config.transform import apply_transforms  # noqa: E402
from torchtitan.models.common.attention import FlexInnerAttention  # noqa: E402
from torchtitan.models.common.cp_attention import (  # noqa: E402
    KVAllGatherCPFlexInnerAttention,
    UlyssesCPFlexInnerAttention,
)
from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel  # noqa: E402
from torchtitan.models.kimi_k3.cp_kda import ContextParallelInnerKDA  # noqa: E402
from torchtitan.models.kimi_k3.kda import InnerKDA  # noqa: E402
from torchtitan.models.llama3.config_registry import llama3_debugmodel  # noqa: E402
from torchtitan.protocols.model import BaseModel  # noqa: E402


def _cp_trainer_config():
    """A K3 debug Trainer.Config staged the way the engine stages one: the degree raised
    on the built config, the balancer pinned, the transform not yet applied."""
    _, compat = _helpers()
    config = kimi_k3_debugmodel()
    config.parallelism.context_parallel_degree = 2
    for key, value in compat("spmd_types", True).items():
        setattr(config.parallelism, key, value)
    return config


def _helpers():
    from verl.workers.engine.torchtitan.transformer_impl import (
        _context_parallel_transform,
        _parallelism_compat_kwargs,
    )

    return _context_parallel_transform, _parallelism_compat_kwargs


class TestContextParallelTransform(unittest.TestCase):
    def test_ulysses_maps_flex_and_kda(self):
        transform, _ = _helpers()
        mapping = transform(kimi_k3_debugmodel().model_spec.model, "ulysses").inner_attention
        self.assertIs(mapping[FlexInnerAttention.Config], UlyssesCPFlexInnerAttention)
        self.assertIs(mapping[InnerKDA.Config], ContextParallelInnerKDA)

    def test_allgather_kv_maps_flex_to_the_sharded_blockmask_backend(self):
        transform, _ = _helpers()
        mapping = transform(kimi_k3_debugmodel().model_spec.model, "allgather_kv").inner_attention
        self.assertIs(mapping[FlexInnerAttention.Config], KVAllGatherCPFlexInnerAttention)
        self.assertIs(mapping[InnerKDA.Config], ContextParallelInnerKDA)

    def test_unknown_backend_names_the_choices(self):
        transform, _ = _helpers()
        with self.assertRaises(ValueError) as caught:
            transform(kimi_k3_debugmodel().model_spec.model, "ring")
        message = str(caught.exception)
        self.assertIn("ulysses", message)
        self.assertIn("allgather_kv", message)

    def test_a_model_without_kda_gets_no_kda_entry(self):
        transform, _ = _helpers()
        mapping = transform(llama3_debugmodel().model_spec.model, "ulysses").inner_attention
        self.assertEqual(list(mapping), [FlexInnerAttention.Config])

    def test_the_vision_tower_keeps_its_local_attention(self):
        transform, _ = _helpers()
        config = _cp_trainer_config()
        self.assertIn("vision_encoder", transform(config.model_spec.model, "ulysses").exclude_fqn_prefixes)
        applied = apply_transforms(config, [transform(config.model_spec.model, "ulysses")])
        rewritten = {
            fqn: type(traversed)
            for fqn, traversed, _, _ in applied.model_spec.model.traverse(FlexInnerAttention.Config)
        }
        # the tower's tokens are not sharded on the cp axis, so its attention stays local
        self.assertIs(rewritten["vision_encoder.block.attn.inner_attention"], FlexInnerAttention.Config)
        self.assertIs(rewritten["layers.3.attention.inner_attention"], UlyssesCPFlexInnerAttention.Config)


class TestDegreeThenTransformOrdering(unittest.TestCase):
    """torchtitan validates the degree against the inner attentions in every
    ``Trainer.Config.__post_init__``; only the transform's validated copy pairs them."""

    def test_degree_raised_before_the_transform_validates(self):
        transform, _ = _helpers()
        config = _cp_trainer_config()
        applied = apply_transforms(config, [transform(config.model_spec.model, "ulysses")])
        self.assertEqual(applied.parallelism.context_parallel_degree, 2)

    def test_the_degree_alone_is_rejected(self):
        config = kimi_k3_debugmodel()
        config.parallelism.context_parallel_degree = 2
        with self.assertRaises(ValueError) as caught:
            config.__post_init__()
        self.assertIn("CPInnerAttention", str(caught.exception))

    def test_the_default_headtail_balancer_is_rejected_under_ulysses(self):
        # Why _parallelism_compat_kwargs pins the balancer to None: the field defaults
        # to "headtail", which Ulysses CP refuses.
        transform, _ = _helpers()
        config = kimi_k3_debugmodel()
        config.parallelism.context_parallel_degree = 2
        self.assertEqual(config.parallelism.context_parallel_load_balancer.load_balancer_type, "headtail")
        with self.assertRaises(ValueError):
            apply_transforms(config, [transform(config.model_spec.model, "ulysses")])


class TestParallelismCompatKwargs(unittest.TestCase):
    def test_cp_pins_the_load_balancer_to_none(self):
        _, compat = _helpers()
        balancer = compat("spmd_types", True)["context_parallel_load_balancer"]
        self.assertIsInstance(balancer, ContextParallelLoadBalancerConfig)
        self.assertIsNone(balancer.load_balancer_type)

    def test_no_balancer_key_without_cp(self):
        _, compat = _helpers()
        self.assertNotIn("context_parallel_load_balancer", compat("spmd_types", False))


class TestEngineConfigDefaults(unittest.TestCase):
    KEYS = ("context_parallel_backend", "sequence_parallel", "initial_load_path")

    def _yaml(self, relative):
        import verl

        return yaml.safe_load((pathlib.Path(verl.__file__).parent / relative).read_text())

    def test_dataclass_defaults(self):
        from verl.workers.config import TorchtitanEngineConfig

        config = TorchtitanEngineConfig()
        self.assertEqual(config.context_parallel_backend, "ulysses")
        self.assertIs(config.sequence_parallel, True)
        self.assertIsNone(config.initial_load_path)

    def test_engine_yaml_declares_the_same_defaults(self):
        from verl.workers.config import TorchtitanEngineConfig

        declared = self._yaml("trainer/config/engine/torchtitan.yaml")
        defaults = {f.name: f.default for f in dataclasses.fields(TorchtitanEngineConfig)}
        for key in self.KEYS:
            self.assertIn(key, declared)
            self.assertEqual(declared[key], defaults[key], key)

    def test_ref_yaml_forwards_the_same_keys(self):
        declared = self._yaml("trainer/config/ref/torchtitan_ref.yaml")["torchtitan"]
        for key in self.KEYS:
            self.assertIn(key, declared)
            # the ref half interpolates from the actor's value rather than restating a default
            self.assertTrue(declared[key].startswith("${oc.select:"), key)
            self.assertIn(key, declared[key])


class _BareModel(BaseModel):
    pass


class TestFoldedTokenStreamProbe(unittest.TestCase):
    """The engine probes the built model, not its name, for the folded [T] token stream."""

    def _folded(self, config):
        with torch.device("meta"):
            model = config.model_spec.model.build()
        return type(model).preprocess_inputs is not BaseModel.preprocess_inputs

    def test_titan_decoders_own_their_preprocessing(self):
        for name, factory in (("kimi_k3", kimi_k3_debugmodel), ("llama3", llama3_debugmodel)):
            with self.subTest(model=name):
                self.assertTrue(self._folded(factory()))

    def test_a_model_without_the_override_reads_false(self):
        self.assertFalse(type(_BareModel()).preprocess_inputs is not BaseModel.preprocess_inputs)
        with self.assertRaises(NotImplementedError):
            _BareModel().preprocess_inputs({}, parallel_dims=None, parallelism=None)


if __name__ == "__main__":
    unittest.main()
