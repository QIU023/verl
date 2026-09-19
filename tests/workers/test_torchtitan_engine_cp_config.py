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
"""Engine config surface of the torchtitan engine: compat kwargs, defaults, the stream probe.

``_parallelism_compat_kwargs`` is what lets one engine build a ``ParallelismConfig`` on
several torchtitan trees, and it pins the context-parallel load balancer the rank-ordered
logit gather depends on. The rest pins the new engine-config keys the yamls carry and the
probe that decides whether the model takes a folded ``[T]`` token stream.

CPU only: config objects and one meta-device model build, no process group and no GPU.
"""

import dataclasses
import pathlib
import unittest

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
# The engine module imports torchtitan at module scope, so skip rather than fail when
# torchtitan is not on the path -- same guard as the sibling engine tests.
pytest.importorskip("torchtitan")

# torchtitan models context_parallel_load_balancer two ways: a dataclass on trees
# that define ContextParallelLoadBalancerConfig, and a plain str | None elsewhere.
try:  # noqa: E402
    from torchtitan.config import ContextParallelLoadBalancerConfig  # noqa: E402
except ImportError:  # noqa: E402
    ContextParallelLoadBalancerConfig = None  # noqa: E402
from torchtitan.models.llama3.config_registry import llama3_debugmodel  # noqa: E402
from torchtitan.protocols.model import BaseModel  # noqa: E402


def _compat():
    from verl.workers.engine.torchtitan.transformer_impl import _parallelism_compat_kwargs

    return _parallelism_compat_kwargs


class TestParallelismCompatKwargs(unittest.TestCase):
    def test_cp_pins_the_load_balancer_to_none(self):
        compat = _compat()
        balancer = compat("spmd_types", True)["context_parallel_load_balancer"]
        if ContextParallelLoadBalancerConfig is None:
            self.assertIsNone(balancer)
        else:
            self.assertIsInstance(balancer, ContextParallelLoadBalancerConfig)
            self.assertIsNone(balancer.load_balancer_type)

    def test_no_balancer_key_without_cp(self):
        compat = _compat()
        self.assertNotIn("context_parallel_load_balancer", compat("spmd_types", False))


class TestEngineConfigDefaults(unittest.TestCase):
    KEYS = ("sequence_parallel", "initial_load_path", "pipeline_token_budget")

    def _yaml(self, relative):
        import verl

        return yaml.safe_load((pathlib.Path(verl.__file__).parent / relative).read_text())

    def test_dataclass_defaults(self):
        from verl.workers.config import TorchtitanEngineConfig

        config = TorchtitanEngineConfig()
        self.assertIs(config.sequence_parallel, True)
        self.assertIsNone(config.initial_load_path)
        self.assertIsNone(config.pipeline_token_budget)

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
        self.assertTrue(self._folded(llama3_debugmodel()))

    def test_a_model_without_the_override_reads_false(self):
        self.assertFalse(type(_BareModel()).preprocess_inputs is not BaseModel.preprocess_inputs)
        with self.assertRaises(NotImplementedError):
            _BareModel().preprocess_inputs({}, parallel_dims=None, parallelism=None)


if __name__ == "__main__":
    unittest.main()
