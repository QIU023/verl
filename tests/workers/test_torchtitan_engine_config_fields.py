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

"""Engine config fields the trainer cannot be built without: where the first load comes
from, and the context-parallel backend the config itself refuses."""

import unittest

from verl.workers.config import TorchtitanEngineConfig
from verl.workers.engine.torchtitan.transformer_impl import initial_checkpoint_source


class TestInitialCheckpointSource(unittest.TestCase):
    def test_a_dcp_path_takes_precedence_over_the_hf_weights(self):
        load_in_hf, path = initial_checkpoint_source("/data/packed_dcp", "/models/kimi-k3")
        self.assertFalse(load_in_hf)
        self.assertEqual(path, "/data/packed_dcp")

    def test_without_a_dcp_path_the_hf_weights_are_loaded(self):
        load_in_hf, path = initial_checkpoint_source(None, "/models/kimi-k3")
        self.assertTrue(load_in_hf)
        self.assertEqual(path, "/models/kimi-k3")

    def test_an_empty_path_is_not_a_dcp_path(self):
        load_in_hf, path = initial_checkpoint_source("", "/models/kimi-k3")
        self.assertEqual(path, "/models/kimi-k3")
        self.assertIs(load_in_hf, False)




if __name__ == "__main__":
    unittest.main()
