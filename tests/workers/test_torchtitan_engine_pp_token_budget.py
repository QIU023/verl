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

"""The pipeline token budget: the config field, its environment fallback, and the refusals."""

import os
import unittest
from unittest import mock

from verl.workers.engine.torchtitan.transformer_impl import pipeline_token_budget


class TestPipelineTokenBudget(unittest.TestCase):
    def test_the_config_field_wins_over_the_environment(self):
        with mock.patch.dict(os.environ, {"VERL_PP_TOKEN_BUDGET": "512"}):
            self.assertEqual(pipeline_token_budget(2048, 100), 2048)

    def test_the_environment_is_the_fallback(self):
        with mock.patch.dict(os.environ, {"VERL_PP_TOKEN_BUDGET": "512"}):
            self.assertEqual(pipeline_token_budget(None, 100), 512)

    def test_a_budget_equal_to_the_micro_batch_pads_nothing(self):
        self.assertEqual(pipeline_token_budget(256, 256), 256)

    def test_no_budget_anywhere_names_the_knob(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as caught:
                pipeline_token_budget(None, 100)
        self.assertIn("pipeline_token_budget", str(caught.exception))

    def test_a_micro_batch_over_the_budget_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            pipeline_token_budget(128, 2048)
        self.assertIn("2048", str(caught.exception))
        self.assertIn("128", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
