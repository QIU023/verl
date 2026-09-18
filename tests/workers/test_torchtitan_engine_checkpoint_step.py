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
"""verl names a checkpoint step in the path; the engine has to read it out."""

import unittest

from verl.workers.engine.torchtitan.transformer_impl import checkpoint_step_from_path


class TestCheckpointStepFromPath(unittest.TestCase):
    def test_reads_the_step_verl_names(self):
        self.assertEqual(7, checkpoint_step_from_path("/ckpt/global_step_7/actor"))
        self.assertEqual(120, checkpoint_step_from_path("/ckpt/global_step_120"))

    def test_a_path_without_a_step_means_the_latest(self):
        self.assertEqual(-1, checkpoint_step_from_path("/ckpt/actor"))
        self.assertEqual(-1, checkpoint_step_from_path(""))

    def test_the_step_is_read_not_guessed_from_other_digits(self):
        # A run directory carrying its own digits must not be mistaken for one.
        self.assertEqual(3, checkpoint_step_from_path("/runs/exp_2026/global_step_3"))
        self.assertEqual(-1, checkpoint_step_from_path("/runs/exp_2026/latest"))


if __name__ == "__main__":
    unittest.main()
