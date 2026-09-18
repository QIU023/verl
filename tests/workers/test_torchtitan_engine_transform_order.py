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
"""The engine appends CP and LoRA in that order, but apply_transforms reorders.

The engine builds its transform list with the context-parallel transform first
and the LoRA transform second, then hands both to ``apply_transforms``. The
resulting order is not the append order: ``apply_transforms`` sorts by each
transform's ``run_after`` declaration. This pins the declaration the engine
relies on, so a change to it fails here rather than silently swapping the order
a LoRA-plus-context-parallel run sees.
"""

import unittest


class TestTransformOrder(unittest.TestCase):
    def test_lora_declares_it_runs_after_context_parallel(self):
        from torchtitan.config.transform import ContextParallelTransform
        from torchtitan.config.transform.lora import LoRATransform

        self.assertIn(ContextParallelTransform, tuple(LoRATransform.run_after))
        self.assertEqual((), tuple(ContextParallelTransform.run_after))

    def test_apply_transforms_orders_context_parallel_first(self):
        from torchtitan.config.transform import ContextParallelTransform
        from torchtitan.config.transform.apply import _ordered
        from torchtitan.config.transform.lora import LoRATransform

        cp = ContextParallelTransform.__new__(ContextParallelTransform)
        lora = LoRATransform.__new__(LoRATransform)
        # Appended the way the engine appends them, and the reverse.
        for appended in ([cp, lora], [lora, cp]):
            with self.subTest(appended=[type(t).__qualname__ for t in appended]):
                ordered = _ordered(list(appended))
                self.assertIs(ordered[0], cp)
                self.assertIs(ordered[1], lora)


if __name__ == "__main__":
    unittest.main()
