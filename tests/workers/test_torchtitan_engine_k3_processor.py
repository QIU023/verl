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
"""The Kimi K3 processor through verl's multimodal helper: images as medias, pads expanded."""

import os
import unittest

K3_EXPORT = os.environ.get("VERL_TEST_K3_EXPORT", "/root/models/kimi-k3-debug-nt-rel")


@unittest.skipUnless(os.path.isdir(K3_EXPORT), "needs a Kimi K3 export with its processor files")
class TestKimiK3ProcessorInputs(unittest.TestCase):
    def test_pads_match_the_processor_token_count(self):
        from PIL import Image
        from transformers import AutoProcessor

        from verl.utils.tokenizer.tokenizer import _processor_takes_medias, build_multimodal_processor_inputs

        processor = AutoProcessor.from_pretrained(K3_EXPORT, trust_remote_code=True)
        self.assertTrue(_processor_takes_medias(processor))
        image = Image.new("RGB", (64, 48), (200, 30, 30))
        expected = int(processor.media_processor.media_tokens_calculator({"type": "image", "image": image}))
        out = build_multimodal_processor_inputs(
            processor, text="<|kimi_image_placeholder|> what colour is this?", images=[image]
        )
        pad = processor.tokenizer.convert_tokens_to_ids("<|media_pad|>")
        ids = out["input_ids"][0].tolist()
        self.assertEqual(ids.count(pad), expected)
        self.assertEqual(out["attention_mask"].shape[-1], len(ids))
        self.assertIn("pixel_values", out)
        self.assertEqual(int(out["grid_thws"][0].prod()), out["pixel_values"].shape[0])


class TestImagesFromMessages(unittest.TestCase):
    def test_bytes_path_and_pil_blocks_become_pil_images(self):
        import io

        from PIL import Image

        from verl.utils.dataset.rl_dataset import _images_from_messages

        image = Image.new("RGB", (32, 24), (1, 2, 3))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        messages = [
            {"role": "system", "content": "text only"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"bytes": buf.getvalue()}},
                    {"type": "image", "image": image},
                    {"type": "text", "text": "what is this"},
                ],
            },
        ]
        images = _images_from_messages(messages)
        self.assertEqual([im.size for im in images], [(32, 24), (32, 24)])
