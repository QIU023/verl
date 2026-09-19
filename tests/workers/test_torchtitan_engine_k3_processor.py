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

        from verl.utils.tokenizer.tokenizer import processor_takes_medias, build_multimodal_processor_inputs

        processor = AutoProcessor.from_pretrained(K3_EXPORT, trust_remote_code=True)
        self.assertTrue(processor_takes_medias(processor))
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


    def test_generic_vl_builder_expands_the_pads_and_the_rollout_collapse_restores_the_placeholder(self):
        from PIL import Image
        from transformers import AutoProcessor, AutoTokenizer

        from verl.utils.tokenizer import collapse_media_blocks, normalize_token_ids
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        tokenizer = AutoTokenizer.from_pretrained(K3_EXPORT, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(K3_EXPORT, trust_remote_code=True)
        builder = create_continuous_token_builder(tokenizer, hf_model_type="kimi_k3", processor=processor)
        image = Image.new("RGB", (84, 56), (10, 200, 30))
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "What colour?"}]}
        ]
        ids = builder.build_initial_tokens(messages, images=[image])
        pad = processor.tokenizer.convert_tokens_to_ids("<|media_pad|>")
        expected = int(processor.media_processor.media_tokens_calculator({"type": "image", "image": image}))
        self.assertEqual(ids.count(pad), expected)
        rollout_ids = collapse_media_blocks(ids, processor)
        template_ids = normalize_token_ids(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        )
        self.assertEqual(rollout_ids, template_ids)
        self.assertNotIn(pad, rollout_ids)
        text_only = [{"role": "user", "content": "hello"}]
        self.assertEqual(
            builder.build_initial_tokens(text_only),
            normalize_token_ids(tokenizer.apply_chat_template(text_only, tokenize=True, add_generation_prompt=True)),
        )

    def test_media_features_match_the_prompt_call_and_the_pad_is_banned(self):
        import torch
        from PIL import Image
        from transformers import AutoProcessor

        from verl.utils.tokenizer import build_multimodal_processor_inputs, media_features, media_pad_token_id
        from verl.workers.rollout.utils import get_vision_placeholder_token_ids

        processor = AutoProcessor.from_pretrained(K3_EXPORT, trust_remote_code=True)
        image = Image.new("RGB", (70, 42), (0, 0, 250))
        features = media_features(processor, [image])
        out = build_multimodal_processor_inputs(processor, text="<|kimi_image_placeholder|> x", images=[image])
        self.assertTrue(torch.equal(features["pixel_values"], out["pixel_values"]))
        self.assertTrue(torch.equal(features["grid_thws"], out["grid_thws"]))
        self.assertEqual(media_features(processor, []), {})
        self.assertEqual(get_vision_placeholder_token_ids(processor), [media_pad_token_id(processor)])


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

    def test_multi_modal_info_falls_back_without_qwen_vl_utils(self):
        import sys
        from unittest import mock

        from PIL import Image

        from verl.utils.dataset.rl_dataset import RLHFDataset

        image = Image.new("RGB", (28, 14), (9, 9, 9))
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "?"}]}]
        with mock.patch.dict(sys.modules, {"qwen_vl_utils": None}):
            images, videos, audios = RLHFDataset._process_multi_modal_info(messages, image_patch_size=14, config=None)
        self.assertEqual([im.size for im in images], [(28, 14)])
        self.assertIsNone(videos)
        self.assertIsNone(audios)
