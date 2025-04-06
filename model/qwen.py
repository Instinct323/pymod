from typing import Callable, Union, Tuple, List

import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from utils import *


class QwenVL:
    device = property(lambda self: self.model.device)

    def __init__(self,
                 pretrained_model_name_or_path: str,
                 pixels_range: Tuple[int, int] = [256 * 28 * 28, 384 * 28 * 28],
                 torch_dtype: torch.dtype = "auto",
                 device_map: Union[str, torch.device] = "auto"):
        pixels_range = pixels_range or (None,) * 2
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path, use_fast=True, min_pixels=pixels_range[0], max_pixels=pixels_range[1]
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, device_map=device_map
        )

    def get_input_tensor(self, messages, batch_inference: bool = False):
        # fixme: batch_inference 不能正常使用
        texts: List[str] = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                            for msg in (messages if batch_inference else [messages])]
        # List[PIL.Image.Image]
        images, videos = process_vision_info(messages)
        # transformers.feature_extraction_utils.BatchFeature
        #   `input_ids`: 输入文本的 token ID
        #   `attention_mask`: bool mask, 用于指示哪些 token 是非填充的
        #   `pixel_values`: 输入图像的像素值
        #   `image_grid_thw`: 时间维度, 高度、宽度上的 patch 数量
        return self.processor(text=texts, images=images, videos=videos, padding=True, return_tensors="pt").to(self.device)

    def generate(self, inputs, max_new_tokens: int, return_ids: bool = False):
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        ids = [outi[len(ini):] for ini, outi in zip(inputs.input_ids, generated_ids)]
        return ids if return_ids else self.processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def fetch_output(self, forward: Callable):
        queue = []
        hook = self.model.get_output_embeddings().register_forward_hook(lambda *args: queue.append(args[2]))
        ret = forward()
        # 模型生成结束, 获取输出
        hook.remove()
        queue[0] = queue[0][:, -1:]
        return ret, torch.cat(queue, dim=1)


if __name__ == '__main__':
    model = QwenVL("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16)

    messages = [
        make_content("user",
                     ("image", "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"),
                     ("text", "描述这张图片"))
    ]
    inputs = model.get_input_tensor(messages)
    print(model.generate(inputs, 128))
