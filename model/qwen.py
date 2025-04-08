from functools import partial
from typing import Callable, Union, Tuple, List

import PIL.Image
import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLProcessor

from .utils import *


class QwenVL:
    device = property(lambda self: self.model.device)

    def __init__(self,
                 pretrained_model_name_or_path: str,
                 pixels_range: Tuple[int, int] = [4 * 28 * 28, 384 * 28 * 28],
                 torch_dtype: torch.dtype = "auto",
                 device_map: Union[str, torch.device] = "auto"):
        pixels_range = pixels_range or (None,) * 2
        self.processor: Qwen2_5_VLProcessor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path, use_fast=True, min_pixels=pixels_range[0], max_pixels=pixels_range[1]
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, device_map=device_map
        )
        # 冻结模型参数
        for k, v in self.model.named_parameters(): v.requires_grad = False

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

    def generate(self, inputs, max_new_tokens: int, requires_grad: bool = False):
        generate = self.model.generate
        if requires_grad: generate = partial(generate.__wrapped__, self.model)

        generated_ids = generate(**inputs, max_new_tokens=max_new_tokens)
        ids = [outi[len(ini):] for ini, outi in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def fetch_output(self, forward: Callable):
        outq = []
        hook = self.model.get_output_embeddings().register_forward_hook(lambda *args: outq.append(args[2]))
        ret = forward()  # self.model(**inputs)
        # 模型生成结束, 获取输出
        hook.remove()
        outq[0] = outq[0][:, -1:]
        return ret, torch.cat(outq, dim=1)

    def reshape_pixels(self, pixel_values, image_grid_thw, channel: int = 3):
        # Qwen2VLImageProcessorFast._preprocess
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size

        t, h, w = image_grid_thw[0]
        pixel_values = pixel_values.view(t, h // merge_size, w // merge_size,
                                         merge_size, merge_size, channel,
                                         -1, patch_size, patch_size)
        pixel_values = pixel_values.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
        return pixel_values.flatten(6, 8).flatten(3, 5).flatten(0, 1)


if __name__ == '__main__':
    model = QwenVL("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16)

    fetch_grad = True

    image = PIL.Image.open("/media/tongzj/Data/Information/Source/image/Travel/东北/东北-长白山12.jpg").resize([336, 224])
    messages = [
        make_content("user",
                     ("image", image),
                     ("text", "描述这张图片"))
    ]
    inputs = model.get_input_tensor(messages)

    if not fetch_grad:
        ret = model.generate(inputs, 128)

    else:
        from pymod.zjdl.utils.deci_weight import DecisionWeight
        import numpy as np

        def save_grad(inputs, file):
            # grad: [B, C, H, W] -> [H, W]
            grad = model.reshape_pixels(inputs.pixel_values.grad, inputs.image_grid_thw).cpu().numpy()
            grad *= np.array([0.299, 0.587, 0.114])[..., None, None]
            grad = np.abs(grad).sum(axis=-3).mean(axis=0)
            print(f"{file}: {grad.mean()}, {grad.max()}")

            cmap = DecisionWeight.to_rgb(grad)
            cmap = PIL.Image.blend(image.resize(grad.shape[::-1]), PIL.Image.fromarray(cmap), .5)
            # cmap.save(file)
            import matplotlib.pyplot as plt
            plt.imshow(cmap)

        inputs.pixel_values.requires_grad_(True)
        # out: [1, token_size, vocab_size]
        ret, out = model.fetch_output(lambda: model.generate(inputs, 128, requires_grad=True))
        out = out[0].max(dim=-1)[0]

        file_len = int(np.ceil(np.log10(len(out) + 1)).item())
        out.sum().backward()
        save_grad(inputs, f"tmp/{'0'.zfill(file_len)}.png")

        # for i, x in enumerate(out):
        # inputs.pixel_values.grad.zero_()
        # x.backward(retain_graph=True)
        # save_grad(inputs, f"tmp/{str(i + 1).zfill(file_len)}.png")
    print(ret)
