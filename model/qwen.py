from functools import partial
from typing import Callable, Union, Tuple, List

import PIL.Image
import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLProcessor, Qwen2VLImageProcessorFast

from .utils import *


class QwenVL:
    patch_size = Qwen2VLImageProcessorFast.patch_size
    device = property(lambda self: self.model.device)

    def __init__(self,
                 pretrained_model_name_or_path: str,
                 patches_range: Tuple[int, int] = (16, 512),
                 torch_dtype: torch.dtype = "auto",
                 device_map: Union[str, torch.device] = "auto"):
        patches_range = patches_range or (None,) * 2
        self.processor: Qwen2_5_VLProcessor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path, use_fast=True,
            min_pixels=patches_range[0] * self.patch_size ** 2,
            max_pixels=patches_range[1] * self.patch_size ** 2
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, device_map=device_map
        )
        # 冻结模型参数
        for k, v in self.model.named_parameters(): v.requires_grad = False

    def get_input_tensor(self,
                         messages,
                         batch_inference: bool = False):
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

    def generate(self,
                 inputs,
                 max_new_tokens: int,
                 requires_grad: bool = False,
                 simplify: bool = True):
        generate = self.model.generate
        if requires_grad: generate = partial(generate.__wrapped__, self.model)

        generated_ids = generate(**inputs, max_new_tokens=max_new_tokens)
        ids = [outi[len(ini):] for ini, outi in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(ids, skip_special_tokens=simplify, clean_up_tokenization_spaces=False)

    def chat(self,
             messages: list,
             max_new_tokens: int):
        messages = messages.copy()
        messages.append(
            make_content("assistant", self.generate(self.get_input_tensor(messages), max_new_tokens))[0]
        )
        return messages

    def fetch_output(self,
                     forward: Callable):
        outq = []
        hook = self.model.get_output_embeddings().register_forward_hook(lambda *args: outq.append(args[2]))
        ret = forward()  # self.model(**inputs)
        # 模型生成结束, 获取输出
        hook.remove()
        outq[0] = outq[0][:, -1:]
        return ret, torch.cat(outq, dim=1)

    def reshape_pixels(self,
                       pixel_values,
                       image_grid_thw, channel: int = 3):
        # Qwen2VLImageProcessorFast._preprocess
        merge_size = self.processor.image_processor.merge_size

        t, h, w = image_grid_thw[0]
        pixel_values = pixel_values.view(t, h // merge_size, w // merge_size,
                                         merge_size, merge_size, channel,
                                         -1, self.patch_size, self.patch_size)
        pixel_values = pixel_values.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
        return pixel_values.flatten(6, 8).flatten(3, 5).flatten(0, 1)


class DevExpansion(QwenVL):
    manipulator = "hand"
    keyword = {"path": "最简的操作路径是否可见", "target": "操作目标是否可见"}

    def save_grad(self,
                  inputs,
                  image: PIL.Image.Image,
                  file: Union[str, Path]):
        from pymod.zjdl.utils.deci_weight import DecisionWeight
        import matplotlib.pyplot as plt
        import numpy as np

        # grad: [B, C, H, W] -> [H, W]
        grad = model.reshape_pixels(inputs.pixel_values.grad, inputs.image_grid_thw).cpu().numpy()
        grad *= np.array([0.299, 0.587, 0.114])[..., None, None]
        grad = np.abs(grad).sum(axis=-3).mean(axis=0)
        print(f"{file}: {grad.mean()=}, {grad.max()=}")

        cmap = DecisionWeight.to_rgb(grad)
        cmap = PIL.Image.blend(image.resize(grad.shape[::-1]), PIL.Image.fromarray(cmap), .5)
        cmap.save(file) if file else plt.imshow(cmap)

    def query(self,
              content: dict,
              id_: str = ""):
        ret_fmt = ",".join(f"{k}=*" for k in self.keyword)

        assert isinstance(content["image"], PIL.Image.Image)
        messages = [make_content("system",
                                 f"现在我将使用{self.manipulator}进行一次操作任务，你需要返回符合如下格式的文本：\n{ret_fmt}\n其中*应该是1/0，" +
                                 ", ".join(f"{k}表示{v}" for k, v in self.keyword)), content]
        print(messages)  # fixme: checkpoint

        inputs = model.get_input_tensor(messages)
        inputs.pixel_values.requires_grad_(True)
        # out: [1, token_size, vocab_size]
        ret, out = model.fetch_output(lambda: model.generate(inputs, len(ret_fmt), requires_grad=True))
        out = out[0].max(dim=-1)[0]

        for i, ans in enumerate(ret[0].split(",")):
            out[2 * i - 1].backward(retain_graph=True)
            self.save_grad(inputs, content["image"], f"tmp/{id_}-{ans}.png")
            inputs.pixel_values.grad.zero_()


if __name__ == '__main__':
    model = DevExpansion("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16)

    fetch_grad = False

    if not fetch_grad:
        image = PIL.Image.open("/media/tongzj/Data/Information/Source/image/Travel/东北/东北-长白山12.jpg")
        messages = [
            make_content("user",
                         ("image", image),
                         ("text", "描述这张图片"))
        ]
        messages = model.chat(messages)
        print(messages)

    else:
        image = PIL.Image.open("/media/tongzj/Data/Workbench/data/mani/3.jpeg")
        model.query(make_content("user",
                                 ("image", image),
                                 ("text", "把盖子拧到保温杯上")))
