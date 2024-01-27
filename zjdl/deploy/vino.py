from pathlib import Path
from typing import Union

import openvino.runtime as ov

VINO_CORE = ov.Core()
DEVICES = VINO_CORE.available_devices


class VinoModel(ov.CompiledModel):
    # OpenVINO™ Runtime API 2.0: https://docs.openvino.ai/2022.1/openvino_2_0_inference_pipeline.html
    cache_dir = Path('__openvino_cache__')

    def __init__(self,
                 xml: Path,
                 device: Union[int, str] = 1):
        assert xml.with_suffix('.bin').is_file(), 'bin file cannot be found'
        self.device = DEVICES[device] if isinstance(device, int) else device
        super().__init__(VINO_CORE.compile_model(
            VINO_CORE.read_model(xml), device_name=self.device, config=self.get_cfg()))
        # 获取输入输出信息
        self.io_node = [self.input(), self.output()]
        self.io_shape = list(map(lambda node: tuple(node.shape), self.io_node))

    def post_process(self, request: ov.ie_api.InferRequest):
        result = tuple(request.results.values())
        return result[0] if len(result) == 1 else result

    def __call__(self, *inputs):
        infer_request = self.create_infer_request()
        infer_request.infer(inputs)
        return self.post_process(infer_request)

    def infer_async(self, dataset, jobs=4):
        buffer = []
        queue = ov.AsyncInferQueue(self, jobs=jobs)
        queue.set_callback(lambda req, i: buffer.append((i, self.post_process(req))))
        # 启动异步推理
        for i, dat in enumerate(dataset): queue.start_async(dat, userdata=i)
        queue.wait_all(), buffer.sort()
        return buffer

    def get_cfg(self) -> dict:
        cfg = {}
        if self.device != 'CPU':
            self.cache_dir.mkdir(exist_ok=True)
            cfg['CACHE_DIR'] = str(self.cache_dir)
        return cfg

    @classmethod
    def from_onnx(cls, src, dst=Path('.'), half=False):
        ''' :param src: model .onnx file
            :param dst: Directory that stores the generated IR'''
        assert src.suffix == '.onnx'
        args = ['mo', f'-w {src}', f'-o {dst}']
        if half: args.append('--compress_to_fp16')
        os.system(' '.join(args))
        return cls(dst / src.with_suffix('.xml').name)


if __name__ == '__main__':
    from mod import *

    file = Path(r'D:\Information\Python\mod\zjdl\cfg\weights\yolov7.onnx')
    vinom = VinoModel.from_onnx(file, dst=Path('__pycache__'), half=False) \
        if 1 else VinoModel(Path(r'__pycache__\yolov7.xml'))

    inp = np.random.random(vinom.io_shape[0]).astype(np.float32)
    print(timer(10)(vinom)(inp))
