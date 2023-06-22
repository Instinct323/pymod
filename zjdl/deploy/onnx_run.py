import cv2 as cv
import numpy as np
import onnxruntime as ort
import torch.onnx


class Timer:
    repeat = 3

    def __new__(cls, fun, *args, **kwargs):
        import time
        start = time.time()
        for _ in range(cls.repeat): fun(*args, **kwargs)
        cost = (time.time() - start) / cls.repeat
        return cost * 1e3  # ms


def onnx_simplify(file, new=None):
    ''' onnx 模型简化'''
    import onnxsim, onnx
    model, check = onnxsim.simplify(onnx.load(file))
    assert check, 'Failure to Simplify'
    onnx.save(model, new if new else file)


class OnnxModel(ort.InferenceSession):
    ''' onnx 推理模型
        provider: 优先使用 GPU'''
    provider = 1 if ort.get_device() == 'GPU' else 0

    def __init__(self, file):
        provider = ort.get_available_providers()[self.provider]
        super(OnnxModel, self).__init__(str(file), providers=[provider])
        # 参考: ort.NodeArg
        self.io_node = list(map(list, [self.get_inputs(), self.get_outputs()]))
        self.io_name = [[node.name for node in nodes] for nodes in self.io_node]
        self.io_shape = [[node.shape for node in nodes] for nodes in self.io_node]

    def __call__(self, *arrays):
        input_feed = {name: x for name, x in zip(self.io_name[0], arrays)}
        return self.run(self.io_name[-1], input_feed)

    def fps(self):
        cost = self.get_profiling_start_time_ns()
        return 1e6 / cost

    @classmethod
    def test(cls, model, args, file, **export_kwargs):
        # 测试 Torch 的运行时间
        torch_output = model(*args).cpu().data.numpy()
        print(f'Torch: {Timer(model, *args):.2f} ms')
        # model: Torch -> onnx
        torch.onnx.export(model, args, file,
                          opset_version=11, **export_kwargs)
        # data: tensor -> array
        args = tuple(map(lambda x: x.cpu().data.numpy(), args))
        onnx_model = cls(file)
        # 测试 onnx 的运行时间
        onnx_output = onnx_model(*args)
        print(f'Onnx: {Timer(onnx_model, *args):.2f} ms')
        # 计算 Torch 模型与 onnx 模型输出的绝对误差
        abs_error = np.abs(torch_output - onnx_output).mean()
        print(f'Mean Error: {abs_error:.2f}')
        return onnx_model


if __name__ == '__main__':
    from torch import nn
    import torch


    class ReOrg(nn.Module):

        def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
            return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                              x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1)


    model = ReOrg()
    image = torch.rand([1, 3, 48, 48])
    onnx_model = OnnxModel.test(model, (image,), r'D:\Information\Python\Laboratory\data\exp.onnx')
