import onnxruntime as ort
import torch
import torch.onnx


def onnx_simplify(src, new=None):
    ''' onnx 模型简化'''
    import onnxsim, onnx
    model, check = onnxsim.simplify(onnx.load(src))
    assert check, 'Failure to Simplify'
    onnx.save(model, new if new else src)


class OnnxModel(ort.InferenceSession):
    ''' onnx 推理模型
        provider: 优先使用 GPU'''
    device = property(fget=lambda self: self.get_providers()[0][:-17])

    def __init__(self, src):
        for pvd in ort.get_available_providers():
            try:
                super().__init__(str(src), providers=[pvd])
                break
            except:
                pass
        assert self.get_providers(), 'No available Execution Providers were found'
        # 参考: ort.NodeArg
        self.io_node = list(map(list, (self.get_inputs(), self.get_outputs())))
        self.io_name = [[n.name for n in nodes] for nodes in self.io_node]
        self.io_shape = [[n.shape for n in nodes] for nodes in self.io_node]

    def __call__(self, *inputs):
        input_feed = {name: x for name, x in zip(self.io_name[0], inputs)}
        return self.run(self.io_name[-1], input_feed)

    @classmethod
    def from_torch(cls, model, args, dst, **export_kwd):
        args = (args,) if isinstance(args, torch.Tensor) else args
        torch.onnx.export(model, args, dst, opset_version=11, **export_kwd)
        return cls(dst)


if __name__ == '__main__':
    from pathlib import Path
    import os
    from torch import nn

    model = nn.Conv2d(3, 8, 3)
    file = Path(os.getenv('lab')) / 'data/exp.onnx'
    onnxm = OnnxModel.from_torch(model, torch.rand([1, 3, 48, 48]), file)

    print(onnxm.device)
    print(onnxm)
