import openvino.runtime


class OpenvinoModel(openvino.runtime.Core):

    def __init__(self, file=Path('yolov7.xml')):
        super(OpenvinoModel, self).__init__()
        self._file = file
        self._device = self.available_devices[-1]
        # 加载模型参数
        self.onnx_model = self.read_model(file)
        self.complied_model = self.compile_model(self.onnx_model, device_name=self._device,
                                                 config=self.get_config())
        self.get_io()

    def __call__(self, *arrays):
        return self.complied_model(arrays)[self.io_node[-1]]

    def get_device(self):
        return {device: self.get_property(device, 'FULL_DEVICE_NAME')
                for device in self.available_devices}

    def get_config(self, cache_dir=Path('__openvino_cache__')):
        if self._device != 'CPU':
            cache_dir.mkdir(exist_ok=True)
            return {'CACHE_DIR': str(cache_dir)}

    def get_io(self):
        self.io_node = tuple(map(
            lambda key: getattr(self.complied_model, key)(0),
            ['input', 'output']
        ))
        self.io_shape = tuple(map(
            lambda node: tuple(node.shape), self.io_node
        ))

    def export(self):
        from openvino.offline_transformations import serialize
        serialize(self.onnx_model,
                  model_path=self._file.with_suffix('.xml'),
                  weights_path=self._file.with_suffix('.bin'))
