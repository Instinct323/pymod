# Project Structure

```
├── config: yaml 配置文件
├── deploy: torch 模型转换
    ├── onnx_run: ONNX 模型管理
    └── openvino: OpenVINO 模型管理
├── engine: 训练相关的引擎
    ├── trainer: 神经网络训练器
    ├── loss: 损失函数 (对比损失，焦点损失)
    ├── crosstab: 混淆矩阵 (多分类)
    ├── evolve: 超参数进化算法 (惯性驱动, 贝叶斯优化)
    ├── scaling: EfficientNet 论文中的模型复合缩放
    ├── result: 训练过程信息的结构化存储方法
    ├── iou: Wise-IoU 的计算方法
    └── supervisor: 自监督学习 (e.g., SimSiam, MAE)、线性探测
├── model: 计算机视觉模型
    ├── common: 复现的 CNN、ViT 网络单元 (e.g., RepConv, PyramidViT)
    ├── model: yaml 文件配置的模型
    ├── fourier: 傅里叶特征映射 (相似度可视化, 图像重建)
    ├── utils: 网络单元的注册方法，局部变量的传递方法
    └── ema: Mean Teacher 的半监督学习方法
├── runs: 程序运行结果
└── utils: 拓展工具包
    ├── imgtf: 图像处理方法 (e.g., 颜色失真, 边界填充)
    ├── data: 数据集相关处理方法 (e.g., 留出法, 欠采样, 数据池)
    ├── gradcam: 梯度加权的类激活映射 (i.e., Grad-CAM)
    ├── prune: 非结构化剪枝
    ├── plot: 损失值曲面绘制, 参数利用率分析
    ├── rfield: 网络感受野可视化
    ├── tta: 测试自适应
    └── teacher: 快速知识蒸馏 (FKD) 的知识管理系统
```