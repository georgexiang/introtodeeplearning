# Lab 2 - Part 1: MNIST 手写数字分类（PyTorch）

本文档解释 [lab2/PT_Part1_MNIST.ipynb](../lab2/PT_Part1_MNIST.ipynb) 这个 notebook 的内容、实验步骤，以及你需要完成的事情。

## 实验目标

你使用 PyTorch 构建并训练两种神经网络，对 MNIST 手写数字（0-9）进行分类，并比较两者的准确率：

1. 全连接网络（FC / Dense）
2. 卷积神经网络（CNN）

数据集说明：

- 60,000 张训练图片 + 10,000 张测试图片
- 每张图片是 28x28 的灰度图

## 实验流程与待完成事项

notebook 中标注 `# TODO` 和 `'''TODO'''` 的位置都需要你补全代码。下面按步骤说明。

### 0. 环境准备

- 导入 PyTorch、torchvision、mitdeeplearning 等库。
- 填写 Comet API Key：将 `COMET_API_KEY = ""` 替换为你自己的 key，用于记录训练曲线。
- 检查 GPU 是否可用：`assert torch.cuda.is_available()`。

### 1.1 加载与可视化数据

- 使用 `datasets.MNIST` 下载数据，`ToTensor()` 把像素从 [0,255] 缩放到 [0,1]。
- 随机展示 36 张图片及其标签。

此步骤无需修改代码。

### 1.2 全连接网络

补全 `build_fc_model()`（Sequential 写法）：

- 补全第一层后的激活函数 `nn.ReLU()`。
- 补全第二个 `nn.Linear` 层，输出 10 个类别。

补全 `FullyConnectedModel`（subclass 写法）：

- `self.relu =` 填 `nn.ReLU()`。
- `self.fc2 =` 填 `nn.Linear(128, 10)`。
- 在 `forward()` 中补全经过 relu 和 fc2 的前向传播。

定义损失与优化器（可实验）：

- 已给出 `CrossEntropyLoss` 加 `SGD(lr=0.1)`。
- 尝试不同优化器和学习率，观察哪个准确率最高。

调用 `train(...)`：

- 将 `train('''TODO''')` 替换为 `train(fc_model, trainset_loader, loss_function, optimizer, EPOCHS)`。
- 目标训练准确率约为 97%。

补全 `evaluate(...)`：

- 把 images 和 labels 移动到 GPU。
- 前向传播得到 `outputs`。
- 累加 `test_loss`。
- 使用 `torch.argmax` 取预测类别。
- 统计 `correct_pred` 和 `total_pred`。
- 最后调用 `evaluate(fc_model, testset_loader, loss_function)`。

### 1.3 卷积神经网络（CNN）

补全 `CNN` 类：

- 定义两个卷积层 `nn.Conv2d` 和两个池化层 `nn.MaxPool2d`，参数参照 notebook 中的网络架构图。
- 定义输出层 `self.fc2 = nn.Linear(128, 10)`，输出 logits。
- 在 `forward()` 中补全第二组卷积和池化层，以及展平和两个全连接层。

补全 CNN 训练循环：

- `loss_function =` 填 `nn.CrossEntropyLoss()`。
- `logits =` 前向传播。
- `loss =` 计算损失。
- 反向传播三步：`optimizer.zero_grad()`、`loss.backward()`、`optimizer.step()`。

补全 CNN 评估与预测：

- 调用 `evaluate(...)`。
- 取第一张测试图的预测：`prediction = np.argmax(predictions_value)`。

### 1.4 结果可视化

- 使用滑块查看单张图片的预测概率分布。
- 网格展示多张图片：蓝色表示预测正确，灰色表示错误。

此步骤基本无需修改代码。

### 1.5 结论

- 对比 FC 与 CNN 的准确率，CNN 通常接近 99%，表现更好。
- 思考学习率、优化器等超参数对结果的影响。

## 核心要点总结

| 步骤 | 你要做的事 |
|------|-----------|
| 环境 | 填写 Comet API Key |
| FC 模型 | 补全激活函数、输出层、forward |
| 训练 | 正确调用 `train()` |
| 评估 | 补全 `evaluate()` 逻辑 |
| CNN | 补全卷积、池化、全连接层及 forward |
| CNN 训练 | 补全损失函数和反向传播三步 |
| 预测 | 使用 `argmax` 取最高概率类别 |

## 相关资源

- [MIT 6.S191 课程主页](http://introtodeeplearning.com)
- [MNIST 数据集](http://yann.lecun.com/exdb/mnist/)
- [PyTorch nn.Module 文档](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
