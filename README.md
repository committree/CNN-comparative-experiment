把 MLP 换成 CNN：两层 nn.Conv2d + nn.MaxPool2d + 全连接头（可以直接用经典 LeNet 结构）。
记录对比：参数量 sum(p.numel() for p in model.parameters())、准确率、每轮耗时。
