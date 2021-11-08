import torch.nn as nn
import torch.nn.functional as F
from data_loader import trainloader
import torch as t

# 定义Net类，运用nn中的Module模块，Module是继承的父类
class Net(nn.Module):
    # 定义神经网络结构   1x32x32
    def __init__(self):
        # super()调用下一个父类，并返回父类实例的一个方法
        super(Net, self).__init__()

        # 第一层：卷积层
        self.conv1 = nn.Conv2d(3, 6, 3)  # 输入频道：1   输出频道：6   3x3卷积核
        # 第二层：卷积层                      #上一层的输出频道是下一层的输入频道
        self.conv2 = nn.Conv2d(6, 16, 3)  # 输入频道：6   输出频道：16  3x3卷积核
        # 第三层：全连接层        28:32-2-2
        self.fc1 = nn.Linear(16 * 28 * 28, 512)  # 输入维度：16x28x28   输出维度：512
        # 第四层：全连接层
        self.fc2 = nn.Linear(512, 64)  # 输入维度：512   输出维度：64
        # 第五层：全连接层
        self.fc3 = nn.Linear(64, 10)  # 输入维度：64   输出维度：2

    # 定义神经网络数据流向:
    def forward(self, x):
        # 第一层卷积层：
        x = self.conv1(x)
        x = F.relu(x)  # 激活

        # 传递到第二层卷积层：
        x = self.conv2(x)
        x = F.relu(x)  # 激活

        # 传递到第三层全连接层：
        x = x.view(-1, 16 * 28 * 28)  # 改变x的形状
        x = self.fc1(x)
        x = F.relu(x)

        # 传递到第四层全连接层：
        x = self.fc2(x)
        x = F.relu(x)

        # 传递到第五层全连接层：
        x = self.fc3(x)

        return x


# 打印神经网络：
net = Net()
print(net)

import torch.optim as optim
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
#计算损失：
#定义损失函数：
criterion=nn.CrossEntropyLoss()         #交叉熵损失函数
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
#实现随机梯度下降    lr（float）:学习速率  momentum（float）:动量因子

for epoch in range(2):
    for i, data in enumerate(trainloader):
        images, labels = data  # 数据包括图像与标签两部分

        net = net.to(device)
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)

        loss = criterion(outputs, labels)  # 计算损失

        # 更新神经网络权重：
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 本次学习的梯度反向传递
        optimizer.step()  # 利用本次的梯度更新权值

        # 定期输出：
        if (i % 1000 == 0):
            print("Epoch:%d,Step:%d,Loss:%.3f" % (epoch, i, loss.item()))

print("Finished!")
