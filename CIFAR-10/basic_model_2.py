from data_load import *
# 引用nn模块
import torch.nn as nn
import torch.nn.functional as  F

class Net(nn.Module):
    #搭建网络
    def __init__(self):
        #super()调用下一个父类，并返回父类实例的一个方法
        super(Net,self).__init__()

        #3*32*32经过6个3*3的filter，变为6*30*30
        self.conv1=nn.Sequential(nn.Conv2d(3,6,3),nn.ReLU())
        #6*30*30经过16个3*3的filter，变为16*28*28
        self.conv2=nn.Sequential(nn.Conv2d(6,16,3),nn.ReLU())
        #经过两次卷积，图片变为16*28*28，全连接层将它变成512
        self.fc1=nn.Linear(16*28*28,512)
        #第二层全连接，变为64
        self.fc2=nn.Linear(512,64)
        #最后一层全连接，输入为64，输出为10（因为最后要分成十类）
        self.fc3=nn.Linear(64,10)

    #定义神经网络数据流向
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        #将16*28*28的数据拉平
        x=x.view(x.size(0),-1)
        #第一层全连接,激活
        x=F.relu(self.fc1(x))
        #第二层全连接，激活
        x=F.relu(self.fc2(x))
        #第三层全连接,不用激活，作为输出
        x=self.fc3(x)
        return x

net=Net()
print(net)

import torch.optim as optim
#定义损失函数,使用交叉熵损失函数
criterion=nn.CrossEntropyLoss()
#随机梯度下降，学习率为0.001，动量为0.9，防止局部最优
# optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
#Adam优化
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)