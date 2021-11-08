from data_load import *
# 引用nn模块
import torch.nn as nn
import torch.nn.functional as  F

class Net(nn.Module):
    #搭建网络
    def __init__(self):
        #super()调用下一个父类，并返回父类实例的一个方法
        super(Net,self).__init__()
        # 卷积层1，输入2，输出16，卷积核5*5，
        # stride=1时，填充padding=(kernel_size-1)/2=2,最后尺寸+2
        # maxpooling为2， 3*32*32->16*14*14->16*16*16
        self.conv1=nn.Sequential(
            nn.Conv2d(3,16,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 卷积层2，输入16，输出32，卷积核5*5，
        # stride=1时，padding=(kernel_size-1)/2
        # maxpooling为2， 16*16*16->32**6*6->32*8*8
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 卷积层3，输入16，输出32，卷积核5*5，
        # stride=1时，padding=(kernel_size-1)/2,目的是维持尺寸，kernel_size不影响
        # maxpooling为2， 32*8*8->64*4*4
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 全连接层1，输入64，输出10
        self.out=nn.Linear(64*4*4,10)



    #定义神经网络数据流向
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        #将16*5*5的数据拉平
        x=x.view(x.size(0),-1)
        #第一层全连接,激活
        output=self.out(x)
        return output

net=Net()
print(net)

import torch.optim as optim
#定义损失函数,使用交叉熵损失函数
criterion=nn.CrossEntropyLoss()
#随机梯度下降，学习率为0.001，动量为0.9，防止局部最优
# optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
#Adam优化
optimizer=optim.Adam(net.parameters(),lr=0.001)