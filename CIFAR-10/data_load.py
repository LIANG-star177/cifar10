import torch
import torchvision
import torchvision.transforms as transforms

#图片信息转化为tensor,同时对其进行标准化，前面为mean，后面为std，因为是3通道，所以有三个
transforms=transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#训练数据
# #下载数据，规定目标路径，train为true时建立训练集，
#  否则建立测试集，转移过来的数据经过transform格式转化
trainset=torchvision.datasets.CIFAR10(root='./data',train=True,
                                      download=True,transform=transforms)

# 下载后的数据集进行加载，batch_size为每次输入数据的行数，默认为1，
# shuffle为true，在每次训练数据迭代时进行随即洗牌，
# num_workers表示导入数据的进程数
trainloader=torch.utils.data.DataLoader(trainset,batch_size=100,shuffle=True,
                                        num_workers=0)
testset=torchvision.datasets.CIFAR10(root='./data',train=False,
                                     download=True,transform=transforms)
testloader=torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False,
                                       num_workers=0)

classes=('plane','car','bird','cat','deer','dog','frog','herse','ship','truck')

import numpy as np
import matplotlib.pyplot as plt
#定义图像展示函数
def data_show(img):
    #初始图像tensor为【通道，高度，宽度】
    #反归一处理，此时tensor为【宽度，高度，通道】
    img=img/2+0.5
    npimg=img.numpy()
    #图片转置,此时变为【高度，宽度，通道】
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

#传入数据用于显示
#随即加载一个minibatch
dataiter=iter(trainloader)
images,labels=dataiter.next()
data_show(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(100)))

