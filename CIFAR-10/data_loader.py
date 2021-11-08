import torch
import torchvision
#transforms 定义了一系列数据转化形式，并对数据进行处理
import torchvision.transforms as transforms

#定义归一化方法：
#transforms.Compose():将多个transforms组合起来使用
transform=transforms.Compose(
    [transforms.ToTensor(),                            #传入数据转化成张量形式
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  #定义归一化方法
     #transforms.Normalize((mean),(std)):用给定的均值和标准差对每个通道数据进行归一化：
     #归一方式：  (input[channel]-mean[channel])/std[channel]
    ]
)

#训练数据集：  CIFAR10数据集
trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
 #root（string）：数据集的根目录在哪里
 #train（bool，optional）：如果为True，则创建数据集training.pt，否则创建数据集test.pt。
 #download（bool，optional）：如果为true，则从Internet下载数据集并将其放在根目录中。如果已下载数据集，则不会再次下载。
 #transform（callable ，optional）：一个函数/转换，它接收PIL图像并返回转换后的版本。
 #target_transform（callable ，optional）：接收目标并对其进行转换的函数/转换。
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=0)
 #dataset(dataset):输入的数据类型
 #batch_size（数据类型 int）:每次输入数据的行数，默认为1。
 #shuffle（数据类型 bool）：洗牌。默认设置为False。在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，
 #num_workers（数据类型 Int）：工作者数量，默认是0。使用多少个子进程来导入数据。

#测试数据集：
testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=0)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#定义显示方法：
import numpy as np
import matplotlib.pyplot as plt
def imshow(img):
    #输入数据：类型(torch.tensor[c,h,w]：[宽度，高度，颜色图层])
    img=img/2+0.5                #反归一处理
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))   #图像进行转置，变为[h,w,c]
    plt.show()

#加载图像：
dataiter=iter(trainloader)        #随机加载一个mini batch
images,labels=dataiter.next()

#显示图像：
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

