from data_load import *
import torch.nn as nn
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=torch.load('cifar10.pkl')
#测试集中正确数量
correct=0
#测试数据总量
total=0
with torch.no_grad():
    for data in testloader:
        images,labels=data
        # net = nn.DataParallel(net)
        # net = net.cuda()vim
        # images = images.cuda()
        # labels = labels.cuda()
        net=net.to(device)
        images=images.to(device)
        labels=labels.to(device)
        outputs=net(images)
        #输出每行的最大值和他的索引，这里只需要得到索引
        _,predicted=torch.max(outputs.data,1)
        correct+=(predicted==labels).sum()
        total+=labels.size(0)

print("Accuracy：%d %%" %(100*correct/total))

class_correct=list(0. for i in range(10))
class_total=list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images,labels=data
        # net = nn.DataParallel(net)
        # net = net.cuda()
        # images = images.cuda()
        # labels = labels.cuda()
        net=net.to(device)
        images=images.to(device)
        labels=labels.to(device)
        outputs=net(images)
        _,predicted=torch.max(outputs,1)
        # print(predicted)
        c=(predicted==labels).squeeze()
        #由于batch_size为4，所以4个为1组
        for i in range(100):
            label=labels[i]
            # print(c[i].item())
            #遍历所有label，按label计算对应的图片有没有推测正确，
            # c[i].item在推测正确时为true，即1，错误时false，为0.
            class_correct[label]+=c[i].item()
            class_total[label]+=1

for i in range(10):
    print('%5s的准确率 : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))