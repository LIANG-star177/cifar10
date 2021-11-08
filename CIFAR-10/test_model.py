from model import *
from data_loader import *
# 测试模型：
# 用 准确率 来评价神经网络：
correct = 0.0  # 测试数据中正确个数
total = 0.0  # 总共测试数据数量

with torch.no_grad():  # 不需要梯度
    for data in testloader:
        images, labels = data
        net = net.to(device)
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        # 统计正确数量和总共数量
        correct += (predicted == labels).sum()
        total += labels.size(0)

print('准确率：',float(correct) / total)
#
# #用  输出图形的类标签  来评价神经网络：
# dataiter = iter(testloader)
# images, labels = dataiter.next()
# net = net.to(device)
# images = images.to(device)
# labels = labels.to(device)
# imshow(torchvision.utils.make_grid(images))
# print('原始类: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
#
# print('预测类: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
#
# #用  输出图形的类标签  来评价神经网络：
# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# imshow(torchvision.utils.make_grid(images))
# print('原始类: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# net = net.to(device)
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
#
# print('预测类: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

#用  每个类预测准确率  来评价神经网络：
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        net = net.to(device)
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('%5s的准确率 : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
