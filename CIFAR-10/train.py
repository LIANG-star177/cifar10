from better_conv_model import *
from data_load import *
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

device=torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")
print(device)
#训练
for epoch in range(15):
    for i,data in enumerate(trainloader):
        # print(data)
        images,labels=data
        # net=nn.DataParallel(net)
        # net=net.cuda()
        # images=images.cuda()
        # labels=labels.cuda()
        net=net.to(device)
        images=images.to(device)
        labels=labels.to(device)
        outputs=net(images)
        loss=criterion(outputs,labels)
        #更新神经网络权重
        #梯度清零
        optimizer.zero_grad()
        #梯度反向传播
        loss.backward()
        #更新权值
        optimizer.step()

        #定期输出：
        if(i%100==0):
            print("Epoch:%d,Step:%d,Loss:%.3f"%(epoch+1,i,loss.item()))

print("Finish training")

#保存模型
torch.save(net,'cifar10.pkl')