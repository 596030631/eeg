import torch
import torch.nn as nn
import torch.optim as opt

from datasets import __data_train__, __label_train__
# Get cpu or gpu device for training.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
from net import EEGNet3D


def __train__():
    train_data = __data_train__()
    train_label = __label_train__()
    eegnet3d = EEGNet3D()
    # eegnet3d = torch.load('model_weights-900.pth')

    eegnet3d.train()

    optimizer = opt.SGD(eegnet3d.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数

    for e in range(1001):
        optimizer.zero_grad()
        pred = eegnet3d.forward(train_data)
        loss = loss_fn(pred, train_label)
        loss.backward()
        optimizer.step()
        print('epoch %03d loss %f' % (e, loss.data))

        if e + 1 % 500 == 0:
            torch.save(eegnet3d, 'model_weights-%d.pth' % e)
            print("模型保存完成")


if __name__ == '__main__':
    __train__()
