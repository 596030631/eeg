import torch

from datasets import __data_train__, __label_train__

model = torch.load('model_weights-1500.pth')
model.train()

train_data = __data_train__()
train_label = __label_train__()

p = model.forward(train_data)

err = 0
b = 0

for i in range(1200):
    v = 0.000000
    for k in range(4):
        if v < p[i][k]:
            v = p[i][k]
            b = k
    if b != train_label[i]: err += 1
    print("%f %f %f %f  pre=%d label=%d  right=%f" % (
    p[i][0], p[i][1], p[i][2], p[i][3], b, train_label[i], 1.00 * (1200 - err) / 1200))
