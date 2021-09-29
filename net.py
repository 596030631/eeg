import torch
import torch.nn as nn
import torch.optim as opt


class EEGNet3D(nn.Module):
    def __init__(self):
        super(EEGNet3D, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv3d(in_channels=1,
                      out_channels=8,
                      kernel_size=(3, 3, 4),
                      stride=(1, 1, 2),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))
        )

        self.cnn2 = nn.Sequential(
            nn.Conv3d(in_channels=8,
                      out_channels=16,
                      kernel_size=(3, 3, 4),
                      stride=(1, 1, 2),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 1))
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 9 * 9 * 31, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.linear(x)
        x = torch.softmax(input=x, dim=-1)
        return x


loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数

if __name__ == '__main__':
    eegnet3d = EEGNet3D()
    optimizer = opt.SGD(eegnet3d.parameters(), lr=0.001, momentum=0.9)
    _input = torch.randn((1200, 1, 9, 9, 256))
    print(_input.shape)
    _label = torch.zeros(1200).to(torch.long)
    print(_label.shape)
    optimizer.zero_grad()
    pred = eegnet3d.forward(_input)
    loss = loss_fn(pred, _label)
    loss.backward()
    optimizer.step()
    print('loss %f' % loss.data)

    print("finish")
