import scipy.io as scio
import torch

person_train = 3


def __data_train__():
    tensor = torch.Tensor()
    for person in range(person_train):
        mat = scio.loadmat('mat/%02d.mat' % (person + 1))
        mat = mat['ans']
        size = mat.size
        for i in range(size):
            t = torch.from_numpy(mat[i][0].reshape(1, 1, 9, 9, 256)).to(torch.float32)
            tensor = torch.cat((tensor, t), dim=0)
    print(tensor.shape)
    return tensor


def __label_train__():
    labels = torch.LongTensor()
    for person in range(person_train):
        mat = scio.loadmat('mat/%02d_label.mat' % (1 + person))
        data = mat['labels']
        flatten = data.reshape(-1)
        flatten -= 1
        t = torch.from_numpy(flatten).to(dtype=torch.long)
        labels = torch.cat((labels, t), dim=0)
    print(labels.shape)
    return labels


if __name__ == '__main__':
    __data_train__()
    __label_train__()
