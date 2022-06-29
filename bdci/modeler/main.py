import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MalwareDataset import MalwareDataset, get_data
from sklearn.model_selection import KFold
from Configure import Configure
from F_CNN import F_CNN


def train(epoch):
    for batch_idx, data in enumerate(train_loader, 0):
        optimizer.zero_grad()

        gray_inputs, markov_inputs, labels = data
        gray_inputs, markov_inputs, labels = gray_inputs.to(device), markov_inputs.to(device), labels.to(device)

        y_pred = modeler(gray_inputs, markov_inputs)
        loss = F.cross_entropy(y_pred, labels.long())

        if batch_idx % 100 == 99:
            print(epoch, loss.item())
        loss.backward()
        optimizer.step()


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            gray_inputs, markov_inputs, target = data
            gray_inputs, markov_inputs, target = gray_inputs.to(device), markov_inputs.to(device), target.to(device)
            outputs = modeler(gray_inputs, markov_inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target.long()).sum()
    acc = 1.0 * 100 * correct / total
    print('测试精度: %f%% [%d/%d]' % (acc, correct, total))

    return acc


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    conf = Configure()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gray_image_path, markov_image_path, y_data = get_data(conf.train_gray_path, conf.train_markov_path)
    kf = KFold(n_splits=conf.folder, shuffle=True, random_state=conf.random_state)

    k = 1
    total_acc = 0.0
    for (train_index, test_index) in kf.split(gray_image_path):
        modeler = F_CNN(num_classes=conf.num_classes)
        modeler.to(device)
        optimizer = torch.optim.SGD(modeler.parameters(), lr=conf.lr,
                                    weight_decay=conf.decay, momentum=conf.momentum)  # 定义优化器

        train_gray_image = []
        train_markov_image = []
        train_y_data = []
        for i in train_index:
            train_gray_image.append(gray_image_path[i])
            train_markov_image.append(markov_image_path[i])
            train_y_data.append(y_data[i])
        train_dataset = MalwareDataset(train_gray_image, train_markov_image, train_y_data)
        train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2)

        test_gray_image = []
        test_markov_image = []
        test_y_data = []
        for i in test_index:
            test_gray_image.append(gray_image_path[i])
            test_markov_image.append(markov_image_path[i])
            test_y_data.append(y_data[i])
        test_dataset = MalwareDataset(test_gray_image, test_markov_image, test_y_data)
        test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=2)

        print("=====================开始训练模型================")
        for i in range(conf.epochs):
            train(i)
        print("=====================模型训练完成================")

        print("第%d折测试结果:" % k)
        total_acc += test()
        k = k + 1
    print("%d折交叉验证的平均精度: %f%%" % (conf.folder, total_acc / conf.folder))
