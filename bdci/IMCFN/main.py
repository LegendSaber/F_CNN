import os
from Configure import Configure
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MalwareDataset import MalwareDataset, get_data
from sklearn.model_selection import KFold
from VGG import vgg16


# 冻结层
def set_parameter_requires_grad(model):
    count = 0
    for param in model.parameters():
        param.requires_grad = False
        if param.size()[0] == 512:
            count += 1
        if count == 6:
            break


def train(epoch):
    for batch_idx, data in enumerate(train_loader, 0):
        optimizer.zero_grad()

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        y_pred = modeler(inputs)
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
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = modeler(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target.long()).sum()
    acc = 1.0 * 100 * correct / total
    print('测试精度: %f%% [%d/%d]' % (acc, correct, total))

    return acc


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    conf = Configure()

    image_data, y_data = get_data(conf.train_dir)
    kf = KFold(n_splits=conf.folder, shuffle=True, random_state=conf.random_state)

    k = 1
    total_acc = 0.0
    for (train_index, test_index) in kf.split(image_data):
        modeler = vgg16(pretrained=True, num_classes=conf.num_classes)
        set_parameter_requires_grad(modeler)
        modeler.to(device)
        optimizer = torch.optim.SGD(modeler.parameters(), lr=conf.lr,
                                    weight_decay=conf.decay, momentum=conf.momentum)  # 定义优化器

        train_image = []
        train_y_data = []
        for i in train_index:
            train_image.append(image_data[i])
            train_y_data.append(y_data[i])
        train_dataset = MalwareDataset(train_image, train_y_data, True)
        train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2)

        test_image = []
        test_y_data = []
        for i in test_index:
            test_image.append(image_data[i])
            test_y_data.append(y_data[i])
        test_dataset = MalwareDataset(test_image, test_y_data, False)
        test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=2)

        print("=====================开始训练模型================")
        for i in range(conf.epochs):
            train(i)
        print("=====================模型训练完成================")

        print("第%d折测试结果:" % k)
        total_acc += test()
        k = k + 1
    print("%d折交叉验证的平均精度: %f%%" % (conf.folder, total_acc / conf.folder))