import os
import sys
from MalwareDataset import MalwareDataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import pandas as pd
from Configure import Configure
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


# 获取需要训练的参数
def train_param_number(model):
    train_num = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("train_num:%d" % train_num)


def load_model(model_path):
    if not os.path.exists(model_path):
        print("模型路径错误，模型加载失败")
        sys.exit(0)
    else:
        return torch.load(model_path)


def save_model(target_model, model_path):
    if os.path.exists(model_path):
        os.remove(model_path)
    torch.save(target_model, model_path)


def train(epoch):
    for batch_idx, data in enumerate(train_loader, 0):
        optimizer.zero_grad()                                   # 梯度清0

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        y_pred = modeler(inputs)                                # 前向传播
        loss = F.cross_entropy(y_pred, labels.long())           # 计算损失

        if batch_idx % 100 == 99:
            print("epoch=%d, loss=%f" % (epoch, loss.item()))

        loss.backward()                                         # 反向传播
        optimizer.step()                                        # 梯度更新


def test():
    df = pd.read_csv(conf.submit_path)
    with torch.no_grad():
        for inputs, file_name in test_loader:
            inputs = inputs.to(device)
            outputs = modeler(inputs)
            predicted = F.softmax(outputs.data, dim=1)
            data_len = len(inputs)
            for i in range(data_len):
                dict_res = {"Id": file_name[i], "Prediction1": 0, "Prediction2": 0,
                            "Prediction3": 0, "Prediction4": 0, "Prediction5": 0, "Prediction6": 0,
                            "Prediction7": 0, "Prediction8": 0, "Prediction9": 0}
                for j in range(9):
                    dict_res["Prediction" + str(j + 1)] = predicted[i][j].item()
                df = df.append(dict_res, ignore_index=True)
    df.to_csv(conf.submit_path, index=0)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    conf = Configure()

    test_dataset = MalwareDataset(conf.test_gray_path, False)
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size,
                             shuffle=False, num_workers=conf.num_workers)

    # 根据是否训练还选择是否加载保存的模型
    if conf.is_train:
        train_dataset = MalwareDataset(conf.train_gray_path, True)
        train_loader = DataLoader(train_dataset, batch_size=conf.batch_size,
                                  shuffle=True, num_workers=conf.num_workers)
        modeler = vgg16(pretrained=True, num_classes=conf.num_classes)
    else:
        print("=====================开始加载模型================")
        modeler = load_model(conf.model_path)
        print("=====================模型加载完成================")
    # train_param_number(modeler)
    set_parameter_requires_grad(modeler)
    # train_param_number(modeler)
    modeler.to(device)

    if conf.is_train:
        optimizer = torch.optim.SGD(modeler.parameters(), lr=conf.lr,
                                    weight_decay=conf.decay, momentum=conf.momentum)
        print("=====================开始训练模型================")
        for i in range(conf.epochs):
            train(i)
        print("=====================模型训练完成================")
        save_model(modeler, conf.model_path)
    print("=====================开始测试模型================")
    test()
    print("=====================模型测试完成================")