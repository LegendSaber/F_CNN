import os
import sys
from Configure import Configure
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MalwareDataset import MalwareDataset
import pandas as pd
from VGG import vgg16, vgg19
from ResNet import resnet34


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
        optimizer.zero_grad()

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        y_pred = modeler(inputs)
        loss = F.cross_entropy(y_pred, labels.long())
        if batch_idx % 100 == 96:
            print(epoch, loss.item())

        loss.backward()
        optimizer.step()


def test():
    df = pd.read_csv(conf.result_dir)
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
    df.to_csv(conf.result_dir, index=0)


# 单独对单模型的马尔科夫图进行测试
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf = Configure()

    test_dataset = MalwareDataset(conf.test_dir, False)
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=2)

    if conf.is_train:
        # modeler = vgg16(pretrained=True, num_classes=conf.num_classes)
        # modeler = vgg19(pretrained=True, num_classes=conf.num_classes)
        modeler = resnet34(pretrained=True, num_classes=conf.num_classes)
        train_dataset = MalwareDataset(conf.train_dir, True)
        train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2)
    else:
        print("===============开始加载模型===============")
        modeler = load_model(conf.model_path)
        print("===============模型加载完成===============")
    modeler.to(device)

    if conf.is_train:
        optimizer = torch.optim.Adam(modeler.parameters(), lr=conf.lr, weight_decay=conf.decay)
        print("===============开始训练模型===============")
        for i in range(conf.epochs):
            train(i)
        print("===============模型训练完成===============")
        save_model(modeler, conf.model_path)
    print("===============开始测试模型===============")
    test()
    print("===============模型测试完成===============")