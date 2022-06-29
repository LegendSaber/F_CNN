import torch
import torch.nn as nn
from ResNet import resnet34


class F_CNN(nn.Module):
    def __init__(self, num_classes):
        super(F_CNN, self).__init__()

        # 特征抽取层
        self.gray_resnet = resnet34()
        self.markov_resnet = resnet34()

        # 降维层
        self.gray_perceptron = nn.Sequential(nn.Linear(512, 256),
                                             nn.ReLU(True))
        self.markov_perceptron = nn.Sequential(nn.Linear(512, 256),
                                               nn.ReLU(True))
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )

    def forward(self, gray_image, markov_image):
        # 从图像中抽取特征
        gray_image = self.gray_resnet(gray_image)
        markov_image = self.markov_resnet(markov_image)

        # 将抽取到的特征进行降维
        gray_image = self.gray_perceptron(gray_image)
        markov_image = self.markov_perceptron(markov_image)

        # 将降维得到的特征进行拼接后进行分类
        image = torch.cat((gray_image, markov_image), dim=1)
        output = self.classifier(image)

        return output


if __name__ == '__main__':
    pass
