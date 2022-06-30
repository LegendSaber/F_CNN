import os


class Configure:
    base_path = ""

    # 马尔可夫图训练集和测试集路径
    train_markov_path = os.path.join(base_path, "train_markov_images")
    test_markov_path = os.path.join(base_path, "test_markov_images")

    # 灰度图训练集和测试集路径
    train_gray_path = os.path.join(base_path, "train_gray_images")
    test_gray_path = os.path.join(base_path, "test_gray_images")

    submit_file = os.path.join(base_path, "submit.csv")

    batch_size = 8
    epochs = 75

    lr = 0.005
    decay = 0.0005
    momentum = 0.9
    num_workers = 2
    num_classes = 9

    model_path = "F_CNN.pth"
    is_train = True
