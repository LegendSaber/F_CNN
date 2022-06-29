import os


class Configure:
    # 设置数据集的路径
    base_path = ""
    train_gray_path = os.path.join(base_path, "train_gray_images")
    test_gray_path = os.path.join(base_path, "test_gray_images")
    submit_path = os.path.join(base_path, "submit.csv")

    is_train = True     # 用来设置是训练模型还是测试模型
    batch_size = 8
    num_workers = 2
    epochs = 40
    lr = 1e-3
    decay = 0.0005
    momentum = 0.9
    model_path = "IMCFN.pth"
    num_classes = 9
