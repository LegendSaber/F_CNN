import os


class Configure:
    # 设置数据集的路径
    base_dir = ""
    train_dir = os.path.join(base_dir, "train_gray_images")

    batch_size = 8
    num_workers = 2
    epochs = 40
    lr = 1e-3
    decay = 0.0005
    momentum = 0.9
    num_classes = 10

    folder = 2
    random_state = 1900
