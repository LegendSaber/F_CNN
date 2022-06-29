import os


class Configure:
    base_path = ""

    train_markov_path = os.path.join(base_path, "train_markov_images")
    train_gray_path = os.path.join(base_path, "train_gray_images")

    batch_size = 8
    epochs = 75
    lr = 0.005
    decay = 0.0005
    momentum = 0.9
    num_workers = 2
    num_classes = 10

    folder = 2
    random_state = 1900
