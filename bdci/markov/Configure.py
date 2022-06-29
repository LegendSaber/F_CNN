import os


class Configure:
    base_dir = ""
    train_dir = os.path.join(base_dir, "train_markov_images")

    # for ResNet34
    lr = 1e-3
    epochs = 30

    # for other model
    # lr = 1e-4
    # epochs = 25

    num_classes = 10
    batch_size = 32
    folder = 2
    random_state = 1900
