import os


class Configure:
    base_dir = ""
    train_dir = os.path.join(base_dir, "train_markov_images")
    test_dir = os.path.join(base_dir, "test_markov_images")
    result_dir = os.path.join(base_dir, "submit.csv")

    lr = 1e-3
    batch_size = 32
    decay = 1e-6
    epochs = 25
    # for DCNN
    # epochs = 20

    model_path = "markov.pth"
    num_classes = 9
    is_train = True
