import os
from pathlib import Path


class Configuration:

    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()

    data_path = Path("exp3_data/")
    image_path = data_path / "classification/"
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Model transformation type.
    transform_type = 'auto'

    # Model optimizer
    #optimizer = 'adam'
    optimizer = 'SGD'

    # Model parameters.
    learning_rate = 0.001
    seed = 42
    nr_epochs = 100

