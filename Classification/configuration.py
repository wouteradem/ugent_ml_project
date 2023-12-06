from pathlib import Path


class Configuration:
    data_path = Path("exp1_data/")
    image_path = data_path / "classification/"
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Model transfor type.
    transform_type = 'manual'

    # Model parameters.
    learning_rate = 0.001
    seed = 42
    nr_epochs = 20

