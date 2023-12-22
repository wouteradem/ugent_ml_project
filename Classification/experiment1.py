import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchinfo import summary
from configuration import Configuration
import data_setup
import engine


# Plot loss curves of a model
def plot_loss_curves(results):
    """
    Plots training curves of a results dictionary.
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig("loss_curve.svg", dpi=150)


if __name__ == '__main__':
    cfg = Configuration()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get model weights.
    weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT

    # Set model transforms.
    if cfg.transform_type == 'auto':
        transforms = weights.transforms()
        train_data = torchvision.datasets.ImageFolder(cfg.train_dir, transform=transforms)
        test_data = torchvision.datasets.ImageFolder(cfg.test_dir, transform=transforms)
    elif cfg.transform_type == 'manual':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        train_data = torchvision.datasets.ImageFolder(cfg.train_dir, transform=transforms)
        test_data = torchvision.datasets.ImageFolder(cfg.test_dir, transform=transforms)
    elif cfg.transform_type == 'augmentation':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.TrivialAugmentWide(num_magnitude_bins=31),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # Get DataLoaders and classification labels.
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=cfg.train_dir,
                                                                                   test_dir=cfg.test_dir,
                                                                                   transform=transforms,
                                                                                   batch_size=32)
    # Get model.
    model = torchvision.models.efficientnet_b1(weights)

    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in model.features.parameters():
        param.requires_grad = False

    # Set the manual seeds
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # Transfer Learning step.
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(
            in_features=1280,
            out_features=len(class_names),
            bias=True
        )
    ).to(device)

    summary(
        model=model,
        input_size=(32, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )

    # Define loss and optimizer.
    loss_fn = nn.CrossEntropyLoss()
    if cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=0.0005)

    # Set random seeds.
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    # Start timer.
    from timeit import default_timer as timer
    start_time = timer()

    # Train model.
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=cfg.nr_epochs,
        device=device
    )

    # End timer.
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

    # Plot the loss curves of our model
    plot_loss_curves(results)
