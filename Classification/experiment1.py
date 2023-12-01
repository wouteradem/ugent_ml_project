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
    """Plots training curves of a results dictionary.

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
    plt.show()


if __name__ == '__main__':
    cfg = Configuration()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup data directories
    train_dir = cfg.image_path / "train"
    test_dir = cfg.image_path / "test"
    print("Train dir: ", train_dir)
    print("Test dir: ", test_dir)

    # AUTO CREATION.
    # Get a set of pretrained model weights.
    weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
    print("Weights: ", weights)

    # Get the transforms used to create our pretrained weights
    auto_transforms = weights.transforms()
    print(auto_transforms)

    # Create training and testing DataLoaders as well as get a list of class names
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                   test_dir=test_dir,
                                                                                   transform=auto_transforms,
                                                                                   batch_size=32)

    print("Train dataloader:", train_dataloader)
    print("Test dataloader:", test_dataloader)
    print("Class labels:", class_names)

    # Get a model.
    model = torchvision.models.efficientnet_b1(weights)

    # Print a summary using torchinfo (uncomment for actual output)
    # summary(model=model,
    #        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
    #        col_names=["input_size", "output_size", "num_params", "trainable"],
    #        col_width=20,
    #        row_settings=["var_names"]
    # )

    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in model.features.parameters():
        param.requires_grad = False

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=output_shape,  # same number of output units as our number of classes
                        bias=True)).to(device)

    summary(model=model,
            input_size=(32, 3, 224, 224),
            # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
            )

    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Set the random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Start the timer
    from timeit import default_timer as timer

    start_time = timer()

    # Setup training and save the results
    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=5,
                           device=device)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

    # Plot the loss curves of our model
    plot_loss_curves(results)
