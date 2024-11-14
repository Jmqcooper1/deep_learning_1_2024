# import Dict type
from typing import Dict


def plot_training_curves(logging_dict: Dict) -> None:
    """
    Plots training curves for the model.

    Args:
        logging_dict (Dict): Dictionary containing training curves.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(logging_dict["losses"]["train"], label="Training")
    plt.plot(logging_dict["losses"]["validation"], label="Validation")
    plt.title("Loss over time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(logging_dict["accuracies"]["train"], label="Training")
    plt.plot(logging_dict["accuracies"]["validation"], label="Validation")
    plt.title("Accuracy over time")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
