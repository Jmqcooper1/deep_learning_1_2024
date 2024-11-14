################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch

from plot import plot_training_curves


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1) if len(targets.shape) > 1 else targets
    accuracy = np.mean(predicted_classes == true_classes)
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    total_accuracy = 0
    num_samples = 0

    for batch_inputs, batch_targets in data_loader:
        batch_inputs = batch_inputs.reshape(batch_inputs.shape[0], -1)

        batch_predictions = model.forward(batch_inputs)
        batch_accuracy = accuracy(batch_predictions, batch_targets)
        total_accuracy += batch_accuracy * batch_inputs.shape[0]
        num_samples += batch_inputs.shape[0]

    avg_accuracy = total_accuracy / num_samples
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def validate(model, val_loader, loss_module):
    val_losses = []
    for batch_inputs, batch_targets in val_loader:
        batch_inputs = batch_inputs.reshape(batch_inputs.shape[0], -1)
        outputs = model.forward(batch_inputs)
        val_loss = loss_module.forward(outputs, batch_targets)
        val_losses.append(val_loss)

    val_accuracy = evaluate_model(model, val_loader)
    return np.mean(val_losses), val_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=True
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    input_dim = 32 * 32 * 3
    model = MLP(input_dim, hidden_dims, 10)
    best_model = None
    loss_module = CrossEntropyModule()
    val_accuracies = []
    best_val_accuracy = 0
    test_accuracy = 0
    logging_dict = {
        "losses": {"train": [], "validation": []},
        "accuracies": {"train": [], "validation": []},
    }

    train_loader = cifar10_loader["train"]
    val_loader = cifar10_loader["validation"]
    test_loader = cifar10_loader["test"]

    for epoch in tqdm(range(epochs)):
        train_loss = []
        train_accuracies = []
        val_losses = []

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.reshape(batch_inputs.shape[0], -1)

            outputs = model.forward(batch_inputs)
            loss = loss_module.forward(outputs, batch_targets)

            grad_outputs = loss_module.backward(outputs, batch_targets)
            model.backward(grad_outputs)

            for module in model.modules:
                if hasattr(module, "params"):
                    for param in module.params:
                        module.params[param] -= lr * module.grads[param]

            train_loss.append(loss)
            train_accuracies.append(accuracy(outputs, batch_targets))

        val_loss, val_accuracy = validate(model, val_loader, loss_module)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)

        logging_dict["losses"]["train"].append(np.mean(train_loss))
        logging_dict["losses"]["validation"].append(np.mean(val_losses))
        logging_dict["accuracies"]["train"].append(np.mean(train_accuracies))
        logging_dict["accuracies"]["validation"].append(val_accuracy)

    model = best_model
    test_accuracy = evaluate_model(model, test_loader)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--hidden_dims",
        default=[128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"',
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the CIFAR10 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    print(f"Test accuracy of best model: {test_accuracy * 100}%")
    plot_training_curves(logging_dict)
