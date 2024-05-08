import numpy as np

import matplotlib.pyplot as plt


def moving_average(data, window_size):
    """
    """
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def plot_losses(loss_history, labels=None):
    """
    """
    # Plotting
    plt.figure(figsize=(10, 6))

    if not labels:
        labels = ["Loss autoencoder", "Loss piggybacker"]

    tuple_size = len(loss_history[0])

    for i, label in zip(range(tuple_size), labels):
        loss_values = [float(vals[i]) for vals in loss_history]
        plt.plot(loss_values, label=label)

    plt.title('Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()


def plot_smoothed_losses(loss_history, window_size, labels=None):
    """
    """
    # Plotting
    plt.figure(figsize=(10, 6))

    if not labels:
        labels = ["Loss autoencoder", "Loss piggybacker"]

    tuple_size = len(loss_history[0])

    for i, label in zip(range(tuple_size), labels):

        # Loss values
        loss_values = [float(vals[i]) for vals in loss_history]

        # Calculate the moving average
        smooth_loss = moving_average(loss_values, window_size)

        # Adjust the length of original loss values to match the smoothed array
        adjusted_loss_values = loss_values[:len(smooth_loss)]

        # Plot
        lines = plt.plot(adjusted_loss_values, alpha=0.5, label=label)
        color = lines[-1].get_color()
        plt.plot(smooth_loss, color=color)

    plt.title('Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()


def plot_smoothed_loss(loss_history, window_size):
    """
    """
    # Plotting
    plt.figure(figsize=(10, 6))

    # Calculate the moving average
    smooth_loss = moving_average(loss_history, window_size)

    # Adjust the length of original loss values to match the smoothed array
    adjusted_loss_values = loss_history[:len(smooth_loss)]

    # Plot
    color = "tab:blue"
    plt.plot(adjusted_loss_values, alpha=0.5, color=color)
    plt.plot(smooth_loss, color=color)

    plt.title('Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid()
    plt.show()
