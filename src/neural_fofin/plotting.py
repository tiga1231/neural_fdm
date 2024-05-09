from math import fabs

from statistics import mean
from statistics import stdev

import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt


def moving_average(data, window_size):
    """
    """
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def plot_losses(loss_history, labels):
    """
    """
    # Plotting
    plt.figure(figsize=(10, 6))

    for label in labels:
        loss_values = [float(vals[label]) for vals in loss_history]
        plt.plot(loss_values, label=label)

    plt.title('Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()


def plot_smoothed_losses(loss_history, window_size, labels):
    """
    """
    # Plotting
    plt.figure(figsize=(10, 6))

    for label in labels:

        # Loss values
        loss_values = [float(vals[label]) for vals in loss_history]

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


def plot_stiffness_condition_num(loss_values):
    """
    Plot the condition number of the stiffness matrix.
    """
    cond_nums = jnp.concatenate([values[-1] for values in loss_values])
    mean_cond_nums = jnp.mean(cond_nums, axis=-1)
    std_cond_nums = jnp.std(cond_nums, axis=-1)
    xs = np.arange(len(loss_values))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(mean_cond_nums)
    plt.fill_between(xs, mean_cond_nums - std_cond_nums, mean_cond_nums + std_cond_nums, alpha=0.5)

    plt.title('Matrix condition number')
    plt.xlabel('Step')
    plt.ylabel('Number')
    plt.yscale('log')
    plt.grid()
    plt.show()


def plot_latent_mean_std(loss_values):
    """
    Plot the norm of the absolute values in the force density vector.
    """
    qs_mean = []
    qs_std = []
    for i, values in enumerate(loss_values):
        qs = [fabs(q) for q in values[-1].tolist()]
        mean_q = mean(qs)
        std_q = stdev(qs)
        qs_mean.append(mean_q)
        qs_std.append(std_q)

    qs_mean = np.array(qs_mean)
    qs_std = np.array(qs_std)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(qs_mean)
    xs = np.arange(len(loss_values))
    plt.fill_between(xs, qs_mean - qs_std, qs_mean + qs_std, alpha=0.3)

    plt.title('Force density')
    plt.xlabel('Step')
    plt.ylabel('Mean value')
    plt.yscale('log')
    plt.grid()
    plt.show()


def plot_latent_norm(loss_values):
    """
    Plot the norm of the latent space vector.
    """
    latent_norm = [values[-1] for values in loss_values]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(latent_norm)

    plt.title('Latent norm')
    plt.xlabel('Step')
    plt.ylabel('Norm')
    plt.yscale('log')
    plt.grid()
    plt.show()


def plot_gradient_norm(loss_values):
    """
    Plot the norm of the gradient vector
    """
    gradient_norm = [values[-2] for values in loss_values]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(gradient_norm)

    plt.title('Gradient norm')
    plt.xlabel('Step')
    plt.ylabel('Norm')
    plt.yscale('log')
    plt.grid()
    plt.show()
