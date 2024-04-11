import equinox as eqx

import jax
from jax import vmap

import jax.random as jrn
import jax.numpy as jnp

from tqdm import tqdm


def select_x_free(_x_hat, structure):
    """
    Select the free XYZ coordinates of a structure.
    """
    _x_hat = jnp.reshape(_x_hat, (-1, 3))
    indices = structure.indices_free

    _x_hat = _x_hat[indices, :]

    return jnp.ravel(_x_hat)


def compute_loss_autoencoder(model, structure, x, has_aux=True):
    """
    Compute loss of autoencoder model.
    """
    x_hat, q = jax.vmap(model, in_axes=(0, None, None))(x, structure, has_aux)
    error = jnp.abs(x - x_hat)
    batch_error = jnp.sum(error, axis=-1)
    mean_batch_error = jnp.mean(batch_error, axis=-1)

    if has_aux:
        return mean_batch_error, (x_hat, q)

    return mean_batch_error


def compute_loss_piggybacker(model, structure, fd_data):
    """
    Compute loss of a piggybacker model.
    """
    # unpack
    x_hat, q = fd_data

    # pick free xyz from encoder predictions
    # x_hat_free = vmap(select_x_free, in_axes=(0, None))(x_hat, structure)

    # predict
    y_hat = jax.vmap(model, in_axes=(0, 0, None))(q, x_hat, structure)
    # y_hat = jax.vmap(model)(q)

    # error = jnp.abs(x_hat_free - y_hat)
    error = jnp.abs(x_hat - y_hat)
    batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


@eqx.filter_jit
def train_step(models, structure, optimizers, generator, opt_states, *, batch_size, key):
    """
    One step to train a model.
    """
    # unpack data
    model, piggybacker = models
    optimizer, optimizer_piggyback = optimizers
    opt_state, opt_state_piggyback = opt_states

    # sample fresh data
    keys = jrn.split(key, batch_size)
    x = vmap(generator)(keys)

    # calculate updates
    (loss, fd_data), grad = eqx.filter_value_and_grad(compute_loss_autoencoder, has_aux=True)(model, structure, x)
    loss_piggyback, grad_piggyback = eqx.filter_value_and_grad(compute_loss_piggybacker)(piggybacker, structure, fd_data)

    # apply model updates
    updates, opt_state = optimizer.update(grad, opt_state)
    model = eqx.apply_updates(model, updates)

    updates_piggyback, opt_state_piggyback = optimizer_piggyback.update(grad_piggyback, opt_state_piggyback)
    piggybacker = eqx.apply_updates(piggybacker, updates_piggyback)

    # pack data
    models = (model, piggybacker)
    opt_states = (opt_state, opt_state_piggyback)
    losses = (loss, loss_piggyback)

    return losses, models, opt_states


def train_models(models, structure, optimizers, generator, *, num_steps, batch_size, key, callback=None):
    """
    Train a pair of models over a number of steps.
    """
    # Sanity check
    assert len(optimizers) == len(models)

    # Initial optimization step
    opt_states = [optimizer.init(eqx.filter(model, eqx.is_array)) for model, optimizer in zip(models, optimizers)]

    loss_history = []
    for _ in tqdm(range(num_steps)):

        # randomnesss
        key, _ = jrn.split(key)

        # train step
        loss_vals, models, opt_states = train_step(
            models,
            structure,
            optimizers,
            generator,
            opt_states,
            key=key,
            batch_size=batch_size
            )

        # store loss
        loss_history.append(loss_vals)

        # callback
        if callback:
            callback(models, opt_states, loss_vals)

    return models, opt_states, loss_history
