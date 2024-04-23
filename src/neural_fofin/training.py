import equinox as eqx

import jax
from jax import vmap

import jax.random as jrn
import jax.numpy as jnp

from tqdm import tqdm


def compute_loss_shape(x, x_hat):
    """
    Calculate the shape reconstruction loss
    """
    error = jnp.abs(x - x_hat)
    batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


def compute_loss_residual(model, x, x_hat, q_hat, structure):
    """
    Calculate the residual loss.
    """
    connectivity = structure.connectivity
    indices = structure.indices_free

    def calculate_residuals(_x, _x_hat, _q):
        """
        """
        _q = _q * model.decoder.mask_edges + model.decoder.qmin
        loads = model.decoder.get_loads(_x, structure)
        _x_hat = jnp.reshape(_x_hat, (-1, 3))
        vectors = connectivity @ _x_hat

        residual_vectors = loads - connectivity.T @ (_q[:, None] * vectors)
        residual_vectors_free = residual_vectors[indices, :]

        return jnp.sum(jnp.square(residual_vectors_free), axis=-1)
        # return jnp.linalg.norm(residual_vectors_free, axis=-1)

    error = vmap(calculate_residuals)(x, x_hat, q_hat)
    batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


def compute_loss_2(model, structure, x, loss_params):
    """
    Compute the model loss.
    """
    x_hat, q_hat = jax.vmap(model, in_axes=(0, None, None))(x, structure, True)

    factor_shape = 1.0 / loss_params["shape"]["factor"]
    loss_shape = factor_shape * compute_loss_shape(x, x_hat)

    factor_residual = 1.0 / loss_params["residual"]["factor"]
    loss_residual = factor_residual * compute_loss_residual(model, x, x_hat, q_hat, structure)

    loss = 0.0
    if loss_params["shape"]["include"]:
        loss = loss + loss_shape
    if loss_params["residual"]["include"]:
        loss = loss + loss_residual

    aux_data = (
        loss + loss_shape + loss_residual,
        loss_shape,
        loss_residual
    )

    return loss, aux_data


def compute_loss(model, structure, x):
    """
    Compute the model loss.
    """
    x_hat = jax.vmap(model, in_axes=(0, None))(x, structure)

    error = jnp.abs(x - x_hat)
    batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


@eqx.filter_jit
def train_step(model, structure, optimizer, generator, opt_state, *, loss_params, batch_size, key):
    """
    One step to train a model.
    """
    # sample fresh data
    keys = jrn.split(key, batch_size)
    x = vmap(generator)(keys)

    # calculate updates
    (loss, loss_vals), grads = eqx.filter_value_and_grad(compute_loss_2, has_aux=True)(model, structure, x, loss_params)

    # apply updates
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss_vals, model, opt_state


def train_model(model, structure, optimizer, generator, *, loss_params, num_steps, batch_size, key, callback=None):
    """
    Train a model over a number of steps.
    """
    # Initial optimization step
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    loss_history = []
    for _ in tqdm(range(num_steps)):

        # randomnesss
        key, _ = jrn.split(key)

        # train step
        loss_vals, model, opt_state = train_step(
            model,
            structure,
            optimizer,
            generator,
            opt_state,
            loss_params=loss_params,
            batch_size=batch_size,
            key=key,
            )

        # store loss
        loss_history.append(loss_vals)

        # callback
        if callback:
            callback(model, opt_state, loss_vals)

    return model, opt_state, loss_history
