import equinox as eqx

import jax
from jax import vmap

import jax.random as jrn
import jax.numpy as jnp

from tqdm import tqdm


def compute_loss(model, structure, x):
    """
    Compute the model loss.
    """
    x_hat = jax.vmap(model, in_axes=(0, None))(x, structure)
    error = jnp.abs(x - x_hat)
    batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


@eqx.filter_jit
def train_step(model, structure, optimizer, generator, opt_state, *, batch_size, key):
    """
    One step to train a model.
    """
    # sample fresh data
    keys = jrn.split(key, batch_size)
    x = vmap(generator)(keys)

    # calculate updates
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, structure, x)

    # apply updates
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss, model, opt_state


def train_model(model, structure, optimizer, generator, *, num_steps, batch_size, key, callback=None):
    """
    Train a model over a number of steps.
    """
    # Initial optimization step
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    loss_vals = []
    for _ in tqdm(range(num_steps)):

        # randomnesss
        key, _ = jrn.split(key)

        # train step
        loss_val, model, opt_state = train_step(
            model,
            structure,
            optimizer,
            generator,
            opt_state,
            key=key,
            batch_size=batch_size
            )

        # store loss
        loss_vals.append(loss_val)

        # callback
        if callback:
            callback(model, opt_state, loss_val)

    return model, opt_state, loss_vals
