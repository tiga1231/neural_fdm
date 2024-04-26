from jax import vmap
import jax.random as jrn
import jax.tree_util as jtu

import equinox as eqx

from tqdm import tqdm

from neural_fofin.losses import compute_loss
from neural_fofin.models import AutoEncoderPiggy


def train_step_piggy(model, structure, optimizer, generator, opt_state, *, loss_params, batch_size, key):
    """
    One step to train a piggy model.
    """
    # sample fresh data
    keys = jrn.split(key, batch_size)
    x = vmap(generator)(keys)

    # calculate updates for main
    val_grad_fn = eqx.filter_value_and_grad(compute_loss, has_aux=True)
    (loss, loss_vals), grads_main = val_grad_fn(
        model,
        structure,
        x,
        loss_params,
        True,
        False
    )

    # calculate updates for piggy
    val_grad_fn = eqx.filter_value_and_grad(compute_loss, has_aux=True)
    (loss, loss_vals), grads_piggy = val_grad_fn(
        model,
        structure,
        x,
        loss_params,
        True,
        True
    )

    # combine gradients
    grads = jtu.tree_map(lambda x, y: x + y, grads_main, grads_piggy)

    # apply updates
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss_vals, model, opt_state


def train_step(model, structure, optimizer, generator, opt_state, *, loss_params, batch_size, key):
    """
    One step to train a model.
    """
    # sample fresh data
    keys = jrn.split(key, batch_size)
    x = vmap(generator)(keys)

    # calculate updates
    val_grad_fn = eqx.filter_value_and_grad(compute_loss, has_aux=True)
    (loss, loss_vals), grads = val_grad_fn(model, structure, x, loss_params, aux_data=True)

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

    train_step_fn = train_step
    if isinstance(model, AutoEncoderPiggy):
        train_step_fn = train_step_piggy

    train_step_fn = eqx.filter_jit(train_step_fn)

    loss_history = []
    for _ in tqdm(range(num_steps)):

        # randomnesss
        key, _ = jrn.split(key)

        # train step
        loss_vals, model, opt_state = train_step_fn(
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
