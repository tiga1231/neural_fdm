from functools import partial

import jax.numpy as jnp

from jax import jit
from jax import vmap
import jax.random as jrn
import jax.tree_util as jtu

import equinox as eqx

from tqdm import tqdm

from neural_fofin.models import AutoEncoderPiggy


def train_step_piggy(model, structure, optimizer, generator, opt_state, *, loss_fn, batch_size, key):
    """
    One step to train a piggy model.
    """
    # sample fresh data
    keys = jrn.split(key, batch_size)
    x = vmap(generator)(keys)

    # calculate updates for main
    val_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
    (loss, loss_vals), grads_main = val_grad_fn(
        model,
        structure,
        x,
        True,
        False
    )

    # calculate updates for piggy
    val_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
    (loss, loss_vals), grads_piggy = val_grad_fn(
        model,
        structure,
        x,
        True,
        True
    )

    # combine gradients
    grads = jtu.tree_map(lambda x, y: x + y, grads_main, grads_piggy)

    # apply updates
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss_vals, model, opt_state


def train_step(model, structure, optimizer, generator, opt_state, *, loss_fn, batch_size, key):
    """
    One step to train a model.
    """
    # sample fresh data
    keys = jrn.split(key, batch_size)
    x = vmap(generator)(keys)

    # calculate updates
    val_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
    (loss, loss_vals), grads = val_grad_fn(model, structure, x, aux_data=True)

    # apply updates
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    # NOTE: return latent space
    q = vmap(model.encode)(x)

    return loss_vals, model, opt_state, grads, q


def train_model(model, structure, optimizer, generator, *, loss_fn, num_steps, batch_size, key, callback=None):
    """
    Train a model over a number of steps.
    """
    # initial optimization step
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # assemble train step
    train_step_fn = train_step
    if isinstance(model, AutoEncoderPiggy):
        train_step_fn = train_step_piggy

    train_step_fn = partial(train_step_fn, loss_fn=loss_fn)
    train_step_fn = eqx.filter_jit(train_step_fn)

    # def get_cond_num(q):
    #     A = model.decoder.model.stiffness_matrix(q, structure)
    #     return jnp.linalg.cond(A)

    # cond_num_fn = jit(vmap(get_cond_num))

    # train
    loss_history = []
    for step in tqdm(range(num_steps)):

        # randomnesss
        key, _ = jrn.split(key)

        # train step
        loss_vals, model, opt_state, grads, qs = train_step_fn(
            model,
            structure,
            optimizer,
            generator,
            opt_state,
            batch_size=batch_size,
            key=key,
            )

        # NOTE: log gradient name
        grads_flat = jtu.tree_flatten(grads)[0]
        grads_flat = jtu.tree_map(lambda x: x.ravel(), grads_flat)
        grads_flat = jnp.concatenate(grads_flat)
        grad_norm = jnp.linalg.norm(jnp.array(grads_flat), axis=-1)
        loss_vals.append(grad_norm)

        # log latent space norm
        loss_vals.append(jnp.ravel(qs))

        # log latent space condition number
        # cond_nums = cond_num_fn(qs)
        # loss_vals.append(cond_nums[None, :])

        # store loss values
        loss_history.append(loss_vals)

        # callback
        if callback:
            callback(model, opt_state, loss_vals, step)

    return model, loss_history
