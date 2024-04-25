from jax import vmap

import jax.random as jrn

import equinox as eqx

from tqdm import tqdm

from neural_fofin.losses import compute_loss

import jax.tree_util as jtu


@eqx.filter_jit
def train_step_piggy(model, structure, optimizer, generator, opt_state, *, loss_params, batch_size, key):
    """
    One step to train a piggy model.
    """
    # sample fresh data
    keys = jrn.split(key, batch_size)
    x = vmap(generator)(keys)

    # model split
    # filter_spec = jtu.tree_map(lambda _: False, model)
    # filter_spec = eqx.tree_at(
    #     lambda tree: tree.autoencoder,
    #     filter_spec,
    #     replace=True
    # )

    # model_main, model_piggy = eqx.partition(model, filter_spec)
    model_main = model.main
    model_piggy = model.piggy

    # calculate updates for main model
    val_grad_fn = eqx.filter_value_and_grad(compute_loss, has_aux=True)
    (loss, loss_vals), grads_main = val_grad_fn(model_main, structure, x, loss_params, True)

    # calculate updates for piggybacker
    val_grad_fn = eqx.filter_value_and_grad(compute_loss_piggy, has_aux=True)
    (loss_piggy, loss_vals_piggy), grads_piggy = val_grad_fn(model_piggy, structure, x, loss_params, True)

    # apply updates
    # grads = eqx.combine(grads_main, grads_piggy)
    grads = type(model)(grads_main, grads_piggy)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss_vals, model, opt_state


@eqx.filter_jit
def train_step(model, structure, optimizer, generator, opt_state, *, loss_params, batch_size, key):
    """
    One step to train a model.
    """
    # sample fresh data
    keys = jrn.split(key, batch_size)
    x = vmap(generator)(keys)

    # calculate updates
    val_grad_fn = eqx.filter_value_and_grad(compute_loss, has_aux=True)
    (loss, loss_vals), grads = val_grad_fn(model, structure, x, loss_params, True)

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
