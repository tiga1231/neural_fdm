from jax import vmap
import jax.numpy as jnp

from neural_fofin.helpers import vertices_residuals_from_xyz


def compute_loss_shape(x, x_hat):
    """
    Calculate the shape reconstruction loss
    """
    error = jnp.abs(x - x_hat)
    batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


def compute_loss_residual(model, x_hat, data_hat, structure):
    """
    Calculate the residual loss.
    """
    indices_free = structure.indices_free

    def calculate_residuals(_x_hat, _data_hat):
        """
        _x_hat: the shape predicted by the model
        _data_hat: the aux data produced by the model
        """
        q_hat, xyz_fixed, loads = _data_hat
        xyz = jnp.reshape(_x_hat, (-1, 3))
        residual_vectors = vertices_residuals_from_xyz(q_hat, loads, xyz, structure)
        residual_vectors_free = residual_vectors[indices_free, :]

        # return jnp.linalg.norm(residual_vectors_free, axis=-1)
        return jnp.sum(jnp.square(residual_vectors_free), axis=-1)

    error = vmap(calculate_residuals)(x_hat, data_hat)
    batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


def compute_loss(model, structure, x, loss_params, aux_data=False):
    """
    Compute the model loss.
    """
    x_hat, data_hat = vmap(model, in_axes=(0, None, None))(x, structure, True)

    shape_params = loss_params["shape"]
    factor_shape = shape_params["weight"] / shape_params["scale"]
    loss_shape = compute_loss_shape(x, x_hat)
    loss_shape = factor_shape * loss_shape

    residual_params = loss_params["residual"]
    factor_residual = residual_params["weight"] / residual_params["scale"]
    loss_residual = compute_loss_residual(model, x_hat, data_hat, structure)
    loss_residual = factor_residual * loss_residual

    loss = 0.0
    if shape_params["include"]:
        loss = loss + loss_shape
    if residual_params["include"]:
        loss = loss + loss_residual

    loss_terms = (
        loss,
        loss_shape,
        loss_residual
    )

    if aux_data:
        return loss, loss_terms

    return loss
