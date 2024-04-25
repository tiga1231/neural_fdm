from jax import vmap
import jax.numpy as jnp

from neural_fofin.models import AutoEncoderPiggy

from neural_fofin.helpers import vertices_residuals_from_xyz


def compute_loss(model, structure, x, loss_params, aux_data=False):
    """
    Compute the model loss according to the model type.
    """
    x_hat, data_hat = vmap(model, in_axes=(0, None, None))(x, structure, True)

    loss_fn = _compute_loss
    if isinstance(model, AutoEncoderPiggy):
        loss_fn = _compute_loss_piggy

    return loss_fn(x, x_hat, data_hat, structure, loss_params, aux_data)


def _compute_loss_piggy(x, x_hat, y_hat, structure, loss_params, aux_data):
    """
    Compute the model loss of a piggy autoencoder.
    """
    x_hat, x_params_hat = x_hat
    loss_fd = _compute_loss(
        x,
        x_hat,
        x_params_hat,
        structure,
        loss_params,
        aux_data
    )

    y_hat, y_params_hat = y_hat
    loss_piggy = _compute_loss(
        x_hat,
        y_hat,
        y_params_hat,
        structure,
        loss_params,
        aux_data
        )

    if aux_data:
        loss_fd, loss_fd_terms = loss_fd
        loss_piggy, loss_piggy_terms = loss_piggy

        return loss_fd, loss_fd_terms

    # return loss_fd + loss_piggy
    return loss_fd


def _compute_loss(x, x_hat, params_hat, structure, loss_params, aux_data):
    """
    Compute the model loss.
    """
    shape_params = loss_params["shape"]
    factor_shape = shape_params["weight"] / shape_params["scale"]
    loss_shape = compute_error_shape(x, x_hat)
    loss_shape = factor_shape * loss_shape

    residual_params = loss_params["residual"]
    factor_residual = residual_params["weight"] / residual_params["scale"]
    loss_residual = compute_error_residual(x_hat, params_hat, structure)
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


def compute_error_shape(x, x_hat):
    """
    Calculate the shape reconstruction error
    """
    error = jnp.abs(x - x_hat)
    batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


def compute_error_residual(x_hat, params_hat, structure):
    """
    Calculate the residual error.
    """
    indices_free = structure.indices_free

    def calculate_residuals(_x_hat, _params_hat):
        """
        _x_hat: the shape predicted by the model
        _data_hat: the aux data produced by the model

        NOTE: Not using jnp.linalg.norm because we hitted NaNs.
        """
        q_hat, xyz_fixed, loads = _params_hat
        residual_vectors = vertices_residuals_from_xyz(q_hat, loads, _x_hat, structure)
        residual_vectors_free = jnp.ravel(residual_vectors[indices_free, :])

        # return jnp.linalg.norm(residual_vectors_free, axis=-1)
        # return jnp.sqrt(jnp.sum(jnp.square(residual_vectors_free), axis=-1))
        # return jnp.square(residual_vectors_free)
        return jnp.square(residual_vectors_free)

    error = vmap(calculate_residuals)(x_hat, params_hat)
    # batch_error = jnp.sum(error, axis=-1)
    batch_error = jnp.sqrt(jnp.sum(error, axis=-1))

    return jnp.mean(batch_error, axis=-1)
