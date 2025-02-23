import jax.numpy as jnp
from jax import vmap

from neural_fdm.helpers import vertices_residuals_from_xyz
from neural_fdm.models import AutoEncoderPiggy


# ===============================================================================
# Loss assemblers
# ===============================================================================

def compute_loss(
    model,
    structure,
    x,
    loss_fn,
    loss_params,
    aux_data=False,
    piggy_mode=False
):
    """
    Compute the model loss according to the model type.
    """
    predict_fn = vmap(model, in_axes=(0, None, None, None))
    x_hat, data_hat = predict_fn(x, structure, True, piggy_mode)

    # TODO: make _loss_fn an input, and make isinstance check before running this function
    _loss_fn = _compute_loss
    if isinstance(model, AutoEncoderPiggy):
        _loss_fn = _compute_loss_piggy

    return _loss_fn(
        loss_fn, loss_params, x, x_hat, data_hat, structure, aux_data, piggy_mode
    )


def _compute_loss(
    loss_fn,
    loss_params,
    x,
    x_hat,
    params_hat,
    structure,
    aux_data,
    piggy_mode=False
):
    """
    Compute the model loss of an autoencoder.
    """
    return loss_fn(x, x_hat, params_hat, structure, loss_params, aux_data)


def _compute_loss_piggy(
    loss_fn,
    loss_params,
    x,
    x_data_hat,
    y_data_hat,
    structure,
    aux_data,
    piggy_mode=True,
):
    """
    Compute the model loss of a piggy autoencoder.
    """
    x_hat, x_params_hat = x_data_hat

    if not piggy_mode:
        loss_data = loss_fn(x, x_hat, x_params_hat, structure, loss_params, aux_data)
    else:
        y_hat, y_params_hat = y_data_hat
        q, xyz_fixed, loads = y_params_hat

        loss_data = loss_fn(
            x_hat, y_hat, y_params_hat, structure, loss_params, aux_data
        )

    return loss_data


# ===============================================================================
# Task losses
# ===============================================================================

def compute_loss_shape_residual(
    x,
    x_hat,
    params_hat,
    structure,
    loss_params,
    aux_data,
    *args
):
    """
    Compute the model loss.

    Parameters
    ----------
    x : target
    x_hat : prediction
    params_hat : parameters prediction (aux data)
    structure : the connectivity graph of the structure
    loss_params : the scaling parameters to combine the loss' error terms
    aux_data : if true, returns auxiliary data
    """
    shape_params = loss_params["shape"]
    factor_shape = shape_params["weight"] / shape_params["scale"]
    loss_shape = compute_error_shape_l1(x, x_hat)
    loss_shape = factor_shape * loss_shape

    indices = structure.indices_free
    residual_params = loss_params["residual"]
    factor_residual = residual_params["weight"] / residual_params["scale"]
    loss_residual = compute_error_residual(x_hat, params_hat, structure, indices)
    loss_residual = factor_residual * loss_residual

    loss = 0.0
    if shape_params["include"]:
        loss = loss + loss_shape
    if residual_params["include"]:
        loss = loss + loss_residual

    loss_terms = {
        "loss": loss,
        "shape error": loss_shape,
        "residual error": loss_residual
    }

    if aux_data:
        return loss, loss_terms

    return loss


def compute_loss_shape_residual_smoothness(
        x,
        x_hat,
        params_hat,
        structure,
        loss_params,
        aux_data,
        *args
):
    """
    Compute the model loss.

    Parameters
    ----------
    x : target
    x_hat : prediction
    params_hat : parameters prediction (aux data)
    structure : the connectivity graph of the structure
    loss_params : the scaling parameters to combine the loss' error terms
    aux_data : if true, returns auxiliary data
    """
    # compression ring shape
    shape_params = loss_params["shape"]
    factor_shape = shape_params["weight"] / shape_params["scale"]
    shape_dims = shape_params["dims"]
    levels_compression = shape_params["levels_compression"]

    def slice_xyz_rings(_x, levels):
        return jnp.reshape(_x, shape_dims)[levels, :, :].ravel()

    slice_xyz_vmap = vmap(slice_xyz_rings, in_axes=(0, None))
    xyz_slice = slice_xyz_vmap(x, levels_compression)
    xyz_hat_slice = slice_xyz_vmap(x_hat, levels_compression)
    assert xyz_slice.shape == xyz_hat_slice.shape

    # NOTE: Using L2 norm here because L1 does not work well
    loss_shape = compute_error_shape_l2(xyz_slice, xyz_hat_slice)
    loss_shape = factor_shape * loss_shape

    # tension rings height
    height_params = loss_params["height"]
    factor_height = height_params["weight"] / height_params["scale"]
    height_dims = height_params["dims"]
    levels_tension = height_params["levels_tension"]

    def slice_z_rings(_x, levels):
        return jnp.reshape(_x, height_dims)[levels, :, 2].ravel()

    slice_z_vmap = vmap(slice_z_rings, in_axes=(0, None))
    z_slice = slice_z_vmap(x, levels_tension)
    z_hat_slice = slice_z_vmap(x_hat, levels_tension)
    assert z_slice.shape == z_hat_slice.shape

    # NOTE: Using L2 norm here because L1 does not work well
    loss_height = compute_error_shape_l2(z_slice, z_hat_slice)
    loss_height = factor_height * loss_height

    # residual
    indices = structure.indices_free
    residual_params = loss_params["residual"]
    factor_residual = residual_params["weight"] / residual_params["scale"]
    loss_residual = compute_error_residual(x_hat, params_hat, structure, indices)
    loss_residual = factor_residual * loss_residual

    # smoothness
    smooth_params = loss_params["energy"]
    factor_smooth = smooth_params["weight"] / smooth_params["scale"]
    loss_smooth = compute_error_smoothness(x_hat, params_hat, structure)
    loss_smooth = factor_smooth * loss_smooth

    # regularization
    regularization_params = loss_params["regularization"]
    factor_regularization = regularization_params["weight"]
    q = params_hat[0]
    regularization = compute_q_regularization(q)
    regularization = factor_regularization * regularization

    loss = 0.0
    if shape_params["include"]:
        loss = loss + loss_shape
    if height_params["include"]:
        loss = loss + loss_height
    if residual_params["include"]:
        loss = loss + loss_residual
    if smooth_params["include"]:
        loss = loss + loss_smooth
    if regularization_params["include"]:
        loss = loss + regularization

    loss_terms = {
        "loss": loss,
        "shape error": loss_shape,
        "height error": loss_height,
        "residual error": loss_residual,
        "smooth error": loss_smooth,
        "regularization": regularization
    }

    if aux_data:
        return loss, loss_terms

    return loss


def compute_loss_residual_smoothness(
        x,
        x_hat,
        params_hat,
        structure,
        loss_params,
        aux_data,
        *args
):
    """
    Compute the model loss.

    Parameters
    ----------
    x : target
    x_hat : prediction
    params_hat : parameters prediction (aux data)
    structure : the connectivity graph of the structure
    loss_params : the scaling parameters to combine the loss' error terms
    aux_data : if true, returns auxiliary data
    """
    # include support ring vertices to compute residual error on
    residual_params = loss_params["residual"]
    indices_rings = residual_params["indices"]
    indices = structure.indices_free
    indices = jnp.concatenate((indices, indices_rings))
    factor_residual = residual_params["weight"] / residual_params["scale"]
    loss_residual = compute_error_residual(x_hat, params_hat, structure, indices)
    loss_residual = factor_residual * loss_residual

    smooth_params = loss_params["energy"]
    factor_smooth = smooth_params["weight"] / smooth_params["scale"]
    loss_smooth = compute_error_smoothness(x_hat, params_hat, structure)
    loss_smooth = factor_smooth * loss_smooth

    # regularization
    regularization_params = loss_params["regularization"]
    factor_regularization = regularization_params["weight"]
    q = params_hat[0]
    regularization = compute_q_regularization(q)
    regularization = factor_regularization * regularization

    loss = 0.0
    if residual_params["include"]:
        loss = loss + loss_residual
    if smooth_params["include"]:
        loss = loss + loss_smooth
    if regularization_params["include"]:
        loss = loss + regularization

    loss_terms = {
        "loss": loss,
        "residual error": loss_residual,
        "smooth error": loss_smooth,
        "regularization": regularization
    }

    if aux_data:
        return loss, loss_terms

    return loss


# ===============================================================================
# Shape approximation error
# ===============================================================================

def compute_error_shape_l1(x, x_hat):
    """
    Calculate the shape reconstruction error
    """
    error = jnp.abs(x - x_hat)
    batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


def compute_error_shape_l2(x, x_hat):
    """
    Calculate the shape reconstruction error
    """
    error = jnp.square(x - x_hat)
    batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


# ===============================================================================
# Residual error
# ===============================================================================

def compute_error_residual(x_hat, params_hat, structure, indices):
    """
    Calculate the residual error.
    """
    def calculate_residuals(_x_hat, _params_hat):
        """
        _x_hat: the shape predicted by the model
        _data_hat: the aux data produced by the model

        NOTE: Not using jnp.linalg.norm because we hitted NaNs.
        """
        q_hat, xyz_fixed, loads = _params_hat
        residual_vectors = vertices_residuals_from_xyz(q_hat, loads, _x_hat, structure)
        residual_vectors_free = jnp.ravel(residual_vectors[indices, :])

        # return jnp.linalg.norm(residual_vectors_free, axis=-1)
        # return jnp.sqrt(jnp.sum(jnp.square(residual_vectors_free), axis=-1))
        return jnp.square(residual_vectors_free)

    residuals = vmap(calculate_residuals)(x_hat, params_hat)
    shape_residuals = jnp.sqrt(jnp.sum(residuals, axis=-1))
    batch_residual = jnp.mean(shape_residuals, axis=-1)

    return batch_residual



# ===============================================================================
# Smoothness energy
# ===============================================================================

def compute_error_smoothness(x_hat, params_hat, structure):
    """
    Calculate the shape smoothness (fairness) error.
    """
    error = vmap(vertices_smoothness, in_axes=(0, None))(x_hat, structure)
    batch_error = jnp.sqrt(jnp.sum(error, axis=-1))
    # batch_error = jnp.sum(error, axis=-1)

    return jnp.mean(batch_error, axis=-1)


def compute_qhat_regularization(q_hat):
    """
    Calculate the q_hat low variance
    probably rename it as params hat
    q_hat, _ , _ = params_hat
    """
    sign_q = jnp.sign(q_hat)
    var_q_pos = jnp.var(q_hat, where=sign_q > 0)
    var_q_neg = jnp.var(q_hat, where=sign_q < 0)

    return jnp.mean(var_q_pos) + jnp.mean(var_q_neg)


def vertices_smoothness(xyz, structure):
    """
    Compute the shape smoothness energy of the vertices of a structure.
    """
    xyz = jnp.reshape(xyz, (-1, 3))

    def vertex_fairness(xyz_vertex, adjacency_vertex):
        """
        Compute the fairness of an n-gon vertex neighborhood.
        """
        xyz_nbrs = adjacency_vertex @ xyz / jnp.sum(adjacency_vertex, axis=-1)

        fvector = xyz_vertex - xyz_nbrs
        assert fvector.shape == xyz_vertex.shape

        return jnp.sum(jnp.square(fvector))

    indices = structure.indices_free
    adjacency_free = structure.adjacency[indices, :]
    xyz_free = xyz[indices, :]

    vertices_fairness_fn = vmap(vertex_fairness)

    return vertices_fairness_fn(xyz_free, adjacency_free)


# ===============================================================================
# Smoothness energy
# ===============================================================================

def compute_q_regularization(q):
    """
    Calculate a regularization term such that q has low variance.
    """
    sign_q = jnp.sign(q)
    var_q_pos = jnp.var(q, where=sign_q > 0)
    var_q_neg = jnp.var(q, where=sign_q < 0)

    # NOTE: jnp.mean is doing nothing here because the size of the variance arrays is 1
    result = jnp.mean(var_q_pos) + jnp.mean(var_q_neg)

    return result


# ===============================================================================
# Utilities
# ===============================================================================

def print_loss_summary(loss_terms, prefix=None):
    """
    """
    msg_parts = []
    if prefix:
        msg_parts.append(prefix)

    for label, term in loss_terms.items():
        part = f"{label.capitalize()}: {term.item():.4f}"
        msg_parts.append(part)

    msg = "\t".join(msg_parts)
    print(msg)
