import jax.numpy as jnp

import equinox as eqx

from jaxtyping import Array, Float

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import nodes_load_from_faces


QMIN = -1e-3


class ForceDensityModel(eqx.Module):
    """
    A force density model that calculates area loads based on the input shapes.
    """
    model: EquilibriumModel
    load: Float
    mask_edges: Array
    qmin: Float

    def __init__(self, model, load, mask_edges, qmin=QMIN):
        self.model = model
        self.load = load
        self.mask_edges = mask_edges
        self.qmin = qmin

    def __call__(self, q, x, structure):

        # NOTE: mask out fully fixed edges
        q = q * self.mask_edges + self.qmin
        xyz_fixed = self.get_xyz_fixed(x, structure)
        loads = self.get_loads(x, structure)

        # NOTE: use instead
        # self.model.nodes_free_positions(q, xyz_fixed, loads_nodes, structure)
        # to predict only free vertices
        x_hat = self.model.equilibrium(q,
                                       xyz_fixed,
                                       loads,
                                       structure)

        return jnp.ravel(x_hat)

    def get_xyz_fixed(self, x, structure):

        indices = structure.indices_fixed
        x = jnp.reshape(x, (-1, 3))

        return x[indices, :]

    def get_loads(self, x, structure):

        num_vertices = structure.num_vertices
        vertices_load_xy = jnp.zeros(shape=(num_vertices, 2))  # (num_vertices, xy)
        vertices_load_z = jnp.ones(shape=(num_vertices, 1)) * self.load   # (num_vertices, xy)

        return jnp.hstack((vertices_load_xy, vertices_load_z))


class ForceDensityWithShapeBasedLoads(ForceDensityModel):
    """
    A force density model that calculates area loads based on the input shapes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, q, x, structure):

        # NOTE: mask out fully fixed edges
        q = q * self.mask_edges + self.qmin
        xyz_fixed = self.get_xyz_fixed(x, structure)
        loads = self.get_loads(x, structure)

        # NOTE: use instead
        # self.model.nodes_free_positions(q, xyz_fixed, loads_nodes, structure)
        # to predict only free vertices
        x_hat = self.model.equilibrium(q,
                                       xyz_fixed,
                                       loads,
                                       structure)

        return jnp.ravel(x_hat)

    def get_loads(self, x, structure):

        x = jnp.reshape(x, (-1, 3))

        # need to convert loads into face loads
        num_faces = structure.num_faces
        faces_load_xy = jnp.zeros(shape=(num_faces, 2))  # (num_faces, xy)
        faces_load_z = jnp.ones(shape=(num_faces, 1)) * self.load  # (num_faces, xy)
        faces_load = jnp.hstack((faces_load_xy, faces_load_z))

        vertices_load = nodes_load_from_faces(x,
                                              faces_load,
                                              structure,
                                              is_local=False)

        return vertices_load


class AutoEncoder(eqx.Module):
    """
    An autoencoder that couples a neural network with a form-finding solver.
    """
    encoder: eqx.Module
    decoder: eqx.Module

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    # NOTE: x must be a flat vector
    def __call__(self, x, structure, return_q=False):
        q = self.encoder(x) * -1.0  # negative q denotes compression
        x_hat = self.decoder(q, x, structure)

        if return_q:
            return x_hat, q

        return x_hat


class PiggyDecoder(eqx.nn.MLP):
    """
    A MLP decoder that piggybacks on an autoencoder.
    """
    # pred_scale: Float
    mask_edges: Array
    qmin: Float

    def __init__(self, mask_edges, qmin=QMIN, *args, **kwargs):
        self.mask_edges = mask_edges
        self.qmin = qmin
        super().__init__(*args, **kwargs)

    # NOTE: x must be a flat vector
    def __call__(self, q, x, structure):
        # NOTE: mask out fully fixed edges
        q = q * self.mask_edges + self.qmin

        x_fixed = self.get_xyz_fixed(x, structure)
        x_free_hat = super().__call__(q)

        return self.get_xyz_hat(x_free_hat, x_fixed, structure)

    def get_xyz_fixed(self, x, structure):
        """
        Select the position of the free nodes.
        """
        x = jnp.reshape(x, (-1, 3))
        indices = structure.indices_fixed

        # return jnp.ravel(x[indices, :])
        return x[indices, :]

    def get_xyz_hat(self, x_free, x_fixed, structure):
        """
        Concatenate the position of the free and the fixed nodes.
        Return a vector, not a matrix.
        """
        indices = structure.indices_freefixed
        x_free = jnp.reshape(x_free, (-1, 3))
        x_hat = jnp.concatenate((x_free, x_fixed))[indices, :]

        return jnp.ravel(x_hat)
