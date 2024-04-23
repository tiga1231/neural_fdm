import jax.numpy as jnp

import equinox as eqx

from jaxtyping import Array, Float

from jax_fdm.equilibrium import EquilibriumModel as FDModel

from neural_fofin.helpers import calculate_area_loads
from neural_fofin.helpers import calculate_equilibrium_state
from neural_fofin.helpers import calculate_fd_params_state


# ===============================================================================
# Autoencoders
# ===============================================================================

class AutoEncoder(eqx.Module):
    """
    An autoencoder that couples a neural network with a form-finding solver.
    """
    encoder: eqx.Module
    decoder: eqx.Module

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, x, structure, aux_data=False):
        # NOTE: x must be a flat vector
        q = self.encoder(x)
        pred = self.decoder(q, x, structure, aux_data)

        if aux_data:
            x_hat, data = pred
            return x_hat, data

        x_hat = pred

        return x_hat

    def predict_states(self, x, structure):
        """
        To interface with JAX FDM visualization.
        """
        # Predict shape
        xyz_hat, data = self(x, structure, True)

        # Unpack aux data
        q, xyz_fixed, loads = data

        # Equilibrium parameters
        fd_params_state = calculate_fd_params_state(
            q,
            xyz_fixed,
            loads
        )

        # Equilibrium state
        xyz_hat = jnp.reshape(xyz_hat, (-1, 3))

        eq_state = calculate_equilibrium_state(
            q,
            xyz_hat,  # xyz_free | xyz_fixed
            loads,
            structure
        )

        return eq_state, fd_params_state


# ===============================================================================
# Encoders
# ===============================================================================

class Encoder(eqx.Module):
    """
    An encoder.
    """
    pass


class MLPEncoder(eqx.nn.MLP, Encoder):
    """
    A MLP encoder.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x):
        """
        Transform a shape from Euclidean to latent representation.
        """
        # MLP prediction (must be positive due to softplus activation)
        q_hat = super().__call__(x)

        # NOTE: negative q denotes compression, positive tension.
        return q_hat * -1.0


# ===============================================================================
# Decoders
# ===============================================================================

class Decoder(eqx.Module):
    """
    A decoder.
    """
    load: Float
    mask_edges: Array
    qmin: Float

    def __init__(self, load, mask_edges, qmin=-1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.load = load
        self.mask_edges = mask_edges
        self.qmin = qmin

    def get_q(self, q_hat):
        """
        TODO: A better model should not be predicting and then masking edges.
        """
        return q_hat * self.mask_edges + self.qmin

    @staticmethod
    def get_xyz_fixed(x, structure):
        """
        """
        indices = structure.indices_fixed
        x = jnp.reshape(x, (-1, 3))

        return x[indices, :]

    def __call__(self, q, x, structure, aux_data=False):
        """
        """
        # gather parameters
        q = self.get_q(q)
        xyz_fixed = self.get_xyz_fixed(x, structure)
        loads = self.get_loads(x, structure)

        # predict x
        x_hat = self.get_xyz((q, xyz_fixed, loads), structure)

        if aux_data:
            data = (q, xyz_fixed, loads)
            return x_hat, data

        return x_hat

    def get_loads(self, x, structure):
        """
        """
        raise NotImplementedError

    def get_xyz(self, params, structure):
        """
        """
        raise NotImplementedError


# ===============================================================================
# Physics-based decoders
# ===============================================================================

class ForceDensityDecoder(Decoder):
    """
    A force density model that calculates area loads based on the input shapes.
    """
    model: FDModel

    def __init__(self, model, *args, **kwargs):
        self.model = model
        super().__init__(*args, **kwargs)

    def get_xyz(self, params, structure):
        """
        """
        q, xyz_fixed, loads = params
        # NOTE: to predict only free vertices, use instead
        # self.model.nodes_free_positions(q, xyz_fixed, loads_nodes, structure)
        x_hat = self.model.equilibrium(q,
                                       xyz_fixed,
                                       loads,
                                       structure)

        return jnp.ravel(x_hat)

    def get_loads(self, x, structure):
        """
        Calculate applied vertex loads.
        """
        num_vertices = structure.num_vertices
        vertices_load_xy = jnp.zeros(shape=(num_vertices, 2))  # (num_vertices, xy)
        vertices_load_z = jnp.ones(shape=(num_vertices, 1)) * self.load   # (num_vertices, xy)

        return jnp.hstack((vertices_load_xy, vertices_load_z))


class ForceDensityDecoderWithAreaLoads(ForceDensityDecoder):
    """
    A force density model that calculates area loads based on the input shapes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_loads(self, x, structure):
        """
        Calculate vertex loads.
        """
        return calculate_area_loads(x, structure, self.load)


# ===============================================================================
# Neural decoders
# ===============================================================================

class MLPDecoder(Decoder, eqx.nn.MLP):
    """
    A MLP decoder that piggybacks on an autoencoder.
    NOTE: Should the inheritance order be reversed?
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_xyz(self, params, structure):
        """
        """
        q, x_fixed, lodas = params
        # x_free = super().__call__(q)
        # NOTE: using this exotic way to call __call__ to map q to x
        # due to multiple inheritance
        x_free = eqx.nn.MLP.__call__(self, q)

        # Concatenate the position of the free and the fixed nodes
        indices = structure.indices_freefixed
        x_free = jnp.reshape(x_free, (-1, 3))
        x_hat = jnp.concatenate((x_free, x_fixed))[indices, :]

        return jnp.ravel(x_hat)

    def get_loads(self, x, structure):
        """
        Calculate vertex loads.
        """
        return calculate_area_loads(x, structure, self.load)
