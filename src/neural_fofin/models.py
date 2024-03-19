import jax.numpy as jnp

import equinox as eqx

from jaxtyping import Array, Float

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import nodes_load_from_faces


class ForceDensityModel(eqx.Module):
    """
    A force density model that calculates area loads based on the input shapes.
    """
    model: EquilibriumModel    
    loads: Array
    mask_edges: Array
    qmin: Float

    def __init__(self, model, loads, mask_edges, qmin=-1e-3):
        self.model = model
        self.loads = loads
        self.mask_edges = mask_edges
        self.qmin = qmin

    def __call__(self, q, x, structure):

        # NOTE: mask out fully fixed edges 
        q = q * self.mask_edges + self.qmin 
        xyz_fixed = self.get_xyz_fixed(x, structure)

        x_hat = self.model.equilibrium(q,
                                       xyz_fixed,
                                       self.loads,
                                       structure)
        
        return jnp.ravel(x_hat)
    
    def get_xyz_fixed(self, x, structure):
        
        indices = structure.indices_fixed
        x = jnp.reshape(x, (-1, 3))

        return x[indices, :]


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
        point_loads = self.get_point_loads(x, structure)

        x_hat = self.model.equilibrium(q,
                                       xyz_fixed,
                                       point_loads,
                                       structure)
                        
        return jnp.ravel(x_hat)


    def get_point_loads(self, x, structure):

        x = jnp.reshape(x, (-1, 3))
        point_loads = nodes_load_from_faces(x,
                                            self.loads,
                                            structure,
                                            is_local=False)
        
        return point_loads


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

    def __init__(self, mask_edges, qmin=-1e-3, *args, **kwargs):
    # def __init__(self, pred_scale, mask_edges, qmin=-1e-3, *args, **kwargs):
    # def __init__(self, *args, **kwargs):
        # self.pred_scale = pred_scale
        self.mask_edges = mask_edges
        self.qmin = qmin
        super().__init__(*args, **kwargs)
    
    # NOTE: x must be a flat vector
    def __call__(self, q):
        # NOTE: mask out fully fixed edges 
        q = q * self.mask_edges + self.qmin 

        x_hat = super().__call__(q)

        # return x_hat * self.pred_scale
        return x_hat
