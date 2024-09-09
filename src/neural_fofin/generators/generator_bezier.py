import jax.random as jrn
import jax.numpy as jnp

from neural_fofin.generators.bezier import BezierSurfaceAsymmetric
from neural_fofin.generators.bezier import BezierSurfaceSymmetric
from neural_fofin.generators.bezier import BezierSurfaceSymmetricDouble


# ===============================================================================
# Generators
# ===============================================================================

class BezierSurfacePointGenerator:
    """
    A generator that outputs point evaluated on a wiggled bezier surface.
    """
    def __init__(self, surface, u, v, minval, maxval):
        self._check_array_shapes(surface, minval, maxval)

        self.surface = surface
        self.u = u
        self.v = v
        self.minval = minval
        self.maxval = maxval

    def _check_array_shapes(self, surface, minval, maxval):
        """
        Verify that input shapes are consistent.
        """
        tile_shape = surface.grid.tile.shape
        minval_shape = minval.shape
        maxval_shape = maxval.shape

        assert minval_shape == tile_shape, f"{minval_shape} vs. {tile_shape}"
        assert maxval_shape == tile_shape, f"{maxval_shape} vs. {tile_shape}"

    def wiggle(self, key):
        """
        Sample a translation vector from a uniform distribution.
        """
        shape = self.surface.grid.tile.shape
        return jrn.uniform(key, shape=shape, minval=self.minval, maxval=self.maxval)

    def evaluate_points(self, transform):
        """
        Generate transformed points.
        """
        points = self.surface.evaluate_points(self.u, self.v, transform)

        return jnp.ravel(points)

    def __call__(self, key, wiggle=True):
        """
        Generate (wiggled) points.
        """
        if wiggle:
            transform = self.wiggle(key)

        return self.evaluate_points(transform)


class BezierSurfaceSymmetricDoublePointGenerator(BezierSurfacePointGenerator):
    """
    A generator that outputs point evaluated on a wiggled, doubly-symmetric bezier surface.
    """
    def __init__(self, size, num_pts, u, v, minval, maxval, *args, **kwargs):
        surface = BezierSurfaceSymmetricDouble(size, num_pts)
        super().__init__(surface, u, v, minval, maxval)


class BezierSurfaceSymmetricPointGenerator(BezierSurfacePointGenerator):
    """
    A generator that outputs point evaluated on a wiggled, symmetric bezier surface.
    """
    def __init__(self, size, num_pts, u, v, minval, maxval, *args, **kwargs):
        surface = BezierSurfaceSymmetric(size, num_pts)
        super().__init__(surface, u, v, minval, maxval)


class BezierSurfaceAsymmetricPointGenerator(BezierSurfacePointGenerator):
    """
    A generator that outputs point evaluated on a wiggled bezier surface.
    """
    def __init__(self, size, num_pts, u, v, minval, maxval, *args, **kwargs):
        surface = BezierSurfaceAsymmetric(size, num_pts)
        super().__init__(surface, u, v, minval, maxval)


class BezierSurfaceBlendPointGenerator(BezierSurfacePointGenerator):
    """
    A generator that outputs points interpolated between two wiggled bezier surfaces.
    One surface is doubly-symmetric and the other completely asymmetric.
    """
    def __init__(self, size, num_pts, u, v, minval, maxval, alpha=0.5):
        minval_a, minval_b = minval
        maxval_a, maxval_b = maxval

        self.generator_a = BezierSurfaceSymmetricDoublePointGenerator(size, num_pts, u, v, minval_a, maxval_a)
        self.generator_b = BezierSurfaceAsymmetricPointGenerator(size, num_pts, u, v, minval_b, maxval_b)

        self.alpha = alpha

        surface = BezierSurfaceAsymmetric(size, num_pts)
        super().__init__(surface, u, v, minval_a, maxval_b)

    def __call__(self, key, wiggle=True):
        """
        Generate (wiggled) points.
        """
        points_a = self.generator_a(key, wiggle)
        points_b = self.generator_b(key, wiggle)

        return self.alpha * points_a + (1.0 - self.alpha) * points_b
