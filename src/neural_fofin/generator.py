import jax.random as jrn
import jax.numpy as jnp

from neural_fofin.bezier import evaluate_bezier_surface


# ===============================================================================
# Transformations
# ===============================================================================


def get_world_mirror_matrix(plane):
    """
    Mirror points across a given plane.

    :param points: A numpy array of shape (n, 3), where each row is a 3D point.
    :param plane: A string representing the plane ('xy', 'yz', 'xz').
    :return: Mirrored points as a numpy array of shape (n, 3).
    """
    if plane.lower() == 'xy':
        # Mirroring across the XY plane (change Z coordinate)
        mirror_matrix = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    elif plane.lower() == 'yz':
        # Mirroring across the YZ plane (change X coordinate)
        mirror_matrix = jnp.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif plane.lower() == 'xz':
        # Mirroring across the XZ plane (change Y coordinate)
        mirror_matrix = jnp.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        raise ValueError("Invalid plane. Choose 'xy', 'yz', or 'xz'.")

    return mirror_matrix


def mirror_points(points, mirror_matrix):
    """
    Mirror points across a given plane.

    :param points: A numpy array of shape (n, 3), where each row is a 3D point.
    :param mirror_matrix: A numpy array of shape (3, 3) representing the mirror transformation.
    :return: Mirrored points as a numpy array of shape (n, 3).
    """
    return points @ mirror_matrix


# ===============================================================================
# Grid functions
# ===============================================================================

def get_grid_tile_quarter(grid_size, grid_num_pts):
    """
    Get the 2D coordinates of a quarter tile.
    """
    half_grid_size = grid_size / 2.0
    grid_step = half_grid_size / (grid_num_pts - 1.0)

    pt0 = [grid_step, grid_step, 0.0]
    pt1 = [half_grid_size, grid_step, 0.0]
    pt2 = [grid_step, half_grid_size, 0.0]
    pt3 = [half_grid_size, half_grid_size, 0.0]

    return jnp.array([pt0, pt1, pt2, pt3])


def calculate_grid_from_tile_quarter(tile, indices, num_pts):
    """
    Compute a grid of control points from a unit tile and a wiggle vector.
    """
    grid_points = tile

    # 2. generate grid
    # mirror tile once
    mirrored_points = mirror_points(grid_points, get_world_mirror_matrix("yz"))
    grid_points = jnp.concatenate((grid_points, mirrored_points))

    # mirror tile again
    mirrored_points = mirror_points(grid_points, get_world_mirror_matrix("xz"))
    grid_points = jnp.concatenate((grid_points, mirrored_points))

    # reindex grid
    grid_points = reindex_grid(grid_points, indices)

    return jnp.reshape(grid_points, (num_pts, num_pts, 3))


def get_grid_tile_half(grid_size, grid_num_pts):
    """
    Get the 2D coordinates of a half tile.
    """
    tile_quarter = get_grid_tile_quarter(grid_size, grid_num_pts)

    raise NotImplementedError


def calculate_grid_from_tile_half(tile, indices, num_pts):
    """
    """
    raise NotImplementedError


def reindex_grid(grid, indices):
    """
    Reconfigure the grid using hard-coded indices (from Rhino).
    """
    return grid[indices, :]


# ===============================================================================
# Grids
# ===============================================================================

class PointGrid:
    """
    A grid of control points.
    """
    def __init__(self, tile, num_pts, indices) -> None:
        self.tile = tile
        self.num_pts = num_pts
        self.indices = indices

    def points(self, transform=None):
        tile = self.tile
        if transform is not None:
            tile = self.tile + transform
        return self._points(tile)

    def _points(self, tile):
        raise NotImplementedError


class PointGridSymmetricDouble(PointGrid):
    """
    A doubly-symmetric grid of control points.
    """
    def __init__(self, size, num_pts):
        # NOTE: indices are hard-coded from Rhino
        indices = [15, 13, 5, 7, 14, 12, 4, 6, 10, 8, 0, 2, 11, 9, 1, 3]
        tile = get_grid_tile_quarter(size, num_pts)

        super().__init__(tile, num_pts, indices)

    def _points(self, tile):
        return calculate_grid_from_tile_quarter(tile, self.indices, self.num_pts)


class PointGridSymmetric(PointGrid):
    """
    A symmetric grid of control points.
    """
    def __init__(self, size, num_pts):
        # NOTE: indices are hard-coded from Rhino
        indices = None
        super().__init__(size, num_pts, indices)

    def _points(self, tile):
        return calculate_grid_from_tile_half(tile, self.indices, self.num_pts)


# ===============================================================================
# Generators
# ===============================================================================

class BezierSurfacePointGenerator:
    """
    A generator that outputs point evaluated on a randomly wiggle bezier surface.
    """
    def __init__(self, grid, u, v, minval, maxval) -> None:
        self.grid = grid
        self.u = u
        self.v = v
        self.minval = minval
        self.maxval = maxval

    def wiggle(self, key):
        """
        Sample a translation vector from a uniform distribution.
        """
        shape = self.grid.tile.shape
        return jrn.uniform(key, shape=shape, minval=self.minval, maxval=self.maxval)

    def control_points(self, key):
        wiggle = self.wiggle(key)
        return self.grid.points(transform=wiggle)

    def bezier_points(self, key):
        control_pts = self.control_points(key)
        return evaluate_bezier_surface(control_pts, self.u, self.v)

    def __call__(self, key):
        return jnp.ravel(self.bezier_points(key))


# def get_wiggled_grid_from_tile(tile, wiggle, indices, num_pts):
#     """
#     Compute a grid of control points from a unit tile and a wiggle vector.
#     """
#     assert tile.shape == wiggle.shape, "Tile and wiggle must have the same shape"

#     # 1. apply wiggle on x, y, z to tile
#     tile = tile + wiggle

#     return calculate_grid_from_tile(tile, indices, num_pts)


# def get_bezier_surface_points_from_tile(tile, indices, num_pts, u, v):
#     """
#     """
#     # generate control points grid
#     control_points = calculate_grid_from_tile(tile, indices, num_pts)

#     # sample surface points on bezier
#     surface_points = evaluate_bezier_surface(control_points, u, v)

#     return jnp.reshape(surface_points, (-1, 3))
