from itertools import product

import jax.numpy as jnp

from jax_fdm.datastructures import FDMesh

from neural_fofin.bezier import evaluate_bezier_surface


def create_mesh_from_grid(grid, u, v):
    """
    Boundary-supported mesh on bezier surface.
    """
    # generate base FD Mesh
    srf_points = calculate_bezier_surface_points_from_grid(grid, u, v)

    num_u = u.shape[0]
    num_v = v.shape[0]
    faces = calculate_mesh_grid_faces(num_u - 1, num_v - 1)
    mesh = FDMesh.from_vertices_and_faces(srf_points, faces)

    # define structural system
    mesh.vertices_supports(mesh.vertices_on_boundary())

    return mesh


def create_mesh_from_bezier(bezier, u, v):
    """
    Boundary-supported mesh on bezier surface.
    """
    # generate base FD Mesh
    srf_points = bezier.evaluate_points(u, v)
    srf_points = jnp.reshape(srf_points, (-1, 3))

    num_u = u.shape[0]
    num_v = v.shape[0]
    faces = calculate_mesh_grid_faces(num_u - 1, num_v - 1)
    mesh = FDMesh.from_vertices_and_faces(srf_points, faces)

    # define structural system
    mesh.vertices_supports(mesh.vertices_on_boundary())

    return mesh


def calculate_mesh_grid_faces(nx, ny):
    """
    Generate the indices of the mesh faces of the grid.
    """
    faces = [
        [
            i * (ny + 1) + j,
            (i + 1) * (ny + 1) + j,
            (i + 1) * (ny + 1) + j + 1,
            i * (ny + 1) + j + 1,
        ]
        for i, j in product(range(nx), range(ny))
    ]
    return faces


def calculate_bezier_surface_points_from_grid(grid, u, v):
    """
    """
    # generate control points grid
    control_points = grid.points()

    # sample surface points on bezier
    surface_points = evaluate_bezier_surface(control_points, u, v)

    return jnp.reshape(surface_points, (-1, 3))
