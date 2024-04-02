from neural_fofin.generator import get_bezier_surface_points_from_grid
from neural_fofin.generator import get_bezier_surface_points_from_tile
from neural_fofin.generator import calculate_grid_faces

from jax_fdm.datastructures import FDMesh


def create_mesh_from_grid_simple(grid, u, v):
    """
    Boundary-supported mesh.
    """
    # generate base FD Mesh
    srf_points = get_bezier_surface_points_from_grid(grid, u, v)

    num_u = u.shape[0]
    num_v = v.shape[0]
    faces = calculate_grid_faces(num_u - 1, num_v - 1)
    mesh = FDMesh.from_vertices_and_faces(srf_points, faces)

    # define structural system
    mesh.vertices_supports(mesh.vertices_on_boundary())

    return mesh


def create_mesh_from_grid(grid, u, v, load, q=-1.0):
    """
    Boundary-supported mesh, with loads and force densities.
    """
    # generate base FD Mesh
    srf_points = get_bezier_surface_points_from_grid(grid, u, v)

    num_u = u.shape[0]
    num_v = v.shape[0]
    faces = calculate_grid_faces(num_u - 1, num_v - 1)
    mesh = FDMesh.from_vertices_and_faces(srf_points, faces)
    
    # define structural system
    mesh.vertices_supports(mesh.vertices_on_boundary())
    mesh.vertices_loads(load)
    mesh.edges_forcedensities(q)

    return mesh



def create_mesh_from_tile(tile, u, v, load, q=-1.0):
    """
    Boundary supported mesh.
    """
    # generate base FD Mesh        
    srf_points = get_bezier_surface_points_from_tile(tile)

    num_u = u.shape[0]
    num_v = v.shape[0]
    faces = calculate_grid_faces(num_u - 1, num_v - 1)
    mesh = FDMesh.from_vertices_and_faces(srf_points, faces)
    
    # define structural system
    mesh.vertices_supports(mesh.vertices_on_boundary())
    mesh.vertices_loads(load)
    mesh.edges_forcedensities(q)

    return mesh


