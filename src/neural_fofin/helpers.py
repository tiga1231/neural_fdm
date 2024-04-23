import jax.numpy as jnp

from jax_fdm.equilibrium import EquilibriumParametersState as FDParametersState
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import LoadState
from jax_fdm.equilibrium import nodes_load_from_faces


# ===============================================================================
# Load helpers
# ===============================================================================

def calculate_area_loads(x, structure, load):
    """
    Convert area loads into vertex loads.
    """
    x = jnp.reshape(x, (-1, 3))

    # need to convert loads into face loads
    num_faces = structure.num_faces
    faces_load_xy = jnp.zeros(shape=(num_faces, 2))  # (num_faces, xy)
    faces_load_z = jnp.ones(shape=(num_faces, 1)) * load  # (num_faces, xy)
    faces_load = jnp.hstack((faces_load_xy, faces_load_z))

    vertices_load = nodes_load_from_faces(
        x,
        faces_load,
        structure,
        is_local=False
    )

    return vertices_load

# ===============================================================================
# Form-finding helpers
# ===============================================================================


def edges_vectors(xyz, connectivity):
    """
    Calculate the unnormalized edge directions (nodal coordinate differences).
    """
    return connectivity @ xyz


def edges_lengths(vectors):
    """
    Compute the length of the edges.
    """
    return jnp.linalg.norm(vectors, axis=1, keepdims=True)


def edges_forces(q, lengths):
    """
    Calculate the force in the edges.
    """
    return jnp.reshape(q, (-1, 1)) * lengths


def vertices_residuals(q, loads, vectors, connectivity):
    """
    Compute the residual forces on the vertices of the structure.
    """
    return loads - connectivity.T @ (q[:, None] * vectors)


def vertices_residuals_from_xyz(q, loads, xyz, structure):
    """
    Compute the residual forces on the vertices of the structure.
    """
    connectivity = structure.connectivity
    vectors = edges_vectors(xyz, connectivity)

    return vertices_residuals(q, loads, vectors, connectivity)


def calculate_equilibrium_state(q, xyz, loads_nodes, structure):
    """
    Assembles an equilibrium state object.
    """
    connectivity = structure.connectivity

    vectors = edges_vectors(xyz, connectivity)
    lengths = edges_lengths(vectors)
    residuals = vertices_residuals(q, loads_nodes, vectors, connectivity)
    forces = edges_forces(q, lengths)

    return EquilibriumState(
        xyz=xyz,
        residuals=residuals,
        lengths=lengths,
        forces=forces,
        loads=loads_nodes,
        vectors=vectors
    )


def calculate_fd_params_state(q, xyz_fixed, loads_nodes):
    """
    Assembles an equilibrium state object.
    """
    load_state = LoadState(loads_nodes, 0.0, 0.0)
    return FDParametersState(
            q,
            xyz_fixed,
            load_state
        )
