from compas.datastructures import Network

from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization import Viewer


def view(xyz_batch, mesh_skeleton, viewer=None, viewer_kwargs=None):
    """
    """
    if not viewer:
        viewer_kwargs = viewer_kwargs or {}
        viewer = Viewer()
