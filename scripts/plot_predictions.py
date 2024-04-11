import os

from math import fabs

import yaml

import jax
from jax import vmap
import jax.numpy as jnp

import jax.random as jrn

from compas.colors import Color
from compas.colors import ColorMap

from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import LoadState
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import datastructure_updated

from jax_fdm.visualization import Viewer

from neural_fofin import DATA

from neural_fofin.experiments import build_data_objects
from neural_fofin.experiments import build_neural_object
from neural_fofin.experiments import build_mesh

from neural_fofin.training_coupled import compute_loss_autoencoder

from neural_fofin.serialization import load_model


# local script parameters
VIEW = True
SAVE = False

NAME = "autoencoder"  # "autoencoder"
COLOR_SCHEME = "fd"
START = 50
STOP = 60

CAMERA_CONFIG = {
    "position": (30.34, 30.28, 42.94),
    "target": (0.956, 0.727, 1.287),
    "distance": 20.0,
}

# load yaml file with hyperparameters
with open("config.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# unpack parameters
seed = config["seed"]
grid_params = config["grid"]
generator_params = config["generator"]
batch_size = config["training"]["batch_size"]

# randomness
# seed = 12
key = jrn.PRNGKey(seed)
model_key, generator_key = jax.random.split(key, 2)

# create data generator
generator, structure = build_data_objects(config)
mesh = build_mesh(generator_params, grid_params)
model_skeleton = build_neural_object(config, model_key)

# load model
filepath = os.path.join(DATA, f"{NAME}.eqx")
model = load_model(filepath, model_skeleton)

# sample data batch
xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))

# compute loss
# NOTE: maybe compute loss of sampled object, one by one?
start_loss, fd_data = compute_loss_autoencoder(model, structure, xyz_batch)
print(f"Autoencoder start loss: {start_loss:.6f}")

# make (batched) predictions
for i in range(START, STOP):

    xyz = xyz_batch[i]
    xyz_hat = model(xyz, structure)

    q = model.encoder(xyz) * -1.0
    q_masked = q * model.decoder.mask_edges + model.decoder.qmin

    loads = model.decoder.get_loads(xyz, structure)
    load_state = LoadState(loads, 0.0, 0.0)

    xyz_fixed = model.decoder.get_xyz_fixed(xyz, structure)

    fdm_params_hat = EquilibriumParametersState(q_masked,
                                                xyz_fixed,
                                                load_state)

    eqstate_hat = model.decoder.model(fdm_params_hat, structure)

    mesh_hat = datastructure_updated(mesh, eqstate_hat, fdm_params_hat)
    network_hat = FDNetwork.from_mesh(mesh_hat)

    mesh_target = mesh.copy()
    _xyz = jnp.reshape(xyz, (-1, 3)).tolist()
    for idx, key in mesh.index_key().items():
        mesh_target.vertex_attributes(key, "xyz", _xyz[idx])

    # ==========================================================================
    # Visualization
    # ==========================================================================

    viewer = Viewer(
        width=900,
        height=900,
        show_grid=False,
        viewmode="lighted"
    )

    # modify view
    viewer.view.camera.position = CAMERA_CONFIG["position"]
    viewer.view.camera.target = CAMERA_CONFIG["target"]
    viewer.view.camera.distance = CAMERA_CONFIG["distance"]

    # approximated mesh
    viewer.add(
        mesh_hat,
        show_points=False,
        show_edges=False,
        opacity=0.2
    )

    # edge colors
    color_end = Color.from_rgb255(12, 119, 184)
    color_start = Color.white()
    cmap = ColorMap.from_two_colors(color_start, color_end)

    edgecolor = {}
    forces = [fabs(network_hat.edge_force(edge)) for edge in network_hat.edges()]
    fmin = min(forces)
    fmax = max(forces)

    for edge in network_hat.edges():
        force = network_hat.edge_force(edge) * -1.0
        if force < 0.0:
            _color = Color.from_rgb255(227, 6, 75)
        else:
            value = (force - fmin) / (fmax - fmin)
            _color = cmap(value)

        edgecolor[edge] = _color

    viewer.add(network_hat,
               edgewidth=(0.01, 0.3),
               edgecolor=edgecolor,
               show_edges=True,
               edges=[edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(*edge)],
               nodes=[node for node in mesh.vertices() if len(mesh.vertex_neighbors(node)) > 2],
               show_loads=False,
               loadscale=0.5,
               show_reactions=True,
               reactionscale=0.5,
               reactioncolor=Color.from_rgb255(0, 150, 10),
               )

    viewer.add(FDNetwork.from_mesh(mesh_target),
               as_wireframe=True,
               show_points=False,
               linewidth=5.0,
               color=Color.black().lightened()
               )

    # show le crème
    viewer.show()

print("Yamanaka-ko!")
