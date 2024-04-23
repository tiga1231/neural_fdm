"""
Predict the force densities and shapes of a batch of target shapes with a pre-trained model.
"""
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
from jax_fdm.equilibrium import datastructure_updated
from jax_fdm.visualization import Viewer

from neural_fofin import DATA

from neural_fofin.builders import build_mesh
from neural_fofin.builders import build_data_generator
from neural_fofin.builders import build_connectivity_structure
from neural_fofin.builders import build_neural_model

from neural_fofin.losses import compute_loss

from neural_fofin.serialization import load_model


# local script parameters
VIEW = True
SAVE = False

NAME = "autoencoder_pinn"  # formfinder, autoencoder, autoencoder_pinn
START = 50
STOP = 53

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
training_params = config["training"]
batch_size = training_params["batch_size"]
loss_params = config["loss"]

# randomness
key = jrn.PRNGKey(seed)
model_key, generator_key = jax.random.split(key, 2)

# create data generator
generator = build_data_generator(config)
structure = build_connectivity_structure(config)
mesh = build_mesh(config)

# load model
filepath = os.path.join(DATA, f"{NAME}.eqx")
_model_name = NAME.split("_")[0]
model_skeleton = build_neural_model(_model_name, config, model_key)
model = load_model(filepath, model_skeleton)

# sample data batch
xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))

# make (batched) predictions
print(f"Making predictions with {NAME}")
for i in range(START, STOP):
    xyz = xyz_batch[i]

    xyz_hat = model(xyz, structure)
    eqstate_hat, fd_params_hat = model.predict_states(xyz, structure)

    _, loss_terms = compute_loss(model, structure, xyz[None, :], loss_params, True)
    train_loss, shape_error, residual_error = loss_terms
    print(f"Shape {i}\tTrain loss: {train_loss:.4f}\tShape error: {shape_error:.4f}\tResidual error: {residual_error:.4f}")

    mesh_hat = datastructure_updated(mesh, eqstate_hat, fd_params_hat)
    network_hat = FDNetwork.from_mesh(mesh_hat)

    # Create target mesh
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
