import os

import yaml

from math import fabs

import matplotlib.cm as cm

import jax

import numpy as np
import jax.numpy as jnp

import jax.random as jrn

from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import distance_point_point
from compas.geometry import length_vector
from compas.utilities import remap_values

from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import LoadState
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import datastructure_updated

from jax_fdm.visualization import Plotter
from jax_fdm.visualization import Viewer

from neural_fofin import DATA
from neural_fofin import FIGURES

from neural_fofin.bezier import evaluate_bezier_surface

from neural_fofin.experiments import build_data_objects
from neural_fofin.experiments import build_neural_objects
from neural_fofin.experiments import build_mesh
from neural_fofin.experiments import build_point_grid

from neural_fofin.models import AutoEncoder
from neural_fofin.models import PiggyDecoder
from neural_fofin.models import ForceDensityModel

from neural_fofin.training_coupled import compute_loss_autoencoder
from neural_fofin.training_coupled import compute_loss_piggybacker

from neural_fofin.serialization import load_model


# local script parameters
VIEW = False
PLOT = True
SAVE = True

NAME_AUTOENCODER = "autoencoder"
NAME_DECODER = "decoder"

EXAMPLE_NAME = "pillow"  # pillow, dome, saddle
PLOT_MODE = "residuals"  # deltas, residuals, forces

# pillow
TRANSLATION_PILLOW = [
    [0.0, 0.0, 10.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
]

# circular dome
TRANSLATION_DOME = [
    [0.0, 0.0, 10.0],
    [2.5, 0.0, 0.0],
    [0.0, 2.5, 0.0],
    [0.0, 0.0, 0.0]
]

# cute saddle
TRANSLATION_SADDLE = [
    [0.0, 0.0, 1.5],
    [-1.25, 0.0, 5.0],
    [0.0, -2.5, 0.0],
    [0.0, 0.0, 0.0]
]

translations = {
    "pillow": TRANSLATION_PILLOW,
    "dome": TRANSLATION_DOME,
    "saddle": TRANSLATION_SADDLE,
}

TRANSLATION = translations[EXAMPLE_NAME]

CAMERA_CONFIG = {
    "color": (1.0, 1.0, 1.0, 1.0),
    "position": (30.34, 30.28, 42.94),
    "target": (0.956, 0.727, 1.287),
    "distance": 20.0,
}


# ===============================================================================
# Helper functions
# ===============================================================================


def get_force_densities(model, xyz):
    """
    """
    q = model.encoder(xyz) * -1.0
    return q * model.decoder.mask_edges + model.decoder.qmin


def get_loads(model, xyz, structure):
    """
    """
    loads = model.decoder.get_loads(xyz, structure)
    return loads, LoadState(loads, 0.0, 0.0)


def get_xyz_hat(xyz_free_hat, xyz_fixed, structure):
    """
    """
    indices = structure.indices_freefixed
    xyz_free_hat = jnp.reshape(xyz_free_hat, (-1, 3))
    return jnp.concatenate((xyz_free_hat, xyz_fixed))[indices, :]


def get_xyz_fixed(model, xyz, structure):
    """
    """
    return model.decoder.get_xyz_fixed(xyz, structure)


def predict_neural_formfinding(model, xyz):
    """
    Predict.
    """
    # Predict geometry
    xyz_hat = model(xyz, structure)

    # Get force densities to compute residuals
    q_masked = get_force_densities(model, xyz)

    # Get applied loads
    loads = model.decoder.get_loads(xyz, structure)
    load_state = LoadState(loads, 0.0, 0.0)

    # Extract support positions
    xyz_fixed = get_xyz_fixed(model, xyz, structure)

    # Equilibrium parameters
    fd_params_hat = EquilibriumParametersState(
        q_masked,
        xyz_fixed,
        load_state
    )

    eqstate_hat = model.decoder.model.equilibrium_state(
        q_masked,
        jnp.reshape(xyz_hat, (-1, 3)),
        loads,
        structure
    )

    return eqstate_hat, fd_params_hat


def predict_neural_neural(models, xyz):
    """
    Predict.
    """
    # Unpack model since it is a tuple of a reference model and an autoencoder
    model, reference_model = models

    # Predict geometry
    xyz_free_hat = model(xyz, structure)

    # Get force densities to compute residuals
    q_masked = get_force_densities(model, xyz)

    # Get applied loads
    loads, load_state = get_loads(reference_model, xyz, structure)

    # Extract support positions
    xyz_fixed = get_xyz_fixed(reference_model, xyz, structure)

    # Concatenate free and fixed coordinates
    xyz_hat = get_xyz_hat(xyz_free_hat, xyz_fixed, structure)

    # Equilibrium parameters
    fd_params_hat = EquilibriumParametersState(
        q_masked,
        xyz_fixed,
        load_state
    )

    eqstate_hat = reference_model.decoder.model.equilibrium_state(
        q_masked,
        xyz_hat,
        loads,
        structure
    )

    return eqstate_hat, fd_params_hat


# ===============================================================================
# Load YAML file with hyperparameters
# ===============================================================================

print("\nCreating experiment")
with open("config.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# unpack parameters
seed = config["seed"]
grid_params = config["grid"]
generator_params = config["generator"]
batch_size = config["training"]["batch_size"]
num_u = generator_params["num_uv"]
num_v = generator_params["num_uv"]

# ===============================================================================
# Create experiment objects
# ===============================================================================

_, structure = build_data_objects(config)
mesh = build_mesh(generator_params, grid_params)
print(mesh)

# ===============================================================================
# Load models
# ===============================================================================

print("\nLoading models")

# randomness
key = jrn.PRNGKey(seed)
model_key, _ = jax.random.split(key, 2)

skeletons = build_neural_objects(config, model_key)

models = []
names = (NAME_AUTOENCODER, NAME_DECODER)
for model_name, skeleton in zip(names, skeletons):
    filepath = os.path.join(DATA, f"{model_name}.eqx")
    _model = load_model(filepath, skeleton)
    models.append(_model)

neural_formfinder, piggy_decoder = models
neural_neural = AutoEncoder(neural_formfinder.encoder, piggy_decoder)

# ===============================================================================
# Generate target XYZ
# ===============================================================================

print("\nGenerating data sample")
grid = build_point_grid(grid_params)
translation = jnp.asarray(TRANSLATION)
control_points = grid.points(translation)

u = jnp.linspace(0.0, 1.0, num_u)
v = jnp.linspace(0.0, 1.0, num_v)
surface_points = evaluate_bezier_surface(control_points, u, v)
xyz_target = surface_points.ravel()  # NOTE: One flat data point!

# reference network
mesh_target = mesh.copy()
_xyz = jnp.reshape(xyz_target, (-1, 3)).tolist()
for idx, key in mesh.index_key().items():
    mesh_target.vertex_attributes(key, "xyz", _xyz[idx])

# ===============================================================================
# Predictions
# ===============================================================================

models_data = {
    "neural_formfinding": {"models": (neural_formfinder, )},
    "neural_neural": {"models": (neural_neural, neural_formfinder)}
}

# make (batched) predictions
print("\nPredicting")
for name, data in models_data.items():
    models = data["models"]

    if isinstance(models[0].decoder, ForceDensityModel):
        # Predict with autoencoder
        eqstate, fd_params = predict_neural_formfinding(models[0], xyz_target)
        example_loss, _ = compute_loss_autoencoder(models[0], structure, xyz_target[None, :])
        print(f"Neural form-finder loss: {example_loss:.6f}")

    elif isinstance(models[0].decoder, PiggyDecoder):
        # Predict with autoencoder
        eqstate, fd_params = predict_neural_neural(models, xyz_target)
        _, fd_data = compute_loss_autoencoder(models[1], structure, xyz_target[None, :])
        example_loss = compute_loss_piggybacker(models[0].decoder, structure, fd_data)
        print(f"Neural-neural loss: {example_loss:.6f}")

    mesh_hat = datastructure_updated(mesh, eqstate, fd_params)
    network_hat = FDNetwork.from_mesh(mesh_hat)

    models_data[name]["mesh"] = mesh_hat
    models_data[name]["network"] = network_hat

# ===============================================================================
# Statistics
# ===============================================================================

print("\nCalculating statistics")
deltas_all = []
residuals_all = []
forces_all = []
for name, data in models_data.items():

    mesh = data["mesh"]
    network = data["network"]

    for vkey in mesh.vertices():
        xyz_target = mesh_target.vertex_coordinates(vkey)
        xyz = mesh.vertex_coordinates(vkey)
        delta = distance_point_point(xyz_target, xyz)
        deltas_all.append(delta)

    for vkey in mesh.vertices():
        if mesh.is_vertex_on_boundary(vkey):
            continue
        residual = mesh.vertex_attributes(vkey, ["rx", "ry", "rz"])
        residuals_all.append(length_vector(residual))

    for edge in network.edges():
        force = network.edge_force(edge)
        if force > 0.0:
            force = 0.0
        forces_all.append(fabs(force))

delta_min = 0.0
delta_max = max(deltas_all)
print(f"{delta_max=}")

residual_min = 0.0
residual_max = max(residuals_all)
print(f"{residual_max=}")

fmin = min(forces_all)
fmax = max(forces_all)
print(f"{fmin=} {fmax=}")

# ===============================================================================
# Viewer
# ===============================================================================

if VIEW:

    print("\nViewing")

    viewer = Viewer(
        width=900,
        height=900,
        show_grid=False,
        viewmode="lighted"
    )

    # modify view
    viewer.view.color = CAMERA_CONFIG["color"]
    viewer.view.camera.position = CAMERA_CONFIG["position"]
    viewer.view.camera.target = CAMERA_CONFIG["target"]
    viewer.view.camera.distance = CAMERA_CONFIG["distance"]

    # view target mesh
    viewer.add(
        mesh_target,
        show_points=False,
        show_edges=False,
        opacity=0.2,
    )

    viewer.add(FDNetwork.from_mesh(mesh_target),
               as_wireframe=True,
               show_points=False,
               linewidth=10.0,
               color=Color.black().lightened())

    viewer.show()

    # display predictions
    for name, data in models_data.items():

        viewer = Viewer(
            width=900,
            height=900,
            show_grid=False,
            viewmode="lighted"
        )

        # modify view
        viewer.view.color = CAMERA_CONFIG["color"]
        viewer.view.camera.position = CAMERA_CONFIG["position"]
        viewer.view.camera.target = CAMERA_CONFIG["target"]
        viewer.view.camera.distance = CAMERA_CONFIG["distance"]

        # query datastructures
        network_hat = data["network"]
        mesh_hat = data["mesh"]

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

# ===============================================================================
# Plotter
# ===============================================================================

if PLOT:

    print(f"\nPlotting {PLOT_MODE}")

    for name, data in models_data.items():
        print(f"\n{name}")

        # NOTE: This is temp!
        if name == "autoencoder":
            continue

        plotter = Plotter(
            figsize=(9, 9),
            dpi=150
        )

        # node work
        network_hat = data["network"]
        mesh = data["mesh"]
        nodes_to_plot = [node for node in mesh.vertices() if len(mesh.vertex_neighbors(node)) > 2]
        edges_to_plot = [edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(*edge)]
        nodesize = 12.0

        # plot sheets
        if PLOT_MODE == "forces":

            edgecolor = "force"
            nodecolor = Color.white()
            nodes_to_plot = [node for node in mesh.vertices() if not mesh.is_vertex_on_boundary(node)]

            edges_to_plot = list(mesh.edges())
            edgewidth = (1.0, 7.0)

            show_reactions = True
            reaction_color = Color.pink()
            reaction_scale = 2.0

            plotter.add(mesh,
                        show_vertices=False,
                        show_edges=False,
                        facecolor={fkey: Color(0.95, 0.95, 0.95) for fkey in mesh.faces()})

        elif PLOT_MODE == "residuals":

            edgecolor = Color(0.1, 0.1, 0.1)
            edgewidth = 1.0

            # nodes_to_plot = [node for node in mesh.vertices() if not mesh.is_vertex_on_boundary(node)]
            # nodes_to_plot = list.
            edges_to_plot = list(mesh.edges())

            show_reactions = False
            reaction_color = Color.pink()
            reaction_scale = 0.5

            # cmap = cm.Oranges_r
            cmap = cm.RdPu
            vmin = residual_min
            vmax = residual_max

            residuals = []
            for vkey in mesh.vertices():
                residual_vector = mesh.vertex_attributes(vkey, ["rx", "ry", "rz"])
                residual = length_vector(residual_vector)
                if mesh.is_vertex_on_boundary(vkey):
                    residual = 0.0
                residuals.append(residual)

            scales = remap_values(
                residuals,
                original_min=residual_min,
                original_max=residual_max
            )

            z = residuals

            # nodesize = {}
            nodecolor = {}
            for key, scale in zip(mesh.vertices(), scales):
                # nodesize[key] = scale * ns_max
                nodecolor[key] = cmap(scale)

            # plotter.add(mesh,
            #             show_vertices=False,
            #             show_edges=False,
            #             facecolor={fkey: Color(0.95, 0.95, 0.95) for fkey in mesh.faces()})

        # delta plotting
        elif PLOT_MODE == "deltas":

            edgecolor = Color(0.1, 0.1, 0.1)
            cmap = cm.turbo
            edgewidth = 1.0
            show_reactions = False
            reaction_color = None
            reaction_scale = 0.5

            deltas = []
            for vkey in mesh.vertices():
                xyz_target = mesh_target.vertex_coordinates(vkey)
                xyz = mesh.vertex_coordinates(vkey)
                delta = distance_point_point(xyz_target, xyz)

                deltas.append(delta)

            scales = remap_values(
                deltas,
                original_min=delta_min,
                original_max=delta_max
            )

            vmin = delta_min
            vmax = delta_max
            z = deltas

        if PLOT_MODE in ("deltas", "residuals"):
            # mesh grid
            x = np.array(mesh.vertices_attribute("x"))
            y = np.array(mesh.vertices_attribute("y"))
            X = np.reshape(x, (10, 10))
            Y = np.reshape(y, (10, 10))
            Z = np.reshape(np.array(z), (10, 10))

            nodecolor = {}
            for key, scale in zip(mesh.vertices(), scales):
                nodecolor[key] = cmap(scale)

            plotter.axes.pcolormesh(
                X,
                Y,
                Z,
                vmin=vmin,
                vmax=vmax,
                shading='gouraud',
                cmap=cmap,
                zorder=1,
            )

        # query datastructures
        plotter.add(
            network_hat,
            edgewidth=edgewidth,
            edgecolor=edgecolor,
            show_edges=True,
            edges=edges_to_plot,
            show_nodes=True,
            nodes=nodes_to_plot,
            nodesize=nodesize,
            nodecolor=nodecolor,
            show_loads=False,
            loadscale=0.5,
            show_reactions=show_reactions,
            reactionscale=reaction_scale,
            reactioncolor=reaction_color,
            sizepolicy="absolute",
        )

        # show le crème
        plotter.zoom_extents()

        if SAVE:
            parts = [EXAMPLE_NAME, name, PLOT_MODE]
            filename = f"{'_'.join(parts)}_plot.png"
            FILE_OUT = os.path.abspath(os.path.join(FIGURES, filename))
            print(f"\nSaving plot to {FILE_OUT}")
            plotter.save(FILE_OUT, bbox_inches=0.0, transparent=True)

        plotter.show()

print("\nYamanaka-ko!")
