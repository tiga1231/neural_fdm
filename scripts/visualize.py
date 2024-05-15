"""
Predict the force densities and shapes of a batch of target shapes with a pretrained model.
"""
import os
from math import fabs
import yaml

import matplotlib.cm as cm
import numpy as np

import jax
import jax.numpy as jnp

from jax import vmap

import jax.random as jrn

from compas.colors import Color
from compas.geometry import Polygon
from compas.geometry import Polyline
from compas.geometry import Line
from compas.geometry import distance_point_point
from compas.geometry import length_vector
from compas.utilities import remap_values

from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import datastructure_updated

from jax_fdm.visualization import Viewer
from jax_fdm.visualization import Plotter

from neural_fofin import DATA
from neural_fofin import FIGURES

from neural_fofin.builders import build_loss_function
from neural_fofin.builders import build_mesh_from_generator
from neural_fofin.builders import build_data_generator
from neural_fofin.builders import build_connectivity_structure_from_generator
from neural_fofin.builders import build_neural_model

from neural_fofin.losses import print_loss_summary

from neural_fofin.serialization import load_model


# ===============================================================================
# Globals -- Don't do this at home!
# ===============================================================================

# pillow
BEZIER_PILLOW = [
    [0.0, 0.0, 10.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
]

# circular dome
BEZIER_DOME = [
    [0.0, 0.0, 10.0],
    [2.75, 0.0, 0.0],
    [0.0, 2.75, 0.0],
    [0.0, 0.0, 0.0]
]

# cute saddle
BEZIER_SADDLE = [
    [0.0, 0.0, 1.5],
    [-1.25, 0.0, 5.0],
    [0.0, -2.5, 0.0],
    [0.0, 0.0, 0.0]
]

beziers = {
    "pillow": BEZIER_PILLOW,
    "dome": BEZIER_DOME,
    "saddle": BEZIER_SADDLE,
}

CAMERA_CONFIG = {
    "color": (1.0, 1.0, 1.0, 1.0),
    "position": (30.34, 30.28, 42.94),
    "target": (0.956, 0.727, 1.287),
    "distance": 20.0,
}


# ===============================================================================
# Script function
# ===============================================================================

def visualize(
        model_names,
        task_name,
        shape_name,
        view=True,
        plot=True,
        save=False,
        edgewidth=(0.01, 0.3),
        edgecolor="force",
        reactionscale=0.5,
        plot_metric="deltas",
):
    """
    Predict a batch of target shapes with a pretrained model.

    Parameters
    ___________
    model_names: `str`
        The model name.
        Supported models are formfinder, autoencoder, and piggy.
        Append the suffix `_pinn` to load model versions that were trained with a PINN loss.
    task_name: `str`
        The filepath (without extension) of the YAML file with the task hyperparameters.
    shape_name: `str`
        The name of the shape to show.
    seed: `int`
        The random seed to generate a batch of target shapes.
        If `None`, it defaults to the input hyperparameters file.
    batch_size: `int` or `None`
        The size of the batch of target shapes.
        If `None`, it defaults to the input hyperparameters file.
    time_batch_inference: `bool`
        If `True`, report the inference time over a data batch, averaged over 10 jitted runs.
    predict_in_sequence: `bool`
        If `True`, predict every shape in the prescribed slice of the data batch.
    slice: `tuple`
        The start of the slice of the batch for saving and viewing.
    view: `bool`
        If `True`, view the predicted shapes.
    plot: `bool`
        If `True`, plot the predicted shapes.
    save: `bool`
        If `True`, save the plots.
    edgecolor: `str`
        The color palette for the edges.
        Supported color palettes are "fd" to display force densities, and "force" to show forces.
    plot_metric: `str`
        The name of the metric to plot.
        Supported metrics are forces, residuals, and deltas.
    """
    PLOT_MODE = plot_metric
    EDGEWIDTH = edgewidth
    EDGECOLOR = edgecolor  # force, fd

    # load yaml file with hyperparameters
    with open(f"{task_name}.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # unpack parameters
    seed = config["seed"]
    training_params = config["training"]
    batch_size = training_params["batch_size"]

    generator_name = config['generator']['name']
    bounds_name = config['generator']['bounds']

    # randomness
    key = jrn.PRNGKey(seed)
    model_key, generator_key = jax.random.split(key, 2)

    # create data generator
    generator = build_data_generator(config)
    structure = build_connectivity_structure_from_generator(config, generator)
    mesh = build_mesh_from_generator(config, generator)
    compute_loss = build_loss_function(config, generator)

    # print info
    print(f"Making predictions with {model_names} on {generator_name} dataset with {bounds_name} bounds\n")
    print(f"Structure size: {structure.num_vertices} vertices, {structure.num_edges} edges")

# ===============================================================================
# Create target
# ===============================================================================

    # sample target points
    # sample target points at random if no shape name
    if shape_name and "bezier" in task_name:
        transform = beziers[shape_name]
        transform = jnp.array(transform)
        xyz = generator.evaluate_points(transform)
    else:
        xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))
        xyz = xyz_batch[0, :]


    # create target mesh
    mesh_target = mesh.copy()
    _xyz = jnp.reshape(xyz, (-1, 3)).tolist()
    for idx, key in mesh.index_key().items():
        mesh_target.vertex_attributes(key, "xyz", _xyz[idx])

# ===============================================================================
# Load models
# ===============================================================================

    print("\nLoading models")
    models_data = {model_name: {} for model_name in model_names}

    for model_name in model_names:
        filepath = os.path.join(DATA, f"{model_name}_{task_name}.eqx")
        _model_name = model_name.split("_")[0]
        model_skeleton = build_neural_model(_model_name, config, generator, model_key)
        model = load_model(filepath, model_skeleton)
        models_data[model_name]["model"] = model

# ===============================================================================
# Predictions
# ===============================================================================

    for model_name, data in models_data.items():
        print(f"\nPredicting with {model_name}")

        # get model
        model = data["model"]

        # calculate loss
        _, loss_terms = compute_loss(
            model,
            structure,
            xyz[None, :],
            aux_data=True
        )
        print_loss_summary(loss_terms)

        # predict equilibrium states for viz and i/o
        eqstate_hat, fd_params_hat = model.predict_states(xyz, structure)

        # assemble datastructure for post-processing
        mesh_hat = datastructure_updated(mesh, eqstate_hat, fd_params_hat)
        network_hat = FDNetwork.from_mesh(mesh_hat)
        network_hat.print_stats()
        print()

        # store datastructures
        models_data[model_name]["mesh"] = mesh_hat
        models_data[model_name]["network"] = network_hat

# ===============================================================================
# Predictions
# ===============================================================================

    print("\nCalculating statistics")
    deltas_all = []
    residuals_all = []
    forces_all = []

    for _, data in models_data.items():

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
    print(f"{delta_min=} {delta_max=:.4f}")

    residual_min = 0.0
    residual_max = max(residuals_all)
    print(f"{residual_min=} {residual_max=:.4f}")

    fmin = min(forces_all)
    fmax = max(forces_all)
    print(f"{fmin=:.4f} {fmax=:.4f}")

# ===============================================================================
# Viewing
# ===============================================================================

    # visualization
    if view:

        # view target mesh alone
        if task_name == "bezier":
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

            # target shape
            viewer.add(
                 mesh_target,
                 show_points=False,
                 show_edges=False,
                 opacity=0.3
             )

            viewer.add(
                FDNetwork.from_mesh(mesh_target),
                as_wireframe=True,
                show_points=False,
                linewidth=4.0,
                color=Color.black().lightened()
                )

            viewer.show()

        # view each model prediction
        for model_name, data in models_data.items():
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

            # query datastructures
            network_hat = data["network"]
            mesh_hat = data["mesh"]

            # edge width
            width_min, width_max = EDGEWIDTH
            _forces = [fabs(network_hat.edge_force(edge)) for edge in network_hat.edges()]
            _forces = remap_values(_forces, fmin, fmax)
            widths = remap_values(_forces, width_min, width_max)
            edgewidth = {edge: width for edge, width in zip(network_hat.edges(), widths)}

            # edge colors
            edgecolor = EDGECOLOR
            # vertices to view
            vertices_2_view = list(mesh.vertices())

            # reaction view
            _reactionscale = reactionscale
            if model_name == "autoencoder":
                _reactionscale *= 0.3
            # reactioncolor = Color.from_rgb255(0, 150, 10)  # load green
            reactioncolor = Color.grey().darkened()  # load green
            show_reactions = True
            # if EDGECOLOR == "fd":
            #    show_reactions = False

            if task_name == "bezier":

                vertices_2_view = []
                for vkey in mesh.vertices():
                    if len(mesh.vertex_neighbors(vkey)) < 3:
                        continue
                    # if mesh.is_vertex_on_boundary(vkey):
                    #    continue
                    vertices_2_view.append(vkey)

                _reactioncolor = {}
                for vkey in mesh.vertices():
                    _color = Color.pink()
                    if mesh.is_vertex_on_boundary(vkey):
                        _color = reactioncolor
                    _reactioncolor[vkey] = _color
                reactioncolor = _reactioncolor

            # display stylized network
            viewer.add(
                network_hat,
                edgewidth=edgewidth,
                edgecolor=edgecolor,
                show_edges=True,
                edges=[edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(*edge)],
                nodes=vertices_2_view,
                show_loads=False,
                loadscale=1.0,
                show_reactions=show_reactions,
                reactionscale=_reactionscale,
                reactioncolor=reactioncolor
            )

            if task_name == "bezier":
                # approximated mesh
                viewer.add(
                     mesh_hat,
                     show_points=False,
                     show_edges=False,
                     opacity=0.3
                 )

                # target mesh
                if edgecolor != "fd":
                    viewer.add(
                        FDNetwork.from_mesh(mesh_target),
                        as_wireframe=True,
                        show_points=False,
                        linewidth=4.0,
                        color=Color.black().lightened()
                        )
                else:
                    viewer.add(
                        mesh_hat,
                        show_points=False,
                        show_edges=False,
                        opacity=0.3
                    )

                    viewer.add(
                        Polyline([mesh_hat.vertex_coordinates(vkey) for vkey in mesh.vertices_on_boundary()]),
                        linewidth=4.0,
                        color=Color.black().lightened()
                    )

            elif task_name == "tower":
                rings = jnp.reshape(xyz, generator.shape_tube)[generator.levels_rings_comp, :, :]
                for ring in rings:
                    ring = Polygon(ring.tolist())
                    viewer.add(ring, opacity=0.5)

                xyz_hat = model(xyz, structure)
                rings_hat = jnp.reshape(xyz_hat, generator.shape_tube)[generator.levels_rings_comp, :, :]
                for ring_a, ring_b in zip(rings, rings_hat):
                    for pt_a, pt_b in zip(ring_a, ring_b):
                        viewer.add(Line(pt_a, pt_b))

            # show le crème
            viewer.show()

# ===============================================================================
# Plotting
# ===============================================================================

    if plot:

        nodes_to_plot = [node for node in mesh.vertices() if len(mesh.vertex_neighbors(node)) > 2]
        edges_to_plot = [edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(*edge)]
        nodesize = 12.0

        plotter = Plotter(
                figsize=(9, 9),
                dpi=150
            )

        plotter.add(mesh,
                    show_vertices=True,
                    vertexsize=nodesize,
                    vertexcolor={vkey: Color.white() for vkey in mesh.vertices()},
                    edgecolor={edge: Color(0.1, 0.1, 0.1) for edge in mesh.edges()},
                    edgewidth={edge: 1.0 for edge in mesh.edges()},
                    show_edges=True,
                    facecolor={fkey: Color(0.95, 0.95, 0.95) for fkey in mesh.faces()})
        # zoom in
        plotter.zoom_extents()

        if save:
            parts = [task_name, shape_name]
            filename = f"plot_{'_'.join(parts)}_target.png"
            FILE_OUT = os.path.abspath(os.path.join(FIGURES, filename))
            print(f"Saving plot to {FILE_OUT}")
            plotter.save(FILE_OUT, bbox_inches=0.0, transparent=True)

        # show le crème
        plotter.show()

        for name, data in models_data.items():
            print(f"\nPlotting {PLOT_MODE} for {name}")

            plotter = Plotter(
                figsize=(9, 9),
                dpi=150
            )

            # node work
            network_hat = data["network"]
            mesh_hat = data["mesh"]

            # plot sheets
            if PLOT_MODE == "fd":

                edgecolor = PLOT_MODE
                nodecolor = Color.white()
                # nodes_to_plot = [node for node in mesh.vertices() if not mesh.is_vertex_on_boundary(node)]
                edges_to_plot = list(mesh.edges())
                edgewidth = (1.0, 9.0)

                show_reactions = False
                reaction_color = Color.pink()
                reaction_scale = 2.0
                if name == "autoencoder":
                    reaction_scale = 0.1

                plotter.add(mesh_hat,
                            show_vertices=False,
                            show_edges=False,
                            facecolor={fkey: Color(0.95, 0.95, 0.95) for fkey in mesh.faces()})

            elif PLOT_MODE == "residual":

                edgecolor = Color(0.1, 0.1, 0.1)
                edgewidth = 1.0

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
                    residual_vector = mesh_hat.vertex_attributes(vkey, ["rx", "ry", "rz"])
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

                nodecolor = {}
                for key, scale in zip(mesh.vertices(), scales):
                    nodecolor[key] = cmap(scale)

            # delta plotting
            elif PLOT_MODE == "delta":

                edgecolor = Color(0.1, 0.1, 0.1)
                cmap = cm.turbo
                edgewidth = 1.0
                show_reactions = False
                reaction_color = None
                reaction_scale = 0.5

                deltas = []
                for vkey in mesh.vertices():
                    xyz_target = mesh_target.vertex_coordinates(vkey)
                    xyz = mesh_hat.vertex_coordinates(vkey)
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

            if PLOT_MODE in ("delta", "residual"):
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
                # sizepolicy="absolute",
            )

            # zoom in
            plotter.zoom_extents()

            if save:
                parts = [task_name, shape_name, PLOT_MODE, name]
                filename = f"plot_{'_'.join(parts)}.png"
                FILE_OUT = os.path.abspath(os.path.join(FIGURES, filename))
                print(f"Saving plot to {FILE_OUT}")
                plotter.save(FILE_OUT, bbox_inches=0.0, transparent=True)

            # show le crème
            plotter.show()


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(visualize)
