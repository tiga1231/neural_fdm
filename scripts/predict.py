"""
Predict the force densities and shapes of a batch of target shapes with a pretrained model.
"""
import os
from math import fabs
import yaml

from time import perf_counter
from statistics import mean
from statistics import stdev

import jax
from jax import jit
from jax import vmap
import jax.numpy as jnp

import jax.random as jrn

from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Polygon
from compas.geometry import Line

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import datastructure_updated
from jax_fdm.visualization import Viewer

from neural_fofin import DATA

from neural_fofin.builders import build_loss_function
from neural_fofin.builders import build_mesh_from_generator
from neural_fofin.builders import build_data_generator
from neural_fofin.builders import build_connectivity_structure_from_generator
from neural_fofin.builders import build_neural_model

from neural_fofin.losses import print_loss_summary

from neural_fofin.serialization import load_model


# ===============================================================================
# Script function
# ===============================================================================

def predict_batch(
        model_name,
        task_name,
        seed=None,
        batch_size=None,
        time_batch_inference=True,
        predict_in_sequence=True,
        slice=(0, -1),  # (50, 53) for bezier
        view=False,
        save=False,
        edgecolor="force"):
    """
    Predict a batch of target shapes with a pretrained model.

    Parameters
    ___________
    model_name: `str`
        The model name.
        Supported models are formfinder, autoencoder, and piggy.
        Append the suffix `_pinn` to load model versions that were trained with a PINN loss.
    task_name: `str`
        The name of the YAML config file with the task hyperparameters.
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
    save: `bool`
        If `True`, save the predicted shapes as JSON files.
    edgecolor: `str`
        The color palette for the edges.
        Supported color palettes are "fd" to display force densities, and "force" to show forces.
    """
    START, STOP = slice
    if STOP == -1:
        STOP = batch_size

    EDGECOLOR = edgecolor  # force, fd

    CAMERA_CONFIG = {
        "position": (30.34, 30.28, 42.94),
        "target": (0.956, 0.727, 1.287),
        "distance": 20.0,
    }

    # load yaml file with hyperparameters
    with open(f"{task_name}.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # unpack parameters
    if seed is None:
        seed = config["seed"]
    training_params = config["training"]
    if batch_size is None:
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
    print(f"Making predictions with {model_name} on {generator_name} dataset with {bounds_name} bounds\n")
    print(f"Structure size: {structure.num_vertices} vertices, {structure.num_edges} edges")

    # load model
    filepath = os.path.join(DATA, f"{model_name}_{task_name}.eqx")
    _model_name = model_name.split("_")[0]
    model_skeleton = build_neural_model(_model_name, config, generator, model_key)
    model = load_model(filepath, model_skeleton)

    # sample data batch
    xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))

    encoding_fn = jit(vmap(model.encode))
    encoding_fn(xyz_batch)

    # time inference time (encoding only) on batch
    if time_batch_inference:
        # warmstart
        # time
        times = []
        for i in range(10):
            start = perf_counter()
            encoding_fn(xyz_batch)
            duration = perf_counter() - start
            times.append(duration)
        print(f"Inference time on batch size {batch_size}: {mean(times):.5f} (+-{stdev(times):.5f}) s")

    # report batch losses
    _, loss_terms = compute_loss(model, structure, xyz_batch, aux_data=True)
    print_loss_summary(loss_terms, prefix="Batch\t")

    # make individual predictions
    if not predict_in_sequence:
        return

    print("\nPredicting shapes in sequence")
    opt_times = []
    loss_terms_batch = []
    num_predictions = 0

    for i in range(START, STOP):
        xyz = xyz_batch[i]

        # do inference on one design
        start = perf_counter()
        encoding_fn(xyz[None, :])
        opt_time = perf_counter() - start
        opt_times.append(opt_time)
        num_predictions += 1

        # calculate loss
        _, loss_terms = compute_loss(
            model,
            structure,
            xyz[None, :],
            aux_data=True
        )

        loss_terms_batch.append(loss_terms)
        print_loss_summary(loss_terms, prefix=f"Shape {i}\t")

        if view or save:
            # predict equilibrium states for viz and i/o
            eqstate_hat, fd_params_hat = model.predict_states(xyz, structure)

            # assemble datastructure for post-processing
            mesh_hat = datastructure_updated(mesh, eqstate_hat, fd_params_hat)
            network_hat = FDNetwork.from_mesh(mesh_hat)
            network_hat.print_stats()
            print()

        # export prediction
        if save:
            filename = f"mesh_{task_name}_{i}"
            filepath = os.path.join(DATA, f"{filename}.json")
            mesh_hat.to_json(filepath)
            print(f"Saved prediction to {filepath}")

        # Create target mesh
        mesh_target = mesh.copy()
        _xyz = jnp.reshape(xyz, (-1, 3)).tolist()
        for idx, key in mesh.index_key().items():
            mesh_target.vertex_attributes(key, "xyz", _xyz[idx])

        # visualization
        if view:
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

            # edge colors
            if EDGECOLOR == "force":

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
            else:
                edgecolor = EDGECOLOR

            viewer.add(
                network_hat,
                edgewidth=(0.01, 0.2),
                edgecolor=edgecolor,
                show_edges=True,
                edges=[edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(*edge)],
                nodes=[node for node in mesh.vertices() if len(mesh.vertex_neighbors(node)) > 2],
                show_loads=False,
                loadscale=1.0,
                show_reactions=True,
                reactionscale=1.0,
                reactioncolor=Color.from_rgb255(0, 150, 10),
            )

            if task_name == "bezier":
                # approximated mesh
                viewer.add(
                     mesh_hat,
                     show_points=False,
                     show_edges=False,
                     opacity=0.2
                 )

                # target mesh
                viewer.add(
                    FDNetwork.from_mesh(mesh_target),
                    as_wireframe=True,
                    show_points=False,
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

        # report statistics
    print(f"Inference time over {num_predictions} samples (s): {mean(opt_times):.4f} (+-{stdev(opt_times):.4f})")

    labels = loss_terms_batch[0].keys()
    for label in labels:
        errors = [terms[label].item() for terms in loss_terms_batch]
        print(f"{label.capitalize()} over {num_predictions} samples: {mean(errors):.4f} (+-{stdev(errors):.4f})")

    if task_name == "tower":
        errors = []
        for terms in loss_terms_batch:
            error = 0.0
            error += terms["shape error"].item()
            error += terms["height error"].item()
            errors.append(error)
        print(f"Task error over {num_predictions} samples: {mean(errors):.4f} (+-{stdev(errors):.4f})")

# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(predict_batch)
