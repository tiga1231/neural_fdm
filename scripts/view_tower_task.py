"""
Predict the force densities and shapes of a batch of target shapes with a pretrained model.
"""
import os
from math import fabs
import yaml

import warnings

from time import perf_counter
from statistics import mean
from statistics import stdev

import jax
from jax import vmap
import jax.numpy as jnp

import jax.random as jrn

import equinox as eqx

from jaxopt import ScipyBoundedMinimize

from compas.colors import Color
from compas.colors import ColorMap
from compas.geometry import Polygon
from compas.geometry import Polyline
from compas.geometry import Plane
from compas.geometry import Line
from compas.geometry import Translation
from compas.geometry import Frame
from compas.utilities import remap_values

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import datastructure_updated
from jax_fdm.visualization import Viewer

from neural_fofin import DATA

from neural_fofin.builders import build_mesh_from_generator
from neural_fofin.builders import build_data_generator
from neural_fofin.builders import build_connectivity_structure_from_generator
from neural_fofin.builders import build_fd_decoder_parametrized
from neural_fofin.builders import build_loss_function

from neural_fofin.losses import print_loss_summary


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

# cannon vault
BEZIER_CANNON = [
    [0.0, 0.0, 6.0],
    [0.0, 0.0, 6.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
]

beziers = {
    "pillow": BEZIER_PILLOW,
    "dome": BEZIER_DOME,
    "saddle": BEZIER_SADDLE,
    "cannon": BEZIER_CANNON,
}

CAMERA_CONFIG_BEZIER = {
    "color": (1.0, 1.0, 1.0, 1.0),
    "position": (30.34, 30.28, 42.94),
    "target": (0.956, 0.727, 1.287),
    "distance": 20.0,
}

CAMERA_CONFIG_TOWER = {
    "color": (1.0, 1.0, 1.0, 1.0),
    "position": (10.718, 10.883, 14.159),
    "target": (-0.902, -0.873, 3.846),
    "distance": 19.482960680274577,
    "rotation": (1.013, 0.000, 2.362),
}


# ===============================================================================
# Script function
# ===============================================================================

def optimize_batch(
        optimizer,
        task_name,
        shape_name=None,
        param_init=None,
        blow=1e-3,  # 1e-3
        bup=20.0,
        maxiter=5000,
        seed=None,
        batch_size=None,
        slice=(0, -1),  # (50, 53) for bezier
        save=False,
        view=False,
        view_result=False,
        edgecolor="force",
        show_reactions=False,
        edgewidth=(0.01, 0.15),
        fmax=None,
        fmax_tens=None,
        fmax_comp=None,
        qmin=None,
        qmax=None,
        verbose=True,
):
    """
    Solve the prediction task on a batch target shapes with direct optimization with box constraints.
    The box constraints help generating compression-only or tension-only solutions.

    Parameters
    ___________
    optimizer: `str`
        The name gradient-based optimizer used to solve this task.
        Supported methods are "slsqp" and "lbfgsb".
    task_name: `str`
        The filepath (without extension) of the YAML file with the task hyperparameters.
    shape_name: `str`
        The name of the shape to show.
    param_init: `float`
        If specified, it determines the starting value of all the model parameters.
        If not, then it samples parameters between b_low and b_up from a uniform distribution.
    b_low: `float`
        The lower bound of the box constraints on the model parameters.
    b_up: `float`
        The lower bound of the box constraints on the model parameters.
    maxiter: `int`
        The maximum number of optimization iterations.
    seed: `int`
        The random seed to generate a batch of target shapes.
        If `None`, it defaults to the input hyperparameters file.
    batch_size: `int` or `None`
        The size of the batch of target shapes.
        If `None`, it defaults to the input hyperparameters file.
    verbose: `bool`
        If `True`, print to stdout intermediary results.
    save: `bool`
        If `True`, save the predicted shapes as JSON files.
    view: `bool`
        If `True`, view the predicted shapes.
    slice: `tuple`
        The start of the slice of the batch for saving and viewing.
    edgecolor: `str`
        The color palette for the edges.
        Supported color palettes are "fd" to display force densities, and "force" to show forces.
    """
    START, STOP = slice
    EDGECOLOR = edgecolor  # force, fd
    SAVE = save
    QMIN = blow
    QMAX = bup
    EDGEWIDTH = edgewidth

    # pick camera configuration for task
    if task_name == "bezier":
        CAMERA_CONFIG = CAMERA_CONFIG_BEZIER
        _width = 900
    elif task_name == "tower":
        CAMERA_CONFIG = CAMERA_CONFIG_TOWER
        _width = 450

    # pick optimizer name
    optimizer_names = {"lbfgsb": "L-BFGS-B", "slsqp": "SLSQP"}
    optimizer_name = optimizer_names[optimizer]

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
    fd_params = config["fdm"]

    # randomness
    key = jrn.PRNGKey(seed)
    _, generator_key = jax.random.split(key, 2)

    # create data generator
    generator = build_data_generator(config)
    compute_loss = build_loss_function(config, generator)
    structure = build_connectivity_structure_from_generator(config, generator)
    mesh = build_mesh_from_generator(config, generator)

    # generate initial model parameters
    q0 = calculate_params_init(mesh, param_init, key, QMIN, QMAX)

    # create model
    print(f"Directly optimizing with {optimizer_name} for {generator_name} dataset with {bounds_name} bounds on seed {seed}")
    model = build_fd_decoder_parametrized(q0, mesh, fd_params)

    # sample data batch
    xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))

    # split mode
    diff_model, static_model = eqx.partition(model, eqx.is_inexact_array)

    # wrap loss function to meet jax and jaxopt's ideosyncracies
    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def compute_loss_diffable(diff_model, xyz_target):
        """
        """
        _model = eqx.combine(diff_model, static_model)
        return compute_loss(_model, structure, xyz_target, aux_data=False)

    # warmstart loss function to eliminate jit compilation time from perf measurements
    _ = compute_loss_diffable(diff_model, xyz_target=xyz_batch[None, 0])

    # define optimization function
    warnings.filterwarnings("ignore")

    opt = ScipyBoundedMinimize(
        fun=compute_loss_diffable,
        method=optimizer_name,
        jit=False,
        tol=1e-6,
        maxiter=maxiter,
        options={"disp": False},
        value_and_grad=True,
    )

    # define parameter bounds
    bound_low, bound_up = calculate_params_bounds(mesh, q0, QMIN, QMAX)
    bound_low_tree = eqx.tree_at(lambda tree: tree.q, diff_model, replace=(bound_low))
    bound_up_tree = eqx.tree_at(lambda tree: tree.q, diff_model, replace=(bound_up))
    bounds = (bound_low_tree, bound_up_tree)

    # optimize
    print("\nOptimizing shapes in sequence")
    opt_times = []
    loss_terms_batch = []

    were_successful = 0
    if STOP == -1:
        STOP == batch_size

    xyz_slice = xyz_batch[START:STOP]

    # sample target points from prescribed shape name
    if shape_name is not None and "bezier" in task_name:
        transform = beziers[shape_name]
        transform = jnp.array(transform)
        xyz = generator.evaluate_points(transform)
        xyz_slice = xyz[None, :]

    num_opts = xyz_slice.shape[0]
    for i, xyz in enumerate(xyz_slice):

        xyz = xyz[None, :]

        # report start losses
        _, loss_terms = compute_loss(model, structure, xyz, aux_data=True)
        if verbose:
            print(f"Shape {i}")
            print_loss_summary(loss_terms, prefix="\tStart")

        # optimize
        start = perf_counter()
        diff_model_opt, opt_res = opt.run(diff_model, bounds, xyz)
        opt_time = perf_counter() - start

        # unite optimal and static submodels
        model_opt = eqx.combine(diff_model_opt, static_model)

        # evaluate loss function at optimum point
        _, loss_terms = compute_loss(model_opt, structure, xyz, aux_data=True)
        if verbose:
            print_loss_summary(loss_terms, prefix="\tEnd")
            print(f"\tOpt success?: {opt_res.success}")
            print(f"\tOpt iters: {opt_res.iter_num}")
            print(f"\tOpt time: {opt_time:.4f} sec")

        if opt_res.success:
            were_successful += 1

        opt_times.append(opt_time)
        loss_terms_batch.append(loss_terms)

        # assemble datastructure for post-processing
        eqstate_hat, fd_params_hat = model_opt.predict_states(xyz, structure)
        mesh_hat = datastructure_updated(mesh, eqstate_hat, fd_params_hat)
        network_hat = FDNetwork.from_mesh(mesh_hat)
        if verbose:
            network_hat.print_stats()

        # export prediction
        if SAVE:
            filename = f"mesh_{i}"
            filepath = os.path.join(DATA, f"{filename}.json")
            mesh_hat.to_json(filepath)
            print(f"Saved prediction to {filepath}")

        # visualization
        if view:
            # create target mesh
            mesh_target = mesh.copy()
            _xyz = jnp.reshape(xyz, (-1, 3)).tolist()
            for idx, key in mesh.index_key().items():
                mesh_target.vertex_attributes(key, "xyz", _xyz[idx])

            viewer = Viewer(
                width=_width,
                height=900,
                show_grid=False,
                viewmode="lighted"
            )

            # modify view
            viewer.view.camera.position = CAMERA_CONFIG["position"]
            viewer.view.camera.target = CAMERA_CONFIG["target"]
            viewer.view.camera.distance = CAMERA_CONFIG["distance"]
            _rotation = CAMERA_CONFIG.get("rotation")
            if _rotation:
                viewer.view.camera.rotation = _rotation

            # edges to view
            # NOTE: we are not visualizing edges on boundaries since they are supported
            edges_2_view = [edge for edge in mesh.edges() if not mesh.is_edge_on_boundary(*edge)]

            # compute stats
            forces_all = []
            forces_comp_all = []
            forces_tens_all = []
            qs_all = []
            for edge in network_hat.edges():

                force = network_hat.edge_force(edge)
                force_abs = fabs(force)
                forces_all.append(force_abs)
                if force < 0.0:
                    forces_comp_all.append(force_abs)
                else:
                    forces_tens_all.append(force_abs)

                if mesh_hat.edge_attribute(edge, "tag") != "ring":
                    qs_all.append(fabs(network_hat.edge_forcedensity(edge)))

            fmin = 0.0
            fmin_comp = 0.0
            fmin_tens = 0.0
            if fmax is None:
                fmax = max(forces_all)
            if fmax_tens is None:
                fmax_tens = max(forces_tens_all)
            if fmax_comp is None:
                fmax_comp = max(forces_comp_all)
            if qmin is None:
                qmin = min(qs_all)
            if qmax is None:
                qmax = max(qs_all)

            # edge width
            width_min, width_max = EDGEWIDTH
            _forces = [fabs(mesh_hat.edge_force(edge)) for edge in mesh_hat.edges()]
            _forces = remap_values(_forces, original_min=fmin, original_max=fmax)
            _widths = remap_values(_forces, width_min, width_max)
            edgewidth = {edge: width for edge, width in zip(mesh_hat.edges(), _widths)}

            # edge colors
            edgecolor = EDGECOLOR
            if edgecolor == "fd_minmax":
                edgecolor = {}

                cmap = ColorMap.from_mpl("viridis")
                _edges = [edge for edge in mesh_hat.edges() if mesh_hat.edge_attribute(edge, "tag") != "ring"]
                values = [fabs(mesh_hat.edge_forcedensity(edge)) for edge in _edges]
                ratios = remap_values(values, original_min=qmin, original_max=qmax)
                edgecolor = {edge: cmap(ratio) for edge, ratio in zip(_edges, ratios)}

                for edge in mesh_hat.edges():
                    if mesh_hat.edge_attribute(edge, "tag") == "ring":
                        edgecolor[edge] = Color.grey()  # .darkened()

            elif edgecolor == "force":
                edgecolor = {}

                color_start = Color.white()
                color_comp_end = Color.from_rgb255(12, 119, 184)
                cmap_comp = ColorMap.from_two_colors(color_start, color_comp_end)
                color_tens_end = Color.from_rgb255(227, 6, 75)
                cmap_tens = ColorMap.from_two_colors(color_start, color_tens_end)

                for edge in mesh_hat.edges():

                    force = mesh_hat.edge_force(edge)

                    if force < 0.0:
                        _cmap = cmap_comp
                        _fmin = fmin_comp
                        _fmax = fmax_comp
                    else:
                        _cmap = cmap_tens
                        _fmin = fmin_tens
                        _fmax = fmax_tens

                    value = (fabs(force) - _fmin) / (_fmax - _fmin)
                    edgecolor[edge] = _cmap(value)

            if view_result:
                viewer.add(
                    network_hat,
                    edgewidth=edgewidth,
                    edgecolor=edgecolor,
                    show_edges=True,
                    edges=edges_2_view,
                    nodes=[node for node in mesh.vertices() if len(mesh.vertex_neighbors(node)) > 2],
                    show_loads=False,
                    loadscale=1.0,
                    show_reactions=show_reactions,
                    reactionscale=1.0,
                    reactioncolor=Color.from_rgb255(0, 150, 10),
                )

                viewer.add(
                    mesh_hat,
                    show_points=False,
                    show_edges=False,
                    opacity=0.7,
                    color=Color.grey().lightened(100),
                )

                for _vertices in mesh.vertices_on_boundaries():
                    viewer.add(
                        Polyline([mesh_hat.vertex_coordinates(vkey) for vkey in _vertices]),
                        linewidth=4.0,
                        color=Color.black().lightened()
                        )

            if task_name == "bezier":
                # target mesh
                viewer.add(
                    FDNetwork.from_mesh(mesh_target),
                    as_wireframe=True,
                    show_points=False,
                    linewidth=4.0,
                    color=Color.black().lightened()
                )

                # approximated mesh
                viewer.add(
                    mesh_hat,
                    show_points=False,
                    show_edges=False,
                    opacity=0.2
                )

            elif task_name == "tower":
                rings = jnp.reshape(xyz, generator.shape_tube)[generator.levels_rings_comp, :, :]

                for ring in rings:

                    ring = ring.tolist()

                    polygon = Polygon(ring)

                    # viewer.add(polygon, opacity=0.5)
                    viewer.add(polygon, opacity=0.5)

                    viewer.add(
                        Polyline(ring + ring[:1]),
                        linewidth=4.0,
                        color=Color.black().lightened()
                    )

                # draw planes, transparent, thick-ish boundary
                heights = jnp.linspace(0.0, generator.height, generator.num_levels)
                counter = 0
                from neural_fofin.generators import points_on_ellipse
                for i, height in enumerate(heights):

                    origin_pt = [0.0, 0.0, height]
                    plane = Plane(origin_pt, [0.0, 0.0, 1.0])
                    frame = Frame(origin_pt, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

                    circle = points_on_ellipse(
                        generator.radius,
                        generator.radius,
                        height,
                        generator.num_sides,
                    )
                    circle = circle.tolist()
                    # circle = Polygon.from_sides_and_radius_xy(
                    #         generator.num_sides,
                    #         generator.radius,
                    #     )
                    # circle.transform(Translation.from_vector(origin_pt))
                    # circle = circle.points

                    if i in generator.levels_rings_comp:

                        viewer.add(
                            Polyline(circle + circle[:1]),
                            linewidth=2.0,
                            color=Color.grey().lightened()
                        )

                        # viewer.add(Line(origin_pt, circle[0]), linewidth=2.0)
                        # viewer.add(Line(origin_pt, rings[counter][0]), linewidth=2.0)
                        # counter += 1

                        continue

                    size = 1.0
                    object = viewer.add(plane,
                                        size=size,
                                        linewidth=0.1,
                                        color=Color.grey().lightened(10),
                                        opacity=0.1,
                                        )

                    # square = [frame.to_world_coordinates([-size, -size, 0]),
                    #           frame.to_world_coordinates([size, -size, 0]),
                    #           frame.to_world_coordinates([size, size, 0]),
                    #           frame.to_world_coordinates([-size, size, 0])]

                    # viewer.add(
                    #     Polyline(square + square[:1]),
                    #     linewidth=1.0,
                    #     color=Color.black()  # .lightened()
                    # )

            # show le crème
            viewer.show()

    # report statistics
    print(f"\nSuccessful optimizations: {were_successful}/{num_opts}")
    print(f"Optimization time over {num_opts} optimizations (s): {mean(opt_times):.4f} (+-{stdev(opt_times):.4f})")

    labels = loss_terms_batch[0].keys()
    for label in labels:
        errors = [terms[label].item() for terms in loss_terms_batch]
        print(f"{label.capitalize()} over {num_opts} optimizations: {mean(errors):.4f} (+-{stdev(errors):.4f})")

    if task_name == "tower":
        errors = []
        for terms in loss_terms_batch:
            error = 0.0
            error += terms["shape error"].item() ** 0.5
            error += terms["height error"].item() ** 0.5
            errors.append(error)
        print(f"Sq root of shape error over {num_opts} samples: {mean(errors):.4f} (+-{stdev(errors):.4f})")

# ===============================================================================
# Helper functions
# ===============================================================================


def calculate_params_init(mesh, param_init, key, minval, maxval):
    """
    """
    num_edges = mesh.number_of_edges()

    signs = []
    for edge in mesh.edges():
        sign = -1.0  # compression by default
        # for tower task
        if mesh.edge_attribute(edge, "tag") == "cable":
            sign = 1.0
        signs.append(sign)

    signs = jnp.array(signs)

    if param_init is not None:
        q0 = jnp.ones(num_edges) * param_init
    else:
        q0 = jrn.uniform(key, shape=(num_edges, ), minval=minval, maxval=maxval)

    return q0 * signs


def calculate_params_bounds(mesh, q0, minval, maxval):
    """
    """
    bound_low = []
    bound_up = []
    for edge in mesh.edges():
        # compression by default
        b_low = maxval * -1.0
        b_up = minval * -1.0
        # for tower task
        if mesh.edge_attribute(edge, "tag") == "cable":
            b_low = minval
            b_up = maxval

        bound_low.append(b_low)
        bound_up.append(b_up)

    bound_low = jnp.array(bound_low)
    bound_up = jnp.array(bound_up)

    return bound_low, bound_up


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(optimize_batch)
