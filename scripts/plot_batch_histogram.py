import os

import yaml

from math import fabs

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import jax

from jax import vmap

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
PLOT = False
SAVE = False

NAME_AUTOENCODER = "autoencoder"
NAME_DECODER = "decoder"

# Normalization constants
NORMALIZE = True
BBOX_DIAG = 30.0  # (20x20x10)
RHO = 0.5  # magnitude of vertical load

SCALE_UP = True
SCALE = 1000

PLOT_CONFIG = {
    "num_bins": 7,
    "figsize": (9, 5),  # (9, 7.5), (9, 5)
    "dpi": 200,
    "extension": "png"
}

MODELS_PLOT_CONFIG = {
    "neural_formfinding": {"color": "tab:blue", "alpha": 0.5, "zorder": 1000, "label": "NN+FF"},
    "neural_neural": {"color": "tab:orange", "alpha": 0.5, "zorder": 900, "label": "NN+NN"},
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

# ==========================================================================
# Viz helpers
# ==========================================================================


def pretty_matplotlib():

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    plt.rc('font', family='serif', size=28)
    plt.rc('axes', linewidth=1.5, labelsize=32)
    plt.rc('xtick', labelsize=24, direction="in")
    plt.rc('ytick', labelsize=24, direction="in")

    # tick settings
    plt.rc('xtick.major', size=10, pad=4)
    plt.rc('xtick.minor', size=5, pad=4)
    plt.rc('ytick.major', size=10)
    plt.rc('ytick.minor', size=5)


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

# randomness
key = jrn.PRNGKey(seed)
model_key, generator_key = jax.random.split(key, 2)

generator, structure = build_data_objects(config)
mesh = build_mesh(generator_params, grid_params)
print(mesh)

# ===============================================================================
# Load models
# ===============================================================================

print("\nLoading models")

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

print("\nGenerating data batch")
xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))

# ===============================================================================
# Predictions
# ===============================================================================

models_data = {
    "neural_formfinding": {"models": (neural_formfinder, )},
    "neural_neural": {"models": (neural_neural, neural_formfinder)}
}

# make (batched) predictions
print("\nPredicting")

statistics_data = {
    "deltas": {},
    "residuals": {}
}

for name, data in models_data.items():

    print("name", name)
    # fetch model
    models = data["models"]

    # statistics
    deltas_batch = []
    residuals_batch = []

    for i in range(batch_size):

        # target
        xyz_target = xyz_batch[i, :]

        mesh_target = mesh.copy()
        _xyz = jnp.reshape(xyz_target, (-1, 3)).tolist()
        for idx, key in mesh.index_key().items():
            mesh_target.vertex_attributes(key, "xyz", _xyz[idx])

        if isinstance(models[0].decoder, ForceDensityModel):
            # Predict with autoencoder
            eqstate, fd_params = predict_neural_formfinding(models[0], xyz_target)
            example_loss, _ = compute_loss_autoencoder(models[0], structure, xyz_target[None, :])
            if NORMALIZE:
                example_loss = example_loss * (1.0 / BBOX_DIAG)
            print(f"Neural form-finder loss: {example_loss:.6f}")

        elif isinstance(models[0].decoder, PiggyDecoder):
            # Predict with autoencoder
            eqstate, fd_params = predict_neural_neural(models, xyz_target)
            _, fd_data = compute_loss_autoencoder(models[1], structure, xyz_target[None, :])
            example_loss = compute_loss_piggybacker(models[0].decoder, structure, fd_data)
            if NORMALIZE:
                example_loss = example_loss * (1.0 / BBOX_DIAG)
            print(f"Neural-neural loss: {example_loss:.6f}")

        # TODO: avoid creation of helper mesh to calculate stats
        mesh_hat = datastructure_updated(mesh, eqstate, fd_params)

        # statistics
        deltas = []
        residuals = []
        for vkey in mesh_hat.vertices():

            # distances
            _xyz_target = mesh_target.vertex_coordinates(vkey)
            _xyz_hat = mesh_hat.vertex_coordinates(vkey)
            delta = distance_point_point(_xyz_target, _xyz_hat)

            if NORMALIZE:
                delta = delta * (1.0 / BBOX_DIAG)
            if SCALE_UP:
                delta = delta * SCALE

            deltas.append(delta)

            # residuals
            if mesh_hat.is_vertex_on_boundary(vkey):  # skip supports
                residual = 0.0
            else:
                residual_vector = mesh_hat.vertex_attributes(vkey, ["rx", "ry", "rz"])
                residual = length_vector(residual_vector)

                if NORMALIZE:
                    residual = residual * (1.0 / RHO)
            residuals.append(residual)

        # store representative statistic
        deltas_batch.append(sum(deltas) / len(deltas))  # mean
        residuals_batch.append(max(residuals))  # maximum

        # deltas_batch.extend(deltas)
        # residuals_batch.extend(residuals)

    statistics_data["deltas"][name] = deltas_batch
    statistics_data["residuals"][name] = residuals_batch


# ===============================================================================
# Plot histogram
# ===============================================================================

if PLOT:
    print("\nPlotting histograms")

    grid_kwargs = {
        'which': 'major',
        'axis': 'y',
        'color': 'lightgray',
        'linestyle': 'dotted',
        'linewidth': 0.75
    }

    xlabels = {
        "deltas": r"Mean distance, $\delta_{\text{avg}}\,[\times 10^{-3}]$",
        "residuals": r"Max residual, $||\mathbf{r}_{\text{max}}||$",
    }

    pretty_matplotlib()

    for statistic_name, models_data in statistics_data.items():

        print(f"\n{statistic_name=}")

        statistic_all = [stat for slist in models_data.values() for stat in slist]
        bin_range = (min(statistic_all), max(statistic_all))
        print(f"{bin_range=}")

        fig, ax = plt.subplots(
            1, 1,
            dpi=PLOT_CONFIG["dpi"],
            figsize=PLOT_CONFIG["figsize"],   # 9, 7.5
            # sharex=True
        )

        _ax = ax
        _ax.set_ylabel("Ratio")
        _ax.grid(**grid_kwargs)

        bars_visited = set()
        for model_name, data in models_data.items():

            print(f"{model_name=}")

            config = MODELS_PLOT_CONFIG[model_name]

            num_bins = PLOT_CONFIG["num_bins"]
            bin_min, bin_max = bin_range
            bin_width = bin_max - bin_min
            bin_width = bin_width / num_bins

            bars = sns.histplot(
                data=data,
                bins=num_bins,
                binrange=bin_range,
                ax=_ax,
                kde=False,
                legend=True,
                element="bars",
                common_bins=True,
                multiple="layer",  # layer, stack
                stat="proportion",  # count, proportion
                color=config["color"],
                label=config["label"],
                alpha=config["alpha"],
                zorder=config["zorder"],
                )

            # Hatch
            if model_name == "neural_formfinding":
                for thisbar in bars.patches:
                    thisbar.set_hatch("\\")

            bar_heights = []
            for i, thisbar in enumerate(bars.patches):
                height = thisbar.get_height()
                if height in bars_visited:
                    continue
                bar_heights.append(height)
                bars_visited.add(height)

            for i, height in enumerate(bar_heights):
                print(f"Bar {i} height: {height}")

        # legend
        _ax.legend(fontsize="large", loc="upper right")

        # y axis
        _ax.set_ylim(-0.05, 1.05)

        # x axis
        _ax.tick_params(
            axis='x',
            which='both',
            bottom=True
        )

        xticks = np.arange(bin_min-bin_width, bin_max+bin_width, bin_width)
        if len(xticks) == num_bins + 3:
            xticks = xticks[1:-1]
        elif len(xticks) == num_bins + 2:
            xticks = xticks[1:]

        xticks_labels = [f"{tick:.2f}" for tick in xticks]
        plt.xticks(xticks, xticks_labels)

        plt.xlabel(xlabels[statistic_name])

        # layour
        plt.tight_layout(pad=0.5)

        if SAVE:
            ext = PLOT_CONFIG['extension']
            plt.savefig(f"histogram_{statistic_name}.{ext}",
                        bbox_inches="tight",
                        pad_inches=0.05,
                        transparent=False)

        plt.show()
