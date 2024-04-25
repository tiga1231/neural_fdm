import jax

import optax

import jax.numpy as jnp

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumMeshStructure

from neural_fofin.generators import BezierSurfaceAsymmetricPointGenerator
from neural_fofin.generators import BezierSurfaceSymmetricPointGenerator
from neural_fofin.generators import BezierSurfaceSymmetricDoublePointGenerator

from neural_fofin.mesh import create_mesh_from_bezier

from neural_fofin.models import AutoEncoder
from neural_fofin.models import AutoEncoderPiggy
from neural_fofin.models import MLPEncoder
from neural_fofin.models import FDDecoder
from neural_fofin.models import MLPDecoder
from neural_fofin.models import MLPDecoderXL


# ===============================================================================
# Shape generator bounds
# ===============================================================================

def pillow_minmax_values():
    """
    The boundary XYZ values for the pillow tile.
    """
    minval = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    maxval = [
        [0.0, 0.0, 10.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    return minval, maxval


def dome_minmax_values():
    """
    The boundary XYZ values for the dome tile.
    """
    minval = [
        [0.0, 0.0, 1.0],
        [-5.0, 0.0, 0.0],
        [0.0, -5.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    maxval = [
        [0.0, 0.0, 10.0],
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    return minval, maxval


def saddle_minmax_values():
    """
    The boundary XYZ values for the dome tile.
    """
    minval = [
        [0.0, 0.0, 1.0],
        [-5.0, 0.0, 0.0],
        [0.0, -5.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    maxval = [
        [0.0, 0.0, 10.0],
        [5.0, 0.0, 10.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0]
        ]

    return minval, maxval


def get_generator_minmax_values(name, bounds):
    experiments = {
        "pillow": pillow_minmax_values,
        "dome": dome_minmax_values,
        "saddle": saddle_minmax_values
    }

    values_fn = experiments.get(bounds)
    if not values_fn:
        raise KeyError(f"Experiment bounds: {bounds} is currently unsupported!")

    # generate values on a quarter tile
    minval, maxval = values_fn()

    # concatenate bounds based on whether generator is symmetric or 2-symmetric
    name_parts = name.split("_")
    if "symmetric" in name_parts:
        if "double" not in name_parts:
            minval = minval + minval
            maxval = maxval + maxval
    elif "asymmetric" in name_parts:
        minval = minval + minval + minval + minval
        maxval = maxval + maxval + maxval + maxval

    # array-ify
    minval = jnp.array(minval)
    maxval = jnp.array(maxval)

    return minval, maxval


# ===============================================================================
# Data generators
# ===============================================================================


def build_bezier_point_generator(generator_params):
    """
    """
    # unpack parameters
    name = generator_params["name"]
    num_u = generator_params["num_uv"]
    num_v = generator_params["num_uv"]
    size = generator_params["size"]
    num_pts = generator_params["num_points"]
    bounds = generator_params["bounds"]

    # wiggle bounds for task
    minval, maxval = get_generator_minmax_values(name, bounds)

    # Create data generator
    u = jnp.linspace(0.0, 1.0, num_u)
    v = jnp.linspace(0.0, 1.0, num_v)

    # Create data generator
    generators = {
        "bezier_symmetric": BezierSurfaceSymmetricPointGenerator,
        "bezier_symmetric_double": BezierSurfaceSymmetricDoublePointGenerator,
        "bezier_asymmetric": BezierSurfaceAsymmetricPointGenerator
    }

    generator = generators.get(name)
    if not generator:
        raise ValueError(f"Generator {name} is not supported yet!")

    return generator(size, num_pts, u, v, minval, maxval)


def build_data_generator(config):
    """
    """
    # unpack parameters
    generator_params = config["generator"]

    # build bezier generator
    return build_bezier_point_generator(generator_params)


# ===============================================================================
# Mesh
# ===============================================================================


def build_mesh_from_generator(generator):
    """
    """
    # unpack parameters
    surface = generator.surface
    u = generator.u
    v = generator.v

    # generate base FD Mesh
    return create_mesh_from_bezier(surface, u, v)


# ===============================================================================
# Structure (Graph)
# ===============================================================================


def build_connectivity_structure_from_generator(generator):
    """
    """
    # generate base FD mesh
    mesh = build_mesh_from_generator(generator)

    return EquilibriumMeshStructure.from_mesh(mesh)


# ===============================================================================
# Activation functions
# ===============================================================================

def get_activation_fn(name):
    """
    """
    functions = {
        "elu": jax.nn.elu,
        "relu": jax.nn.relu,
        "softplus": jax.nn.softplus
    }

    activation_fn = functions.get(name)
    if not activation_fn:
        raise KeyError(f"Activation name: {name} is currently unsupported!")

    return activation_fn


# ===============================================================================
# Optimizers
# ===============================================================================

def get_optimizer_fn(name):
    optimizers = {
        "adam": optax.adam,
        "sgd": optax.sgd,
    }

    optimizer_fn = optimizers.get(name)
    if not optimizer_fn:
        raise KeyError(f"Optimize name: {name} is currently unsupported!")

    return optimizer_fn


def build_optimizer(config):
    """
    """
    hyperparams = config["optimizer"]

    name = hyperparams["name"]
    learning_rate = hyperparams["learning_rate"]
    assert isinstance(learning_rate, float)

    optimizer_fn = get_optimizer_fn(name)
    optimizer = optimizer_fn(learning_rate=learning_rate)

    return optimizer


# ===============================================================================
# Force density solver
# ===============================================================================

def build_fd_model():
    """
    Dense, because batching rule is undefined to vmap a sparse model.
    """
    fd_model = EquilibriumModel(
        tmax=1,
        eta=1e-6,
        is_load_local=False,
        itersolve_fn=None,
        implicit_diff=True,
        verbose=False
        )

    return fd_model


def calculate_edges_mask(mesh):
    """
    Mask out to indicate what mesh edges are fully supported.
    """
    mask_edges = []
    for edge in mesh.edges():
        mask_val = 1.0
        if mesh.is_edge_fully_supported(edge):
            mask_val = 0.0
        mask_edges.append(mask_val)

    return jnp.array(mask_edges, dtype=jnp.int64)


# ===============================================================================
# Decoders
# ===============================================================================

def build_fd_decoder(mesh, params):
    """
    """
    # unpack hyperparams
    load = params["load"]

    # create FD model
    fd_model = build_fd_model()

    # get mask of supported edges
    mask_edges = calculate_edges_mask(mesh)

    # instantiate FD decoder
    decoder = FDDecoder(
        fd_model,
        load,
        mask_edges
        )

    return decoder


def build_neural_decoder(mesh, key, params):
    """
    """
    # unpack hyperparameters
    nn_params, fd_params = params

    # get neural network params
    include_xl = nn_params["include_params_xl"]
    hidden_layer_size = nn_params["hidden_layer_size"]
    hidden_layer_num = nn_params["hidden_layer_num"]
    activation_name = nn_params["activation_fn_name"]

    # get load
    load = fd_params["load"]

    # mesh quantities
    num_vertices = mesh.number_of_vertices()
    num_edges = mesh.number_of_edges()
    num_vertices_free = len(list(mesh.vertices_free()))
    num_vertices_fixed = len(list(mesh.vertices_fixed()))

    # get mask of supported edges
    mask_edges = calculate_edges_mask(mesh)

    # define size of input layer
    in_size = num_edges
    decoder_cls = MLPDecoder

    if include_xl:
        in_size += num_vertices_fixed * 3
        in_size += num_vertices
        decoder_cls = MLPDecoderXL

    # instantiate MLP
    decoder = decoder_cls(
        load=load,
        mask_edges=mask_edges,
        in_size=in_size,
        out_size=num_vertices_free * 3,
        width_size=hidden_layer_size,
        depth=hidden_layer_num,
        activation=get_activation_fn(activation_name),
        key=key
        )

    return decoder


# ===============================================================================
# Encoders
# ===============================================================================

def build_neural_encoder(mesh, key, params):
    """
    """
    # unpack hyperparameters
    hidden_layer_size = params["hidden_layer_size"]
    hidden_layer_num = params["hidden_layer_num"]
    activation_name = params["activation_fn_name"]
    final_activation_name = params["final_activation_fn_name"]

    # mesh quantities
    num_vertices = mesh.number_of_vertices()
    num_edges = mesh.number_of_edges()

    # instantiate MLP
    encoder = MLPEncoder(
        in_size=num_vertices * 3,
        out_size=num_edges,
        width_size=hidden_layer_size,
        depth=hidden_layer_num,
        activation=get_activation_fn(activation_name),
        final_activation=get_activation_fn(final_activation_name),
        key=key
        )

    return encoder


# ===============================================================================
# Autoencoder models
# ===============================================================================

def build_neural_formfinder(mesh, key, params):
    """
    """
    # Unpack hyperparams
    nn_params, fd_params = params

    # Create MLP encoder
    encoder = build_neural_encoder(mesh, key, nn_params)

    # Build FD decoder
    decoder = build_fd_decoder(mesh, fd_params)

    # Assemble autoencoder
    model = AutoEncoder(encoder, decoder)

    return model


def build_neural_autoencoder(mesh, key, params):
    """
    """
    # Unpack hyperparams
    enc_params, dec_params = params

    # Create MLP encoder
    encoder = build_neural_encoder(mesh, key, enc_params)

    # Build MLP decoder
    decoder = build_neural_decoder(mesh, key, dec_params)

    # Assemble autoencoder
    model = AutoEncoder(encoder, decoder)

    return model


def build_neural_autoencoder_piggy(mesh, key, params):
    """
    """
    # Unpack hyperparams

    enc_params, dec_params, fd_params = params

    # Create MLP encoder
    encoder = build_neural_encoder(mesh, key, enc_params)

    # Build FD decoder
    decoder = build_fd_decoder(mesh, fd_params)

    # Build MLP decoder
    decoder_piggy = build_neural_decoder(mesh, key, dec_params)

    # Assemble autoencoder
    model = AutoEncoderPiggy(encoder, decoder, decoder_piggy)

    return model


def build_neural_model(name, config, generator, model_key):
    """
    """
    # generate base FD mesh
    mesh = build_mesh_from_generator(generator)

    # build model
    fd_params = config["fdm"]
    encoder_params = config["encoder"]
    decoder_params = config["decoder"]

    # select model
    if name == "formfinder":
        build_fn = build_neural_formfinder
        params = (encoder_params, fd_params)
    elif name == "autoencoder":
        build_fn = build_neural_autoencoder
        params = (encoder_params, (decoder_params, fd_params))
    elif name == "piggy":
        build_fn = build_neural_autoencoder_piggy
        params = (encoder_params, (decoder_params, fd_params), fd_params)
    else:
        raise ValueError(f"Model name {name} is unsupported")

    return build_fn(mesh, model_key, params)
