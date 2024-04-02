import jax

import optax

import jax.numpy as jnp

import equinox as eqx

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumMeshStructure

from neural_fofin.generator import PointGrid
from neural_fofin.generator import BezierSurfacePointGenerator

from neural_fofin.mesh import create_mesh_from_grid_simple

from neural_fofin.models import AutoEncoder
from neural_fofin.models import ForceDensityModel
from neural_fofin.models import ForceDensityWithShapeBasedLoads
from neural_fofin.models import PiggyDecoder


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


def get_generator_minmax_values(name):
    experiments = {
        "pillow": pillow_minmax_values,
        "dome": dome_minmax_values,
        "saddle": saddle_minmax_values
    }

    values_fn = experiments.get(name)
    if not values_fn:
        raise KeyError(f"Experiment name: {name} is currently unsupported!")

    return values_fn


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


def get_optimizer_fn(name):
    optimizers = {
        "adam": optax.adam,
        "sgd": optax.sgd,
    }

    optimizer_fn = optimizers.get(name)
    if not optimizer_fn:
        raise KeyError(f"Optimize name: {name} is currently unsupported!")

    return optimizer_fn


def build_optimizer(hyperparams):
    """
    """
    name = hyperparams["name"]
    learning_rate = hyperparams["learning_rate"]
    assert isinstance(learning_rate, float)

    optimizer_fn = get_optimizer_fn(name)
    optimizer = optimizer_fn(learning_rate=learning_rate)

    return optimizer


def get_fd_solver_fn(name):
    """
    """
    solvers = {
        "constant": ForceDensityModel,
        "shape_based_loads": ForceDensityWithShapeBasedLoads,
    }

    solver_fn = solvers.get(name)
    if not solver_fn:
        raise KeyError(f"FD model name: {name} is currently unsupported!")

    return solver_fn


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


def calculate_fd_loads(mesh, name, load):
    """
    """
    # Get loads
    num_loads = mesh.number_of_vertices()

    if name == "shape_based_loads":
        num_loads = mesh.number_of_faces()

    loads = [load] * num_loads

    return jnp.array(loads)


def build_fd_model():
    """
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


def build_fd_decoder(mesh, hyperparams):
    """
    """
    # unpack hyperparams
    name = hyperparams["name"]
    load = hyperparams["load"]

    # create FD model
    fd_model = build_fd_model()

    # calculate initial loads
    # loads = calculate_fd_loads(mesh, name, load)

    # get mask of supported edges
    mask_edges = calculate_edges_mask(mesh)

    # instantiate FD decoder
    fd_solver = get_fd_solver_fn(name)
    decoder = fd_solver(
        fd_model,
        load,
        mask_edges
        )

    return decoder


def build_neural_encoder(mesh, key, hyperparams):
    """
    """    
    # unpack hyperparameters
    hidden_layer_size = hyperparams["hidden_layer_size"]
    hidden_layer_num = hyperparams["hidden_layer_num"]
    activation_name = hyperparams["activation_fn_name"]
    final_activation_name = hyperparams["final_activation_fn_name"]

    # mesh quantities
    num_vertices = mesh.number_of_vertices()
    num_edges = mesh.number_of_edges()

    # instantiate MLP
    encoder = eqx.nn.MLP(
        in_size=num_vertices * 3,
        out_size=num_edges,
        width_size=hidden_layer_size,
        depth=hidden_layer_num,
        activation=get_activation_fn(activation_name),
        final_activation=get_activation_fn(final_activation_name),  # NOTE: needs softplus to ensure positive encoder output
        key=key    
        )
    
    return encoder


def build_neural_formfinder(mesh, key, hyperparams):
    """
    """
    # Unpack hyperparams
    nn_hyperparams, fd_hyperparams = hyperparams

    # Create MLP encoder
    encoder = build_neural_encoder(mesh, key, nn_hyperparams)
    
    # Build FD decoder
    decoder = build_fd_decoder(mesh, fd_hyperparams)

    # Assemble autoencoder
    model = AutoEncoder(encoder, decoder)

    return model


def build_piggy_decoder(mesh, key, hyperparams):
    """
    """    
    # unpack hyperparameters
    hidden_layer_size = hyperparams["hidden_layer_size"]
    hidden_layer_num = hyperparams["hidden_layer_num"]
    activation_name = hyperparams["activation_fn_name"]

    # mesh quantities
    num_edges = mesh.number_of_edges()
    num_vertices_free = len(list(mesh.vertices_free()))

    # get mask of supported edges
    mask_edges = calculate_edges_mask(mesh)

    # instantiate MLP
    decoder = PiggyDecoder(        
        mask_edges=mask_edges,
        in_size=num_edges,
        out_size=num_vertices_free * 3,
        width_size=hidden_layer_size,
        depth=hidden_layer_num,
        activation=get_activation_fn(activation_name),
        key=key
        )

    return decoder


def build_point_grid(hyperparams):
    """
    """
    size = hyperparams["size"]
    num_pts = hyperparams["num_points"]
    indices = hyperparams["indices"]

    assert num_pts == 4, "Only 4x4 grids are currently supported!"

    return PointGrid(size, num_pts, indices)


def _build_data_generator(grid, generator_hyperparams):
    """
    """
    # unpack parameters
    num_u = generator_hyperparams["num_uv"]
    num_v = generator_hyperparams["num_uv"]
    name = generator_hyperparams["name"]

    # wiggle bounds for task
    minmax_fn = get_generator_minmax_values(name)
    minval, maxval = minmax_fn()

    # array-ify
    minval = jnp.array(minval)
    maxval = jnp.array(maxval)

    # Create data generator
    u = jnp.linspace(0.0, 1.0, num_u)
    v = jnp.linspace(0.0, 1.0, num_v)

    # Create data generator
    generator = BezierSurfacePointGenerator(grid, u, v, minval, maxval)

    return generator


def build_data_generator(generator_hyperparams, grid_hyperparams):
    """
    """
    # build grid
    grid = build_point_grid(grid_hyperparams)

    # build data generator
    return _build_data_generator(grid, generator_hyperparams)


def build_mesh(generator_hyperparams, grid_hyperparams):
    """
    """
    # build grid
    grid = build_point_grid(grid_hyperparams)

    # unpack parameters
    num_u = generator_hyperparams["num_uv"]
    num_v = generator_hyperparams["num_uv"]

    # Create data generator
    u = jnp.linspace(0.0, 1.0, num_u)
    v = jnp.linspace(0.0, 1.0, num_v)

    # generate base FD Mesh
    return create_mesh_from_grid_simple(grid, u, v)


def build_structure(mesh):
    """
    """
    return EquilibriumMeshStructure.from_mesh(mesh)


def build_experiment(config, model_key):
    """
    """
    data = build_data_objects(config)
    neural = build_neural_objects(config, model_key)
    optimizer = build_optimization_object(config)

    return data, neural, optimizer


def build_experiment_coupled(config, model_key):
    """
    """
    data = build_data_objects(config)
    neural = build_neural_objects(config, model_key)
    optimizers = build_optimization_objects(config)

    return data, neural, optimizers


def build_data_objects(config):
    """
    """
    # unpack parameters
    grid_params = config["grid"]
    generator_params = config["generator"]

    # create data generator
    generator = build_data_generator(generator_params, grid_params)

    # generate base FD mesh
    mesh = build_mesh(generator_params, grid_params)
    structure = build_structure(mesh)

    return generator, structure


def build_neural_object(config, model_key):
    """
    """
    # unpack parameters
    grid_params = config["grid"]
    generator_params = config["generator"]

    fd_params = config["fdm"]

    encoder_params = config["encoder"]

    # generate base FD mesh
    mesh = build_mesh(generator_params, grid_params)

    # assemble autoencoder
    model = build_neural_formfinder(mesh, model_key, (encoder_params, fd_params))

    return model


def build_neural_objects(config, model_key):
    """
    """
    # unpack parameters
    grid_params = config["grid"]
    generator_params = config["generator"]

    fd_params = config["fdm"]

    encoder_params = config["encoder"]
    decoder_params = config["decoder"]

    # generate base FD mesh
    mesh = build_mesh(generator_params, grid_params)

    # assemble autoencoder
    model = build_neural_formfinder(mesh, model_key, (encoder_params, fd_params))

    # create MLP piggibacking decoder
    decoder = build_piggy_decoder(mesh, model_key, decoder_params)

    return model, decoder


def build_optimization_object(config):
    """
    """
    # unpack parameters
    optimizer_params = config["optimizer"]["encoder"]

    return build_optimizer(optimizer_params)


def build_optimization_objects(config):
    """
    """
    # unpack parameters
    encoder_params = config["optimizer"]["encoder"]
    optimizer = build_optimizer(encoder_params)

    decoder_params = config["optimizer"]["decoder"]
    optimizer_piggyback = build_optimizer(decoder_params)
    
    return optimizer, optimizer_piggyback
