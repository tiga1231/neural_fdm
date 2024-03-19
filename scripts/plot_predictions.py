import yaml

import jax
from jax import vmap

import jax.random as jrn

from jax_fdm.datastructures import FDNetwork

from jax_fdm.equilibrium import LoadState
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import datastructure_updated

from jax_fdm.visualization import Viewer

from experiments import build_data_objects
from experiments import build_neural_object
from experiments import build_mesh

from train_coupled import compute_loss_autoencoder

from serialization import load_model


# local script parameters
SAVE = False
NAME = "form_former"  # "autoencoder"
COLOR_SCHEME = "fd"
START = 0
STOP = 1
CAMERA_CONFIG = {
    "position": (30.34, 30.28, 42.94),
    "target": (0.956,0.727,1.287),
    "distance": 20.0, 
}


# load yaml file with hyperparameters
with open("./config.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    
# unpack parameters
seed = config["seed"]
grid_params = config["grid"]
generator_params = config["generator"]
batch_size = config["training"]["batch_size"]

# randomness
key = jrn.PRNGKey(seed)
model_key, generator_key = jax.random.split(key, 2)

# create data generator
generator, structure = build_data_objects(config)
mesh = build_mesh(generator_params, grid_params)
model_skeleton = build_neural_object(config, model_key)

# for load in loads_nodes:
for load in model_skeleton.decoder.loads:
    print(load)

# load model
model = load_model(f"{NAME}.eqx", model_skeleton)

print()
for load in model.decoder.loads:
    print(load)

for edge_mask in model.decoder.mask_edges:
    print(load)

raise

# sample data batch
xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))

# compute loss
# NOTE: maybe compute loss of sampled object, one by one?
start_loss, fd_data = compute_loss_autoencoder(model, structure, xyz_batch)
print(f"Autoencoder start loss: {start_loss:.6f}")

# make (batched) predictions
for i in range(start, stop):
    print(i)
    xyz = xyz_batch[i]

    xyz_hat = model(xyz, structure)

    q = model.encoder(xyz) * -1.0
    q_masked = q * model.decoder.mask_edges + model.decoder.qmin

    loads_nodes = model.decoder.get_point_loads(xyz, structure)
    load_state = LoadState(loads_nodes, 0.0, 0.0)

    # for load in loads_nodes:
    for load in model.decoder.loads:
        print(load)

    xyz_fixed = model.decoder.get_xyz_fixed(xyz, structure)
    
    fdm_params_hat = EquilibriumParametersState(q_masked,
                                                xyz_fixed,
                                                load_state)
    
    eqstate_hat = model.decoder.model(fdm_params_hat, structure)

    # eqstate_hat = model.decoder.model.equilibrium(
    #     q_hat_masked,
    #     xyz_fixed,
    #     loads,
    #     structure
    #     )
    
    mesh_hat = datastructure_updated(mesh, eqstate_hat, fdm_params_hat)
    network_hat = FDNetwork.from_mesh(mesh_hat)

    print("Yamanaka-ko!")
    raise

    # ==========================================================================
    # Visualization
    # ==========================================================================

    viewer = Viewer(
        width=1600,
        height=900,
        show_grid=False,
        viewmode="lighted")

    # modify view
    viewer.view.camera.position = CAMERA_CONFIG["position"]
    viewer.view.camera.target = CAMERA_CONFIG["target"]
    viewer.view.camera.distance = CAMERA_CONFIG["distance"]

    # optimized mesh
    viewer.add(network_hat,
               edgewidth=(0.1, 0.25),           
               edgecolor=COLOR_SCHEME,
               show_loads=True,
               loadscale=1.0,  # 5.0
               show_reactions=True,
               reactionscale=1.0)

    viewer.add(
        mesh_hat,
        show_points=False,
        show_edges=False,
        opacity=0.3
    )

    # reference network
    viewer.add(FDNetwork.from_mesh(mesh),
               as_wireframe=True,
               show_points=False,
               linewidth=1.0,
               # color=Color.grey().darkened()
               )

    # show le crème
    viewer.show()
