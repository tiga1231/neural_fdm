import yaml

import jax
from jax import vmap

import jax.random as jrn

from experiments import build_experiment_coupled

from train_coupled import train_models
from train_coupled import compute_loss_autoencoder
from train_coupled import compute_loss_piggybacker

from plotting import plot_smoothed_losses

from serialization import save_model


# local script parameters
save = True

# load yaml file with hyperparameters
with open("./config.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    
# unpack parameters
seed = config["seed"]
grid_params = config["grid"]
training_params = config["training"]
batch_size = training_params["batch_size"]
steps = training_params["steps"]

# randomness
key = jrn.PRNGKey(seed)
model_key, generator_key = jax.random.split(key, 2)

# create data generator
experiment = build_experiment_coupled(config, model_key)

# unpack objects
data_objects, neural_models, optimizers = experiment
generator, structure = data_objects
model, piggy_decoder = neural_models
optimizer, optimizer_piggyback = optimizers

# sample initial data batch
xyz = vmap(generator)(jrn.split(generator_key, batch_size))

# warmstart
print("Warmstarting autoencoder")
start_loss, fd_data = compute_loss_autoencoder(model, structure, xyz)
print(f"Start loss: {start_loss:.6f}")

print("Warmstarting piggybacker")    
start_loss = compute_loss_piggybacker(piggy_decoder, structure, fd_data)
print(f"Start loss: {start_loss:.6f}")

# train models
print("Training")
models = (model, piggy_decoder)
optimizers = (optimizer, optimizer_piggyback)

train_data = train_models(
    models,
    structure,
    optimizers,
    generator,
    num_steps=steps,
    batch_size=batch_size,
    key=generator_key,
    callback=None
    )

trained_models, trained_opt_states, loss_history = train_data
trained_model, trained_piggy_decoder = trained_models
print(f"Last loss autoencoder: {loss_history[-1][0]:.6f}")
print(f"Last loss piggybacker: {loss_history[-1][1]:.6f}")

# plot loss curves
plot_smoothed_losses(loss_history, 100)

# save models
if save:
    for _model, _filename in zip(trained_models, ("autoencoder", "decoder")):
        _filepath = f"{_filename}.eqx" 
        save_model(_filepath, _model)
        print(f"\nSaved model to {_filepath}")
