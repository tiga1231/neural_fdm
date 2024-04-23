import time

import yaml

import jax

from jax import vmap

import jax.random as jrn

from neural_fofin.experiments import build_data_generator
from neural_fofin.experiments import build_connectivity_structure
from neural_fofin.experiments import build_neural_model
from neural_fofin.experiments import build_optimizer

from neural_fofin.training_coupled import train_models
from neural_fofin.training_coupled import compute_loss_autoencoder
from neural_fofin.training_coupled import compute_loss_piggybacker

from neural_fofin.plotting import plot_smoothed_losses

from neural_fofin.serialization import save_model


# local script parameters
SAVE = False
MODEL_NAMES = ("autoencoder", "decoder")

# load yaml file with hyperparameters
with open("config.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# unpack parameters
seed = config["seed"]
training_params = config["training"]
batch_size = training_params["batch_size"]
steps = training_params["steps"]
loss_params = config["loss"]

# randomness
key = jrn.PRNGKey(seed)
model_key, generator_key = jax.random.split(key, 2)

# create experiment
generator = build_data_generator(config)
structure = build_connectivity_structure(config)

# unpack objects
model, piggy_decoder = neural_models
optimizer, optimizer_piggyback = optimizers

# sample initial data batch
xyz = vmap(generator)(jrn.split(generator_key, batch_size))

# warmstart
print("\nWarmstarting")
start_loss, fd_data = compute_loss_autoencoder(model, structure, xyz)
print(f"Autoencoder start loss: {start_loss:.6f}")

start_loss = compute_loss_piggybacker(piggy_decoder, structure, fd_data)
print(f"Piggybacker start loss: {start_loss:.6f}")

# train models
print("\nTraining")
models = (model, piggy_decoder)
optimizers = (optimizer, optimizer_piggyback)

start = time.perf_counter()
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
end = time.perf_counter()

trained_models, trained_opt_states, loss_history = train_data
trained_model, trained_piggy_decoder = trained_models
print("\nTraining completed")
print(f"Training time: {end:.4f} s")
print(f"Autoencoder last loss: {loss_history[-1][0]:.6f}")
print(f"Piggybacker last loss: {loss_history[-1][1]:.6f}")

# plot loss curves
print("\nPlotting")
plot_smoothed_losses(loss_history, 100)

# save models
if SAVE:
    print("\nSaving results")
    # _filepath = "losses_coupled.txt"

    for i, _filename in enumerate(MODEL_NAMES):
        _filepath = f"losses_{_filename}.txt"

        with open(_filepath, "w") as file:
            for values in loss_history:
                _value = values[i].item()
                file.write(f"{_value}\n")

        print(f"Saved loss history to {_filepath}")

    for _model, _filename in zip(trained_models, MODEL_NAMES):
        _filepath = f"{_filename}.eqx"
        save_model(_filepath, _model)
        print(f"Saved model to {_filepath}")
