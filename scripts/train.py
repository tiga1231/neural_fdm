import time

import yaml

import jax

from jax import vmap

import jax.random as jrn

from neural_fofin.training import train_model
from neural_fofin.training import compute_loss

from neural_fofin.plotting import plot_smoothed_loss

from neural_fofin.experiments import build_data_objects
from neural_fofin.experiments import build_neural_model
from neural_fofin.experiments import build_optimizer

from neural_fofin.serialization import save_model


# local script parameters
SAVE = False
MODEL_NAME = "formfinder"  # formfinder, autoencoder

# load yaml file with hyperparameters
with open("config.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# unpack parameters
seed = config["seed"]
grid_params = config["grid"]
training_params = config["training"]
optimizer_params = config["optimizer"]["encoder"]
batch_size = training_params["batch_size"]
steps = training_params["steps"]

# randomness
key = jrn.PRNGKey(seed)
model_key, generator_key = jax.random.split(key, 2)

# create data generator
generator, structure = build_data_objects(config)
optimizer = build_optimizer(optimizer_params)
model = build_neural_model(MODEL_NAME, config, model_key)
print(model)

# sample initial data batch
xyz = vmap(generator)(jrn.split(generator_key, batch_size))

# warmstart
print("\nWarmstarting")
start_loss = compute_loss(model, structure, xyz)
print(f"{MODEL_NAME} start loss: {start_loss:.6f}")

# train models
print("\nTraining")

start = time.perf_counter()
train_data = train_model(
    model,
    structure,
    optimizer,
    generator,
    num_steps=steps,
    batch_size=batch_size,
    key=generator_key,
    callback=None
    )
end = time.perf_counter()

trained_model, trained_opt_states, loss_history = train_data
# trained_model, trained_piggy_decoder = trained_models
print("\nTraining completed")
print(f"Training time: {end:.4f} s")
# print(f"Autoencoder last loss: {loss_history[-1][0]:.6f}")
# print(f"Piggybacker last loss: {loss_history[-1][1]:.6f}")

# plot loss curves
print("\nPlotting")
plot_smoothed_loss(loss_history, 100)

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
