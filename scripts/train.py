import os

import time

import yaml

import jax

from jax import vmap

import jax.random as jrn

from neural_fofin import DATA

from neural_fofin.training import train_model
from neural_fofin.training import compute_loss

from neural_fofin.plotting import plot_smoothed_losses

from neural_fofin.experiments import build_data_objects
from neural_fofin.experiments import build_neural_model
from neural_fofin.experiments import build_optimizer

from neural_fofin.serialization import save_model


# local script parameters
SAVE = True
MODEL_NAME = "autoencoder"  # formfinder, autoencoder

# load yaml file with hyperparameters
with open("config.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# unpack parameters
seed = config["seed"]
grid_params = config["grid"]
training_params = config["training"]
optimizer_params = config["optimizer"]
batch_size = training_params["batch_size"]
steps = training_params["steps"]
loss_params = config["loss"]

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
print(f"{loss_params=}")
print(f"{MODEL_NAME} start loss: {start_loss:.6f}")

# train models
print("\nTraining")

start = time.perf_counter()
train_data = train_model(
    model,
    structure,
    optimizer,
    generator,
    loss_params=loss_params,
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

end_loss = compute_loss(trained_model, structure, xyz)
print(f"{MODEL_NAME} last loss: {end_loss}")

# plot loss curves
print("\nPlotting")
loss_labels = ["loss", "shape", "residual"]

plot_smoothed_losses(loss_history,
                     window_size=50,
                     labels=loss_labels)

# save models
if SAVE:
    print("\nSaving results")

    _filename = MODEL_NAME
    if loss_params["residual"]["include"] > 0 and MODEL_NAME != "formfinder":
        _filename += "_pinn"

    # save trained model
    _filepath = os.path.join(DATA, f"{_filename}.eqx")
    save_model(_filepath, trained_model)
    print(f"Saved model to {_filepath}")

    # save losses
    for i, _label in enumerate(loss_labels):

        _filename_loss = f"losses_{_filename}_{_label}.txt"

        _filepath = os.path.join(DATA, _filename_loss)
        with open(_filepath, "w") as file:
            for values in loss_history:
                _value = values[i].item()
                file.write(f"{_value}\n")

        print(f"Saved loss history to {_filepath}")
