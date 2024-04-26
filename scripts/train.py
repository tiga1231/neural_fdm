import os
import time
import yaml

import jax
from jax import vmap
import jax.random as jrn

from neural_fofin import DATA

from neural_fofin.training import train_model

from neural_fofin.losses import compute_loss

from neural_fofin.plotting import plot_smoothed_losses

from neural_fofin.builders import build_data_generator
from neural_fofin.builders import build_connectivity_structure_from_generator
from neural_fofin.builders import build_neural_model
from neural_fofin.builders import build_optimizer

from neural_fofin.serialization import save_model


# ===============================================================================
# Script function
# ===============================================================================

def train(model, save=True, config="config"):
    """
    Train a model to approximate a family of arbitrary shapes with mechanically-feasible geometries.

    Parameters
    ___________
    model: `str`
        The model name.
        Supported models are formfinder, autoencoder, and piggy.
    save: `bool`
        If `True`, save the trained model and the loss histories.
    config: `str`
        The filepath (without extension) of the YAML config file with the training hyperparameters.
    """
    MODEL_NAME = model
    SAVE = save
    CONFIG_NAME = config

    # load yaml file with hyperparameters
    with open(f"{CONFIG_NAME}.yml") as file:
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
    structure = build_connectivity_structure_from_generator(generator)
    optimizer = build_optimizer(config)
    model = build_neural_model(MODEL_NAME, config, generator, model_key)
    print(f"\nTraining {MODEL_NAME}")

    # sample initial data batch
    xyz = vmap(generator)(jrn.split(generator_key, batch_size))

    # warmstart
    start_loss = compute_loss(model, structure, xyz, loss_params)
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
    print("\nTraining completed")
    print(f"Training time: {end - start:.4f} s")

    end_loss = compute_loss(trained_model, structure, xyz, loss_params)
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

        _filepath = os.path.join(DATA, f"{_filename}.eqx")
        save_model(_filepath, trained_model)
        print(f"Saved model to {_filepath}")

        for i, _label in enumerate(loss_labels):
            _filename_loss = f"losses_{_filename}_{_label}.txt"

            _filepath = os.path.join(DATA, _filename_loss)
            with open(_filepath, "w") as file:
                for values in loss_history:
                    _value = values[i].item()
                    file.write(f"{_value}\n")

            print(f"Saved loss history to {_filepath}")


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(train)
