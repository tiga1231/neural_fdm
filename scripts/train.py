import os
import time
import yaml

from functools import partial

import jax
import jax.random as jrn
from jax import vmap

from neural_fofin import DATA

from neural_fofin.training import train_model

from neural_fofin.plotting import plot_losses

from neural_fofin.builders import build_loss_function
from neural_fofin.builders import build_data_generator
from neural_fofin.builders import build_connectivity_structure_from_generator
from neural_fofin.builders import build_neural_model
from neural_fofin.builders import build_optimizer

from neural_fofin.serialization import load_model
from neural_fofin.serialization import save_model as save_model_fn


# ===============================================================================
# Script function
# ===============================================================================

def train(
        model_name,
        task_name,
        from_pretrained=False,
        checkpoint_every=None,
        plot=True,
        save_model=True,
        save_losses=True,
):
    """
    Train a model to approximate a family of arbitrary shapes with mechanically-feasible geometries.

    Parameters
    ___________
    model_name: `str`
        The model name.
        Supported models are formfinder, autoencoder, and piggy.
    task_name: `str`
        The name of the YAML config file with the task hyperparameters.
    from_pretrained: `bool`
        If `True`, train the model starting from a pretrained version of it.
    checkpoint_every: `int`
        If not None, save a model every checkpoint steps.
    plot: `bool`
        If `True`, plot the loss curves.
    save: `bool`
        If `True`, save the trained model and the loss histories.
    """
    # load yaml file with hyperparameters
    with open(f"{task_name}.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # resolve model name for saving and checkpointing
    filename = f"{model_name}"
    loss_params = config["loss"]
    if loss_params["residual"]["include"] > 0 and model_name != "formfinder":
        filename += "_pinn"
    filename += f"_{task_name}"

    # pick callback
    callback = None
    if checkpoint_every:
        callback = partial(
            checkpoint_model,
            checkpoint_step=checkpoint_every,
            filename=filename
        )

    # train model
    trained_model, loss_history = train_model_from_config(
        model_name,
        config,
        from_pretrained,
        callback=callback
    )

    # define loss labels

    # plot loss curves
    if plot:
        print("\nPlotting")
        plot_losses(loss_history, labels=["loss"])
        # plot_gradient_norm(loss_history)
        # plot_latent_norm(loss_history)
        # plot_latent_mean_std(loss_history)
        # plot_stiffness_condition_num(loss_history)

    if save_model:
        print("\nSaving model")

        # save trained model
        filepath = os.path.join(DATA, f"{filename}_lyrical.eqx")
        save_model_fn(filepath, trained_model)
        print(f"Saved model to {filepath}")

    if save_losses:
        # save loss history
        labels = loss_history[0].keys()
        for label in labels:
            _label = "_".join(label.split())
            filename_loss = f"losses_{filename}_lyrical_{_label}.txt"

            filepath = os.path.join(DATA, filename_loss)
            with open(filepath, "w") as file:
                for values in loss_history:
                    _value = values[label].item()
                    file.write(f"{_value}\n")

            print(f"Saved loss history to {filepath}")


# ===============================================================================
# Train functions
# ===============================================================================

def train_model_from_config(model_name, config, pretrained=False, callback=None):
    """
    Train a model to approximate a family of arbitrary shapes with mechanically-feasible geometries.

    Parameters
    ___________
    model_name: `str`
        The model name.
        Supported models are formfinder, autoencoder, and piggy.
    config: `dict`
        A dictionary with the hyperparameters configuration.
    task_name: `str`
        The name of the YAML config file with the task hyperparameters.
    pretrained: `bool`
        If `True`, train the model starting from a pretrained version of it.
    callback: `Callable`
        A callback function to call at every train step.
    """
    # unpack parameters
    seed = config["seed"]
    training_params = config["training"]
    batch_size = training_params["batch_size"]
    steps = training_params["steps"]
    generator_name = config['generator']['name']
    bounds_name = config['generator']['bounds']

    # randomness
    key = jrn.PRNGKey(seed)
    model_key, generator_key = jax.random.split(key, 2)

    # create experiment
    print(f"\nTraining {model_name} on {generator_name} dataset with {bounds_name} bounds")
    generator = build_data_generator(config)
    structure = build_connectivity_structure_from_generator(config, generator)
    optimizer = build_optimizer(config)
    compute_loss = build_loss_function(config, generator)
    model = build_neural_model(model_name, config, generator, model_key)

    if pretrained:
        print("Starting from pretrained model")
        task_name = generator_name.split("_")[0]
        filepath = os.path.join(DATA, f"{model_name}_{task_name}_pretrain.eqx")
        model = load_model(filepath, model)

    # sample initial data batch
    xyz = vmap(generator)(jrn.split(generator_key, batch_size))

    # warmstart
    start_loss = compute_loss(model, structure, xyz)
    print(f"The structure has {structure.num_vertices} vertices and {structure.num_edges} edges")
    print(f"{model_name.capitalize()} start loss: {start_loss:.6f}")

    # train models
    print("\nTraining")
    start = time.perf_counter()
    train_data = train_model(
        model,
        structure,
        optimizer,
        generator,
        loss_fn=compute_loss,
        num_steps=steps,
        batch_size=batch_size,
        key=generator_key,
        callback=callback
        )
    end = time.perf_counter()

    print("\nTraining completed")
    print(f"Training time: {end - start:.4f} s")

    trained_model, _ = train_data

    end_loss = compute_loss(trained_model, structure, xyz)
    print(f"{model_name.capitalize()} last loss: {end_loss}")

    return train_data


# ===============================================================================
# Helper functions
# ===============================================================================

def checkpoint_model(
        model,
        opt_state,
        loss_vals,
        step,
        checkpoint_step,
        filename
):
    """
    Checkpoint a model as callback.
    """
    if step > 0 and step % checkpoint_step == 0:
        filepath = os.path.join(DATA, f"{filename}_{step}.eqx")
        save_model(filepath, model)


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(train)
