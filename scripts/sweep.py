import wandb

from collections.abc import Mapping

from neural_fofin.serialization import save_model

from train import train_model_from_config


# ===============================================================================
# Helper functions
# ===============================================================================

def update_dict(d, u):
    """
    Recursively update a dictionary (d) with another (u).

    The function assumes that the keys in u match those in d.
    Otherwise, the function will not work.

    Source: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth?page=1&tab=scoredesc#tab-top
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v

    return d


def log_to_wandb(model, opt_state, loss_vals, step):
    """
    Record metrics in weights and biases.
    """
    metrics = {}
    for key, value in loss_vals.items():
        metrics[key] = value.item()

    wandb.log(metrics)


# ===============================================================================
# Script function
# ===============================================================================

def sweep(**kwargs):
    """
    Sweep a model to find adequate hyper-parameters that best solve a design task on form-found geometries.
    """
    wandb.init()

    config = wandb.config
    MODEL_NAME = config.model
    TASK_NAME = config.generator["name"]

    # train model with wandb config
    train_data = train_model_from_config(
        MODEL_NAME,
        config,
        pretrained=False,
        callback=log_to_wandb
    )
    trained_model, _ = train_data

    # save trained model to local folder
    filename = MODEL_NAME
    loss_params = config["loss"]
    if loss_params["residual"]["include"] > 0 and MODEL_NAME != "formfinder":
        filename += "_pinn"
    filename += f"_{TASK_NAME}"

    filepath = f"{filename}.eqx"
    save_model(filepath, trained_model)

    # save trained model to wandb
    wandb.save(filepath)


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(sweep)
