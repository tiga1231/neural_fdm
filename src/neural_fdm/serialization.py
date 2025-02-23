import equinox as eqx


def save_model(filename, model):
    """
    Serialize and save a model.
    """
    with open(filename, "wb") as f:
        # hyperparam_str = json.dumps(hyperparams)
        # f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(filename, model_skeleton):
    """
    Load a serialized model.
    """
    with open(filename, "rb") as f:
        # hyperparams = json.loads(f.readline().decode())        
        return eqx.tree_deserialise_leaves(f, model_skeleton)
