import os
import numpy as np

HERE = os.path.dirname(__file__)
HOME = os.path.abspath(os.path.join(HERE, "../../"))
DATA = os.path.abspath(os.path.join(HOME, "data"))
FIGURES = os.path.abspath(os.path.join(HOME, "figures"))
SCRIPTS = os.path.abspath(os.path.join(HOME, "scripts"))

# Monkey patch numpy for compas_view2==0.7.0
if not hasattr(np, 'int'):
    np.int = np.int64  # noqa: F821