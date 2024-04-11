import os
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

from jax_fdm.datastructures import FDMesh

# ==========================================================================
# Parameters
# ==========================================================================

SAVE = False

name = "cone"
dir = "cone"

qtol = -1.1e-1
num_bins = 10

# ==========================================================================
# Helpers
# ==========================================================================


def pretty_matplotlib():

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('font', family='serif', size=30)  # 24
    plt.rc('axes', linewidth=1.5, labelsize=22)  # 22
    plt.rc('xtick', labelsize=24, direction="in")
    plt.rc('ytick', labelsize=24, direction="in")

    # tick settings
    plt.rc('xtick.major', size=10, pad=4)
    plt.rc('xtick.minor', size=5, pad=4)
    plt.rc('ytick.major', size=10)
    plt.rc('ytick.minor', size=5)


# ==========================================================================
# Import mesh
# ==========================================================================

HERE = os.path.dirname(__file__)

grid_kwargs = {'which': 'major',
               'axis': 'y',
               'color': 'lightgray',
               'linestyle': 'dotted',
               'linewidth': 0.75}


pretty_matplotlib()

fig, ax = plt.subplots(1, 1,
                       figsize=(8, 5),
                       sharex=True)

color_pink = (255, 123, 171)
color_pink = [i / 255.0 for i in color_pink]

configs = {
    "optimized": {"color": color_pink, "alpha": 0.5, "zorder": 1000},
    "initial": {"color": "tab:blue", "alpha": 0.5, "zorder": 900},
}

# get binrange
filename = f"{dir}/{name}_optimized.json"
FILE_IN = os.path.join(HERE, filename)
mesh = FDMesh.from_json(FILE_IN)
q = mesh.edges_forcedensities()
q = [qval for qval in q if qval <= qtol]
binrange = (min(q), max(q))

experiments = defaultdict(dict)
for i, (subname, config) in enumerate(configs.items()):

    # load mesh
    filename = f"{dir}/{name}_{subname}.json"
    FILE_IN = os.path.join(HERE, filename)
    mesh = FDMesh.from_json(FILE_IN)
    mesh.print_stats()

    # get force densities
    q = mesh.edges_forcedensities()
    q = [qval for qval in q if qval <= qtol]

    # _ax = ax[i]
    _ax = ax
    _ax.set_ylabel("Ratio")
    _ax.grid(**grid_kwargs)

    bars = sns.histplot(
        data=q,
        bins=num_bins,
        binrange=binrange,
        ax=_ax,
        kde=False,
        legend=True,
        element="bars",
        common_bins=True,
        multiple="layer",  # layer, stack
        stat="proportion",
        color=config["color"],
        label=subname.capitalize(),
        alpha=config["alpha"],
        zorder=config["zorder"],
        )

    # Hatch
    if subname == "optimized":
        for i, thisbar in enumerate(bars.patches):
            thisbar.set_hatch("\\")

    _ax.set_ylim(0.0, 1.05)
    _ax.legend(fontsize="small", loc="upper left")
    _ax.tick_params(axis='x', which='both', bottom=False)

plt.xlabel("Force density, $q$")
plt.tight_layout(pad=0.5)
if SAVE:
    plt.savefig(f"{name}_histogram.pdf", bbox_inches=0.0, transparent=True)
plt.show()