# Towards neural form-finding

Integrating form-finding simulations into neural networks

## Installation

Create a new [Anaconda](https://www.anaconda.com/) environment and then activate it:

```bash
conda create -n neural
conda activate neural
```

Install some dependencies from `pip`-land:

```bash
pip install --upgrade jax==0.4.23
pip install optax==0.1.5 equinox==0.11.3
pip install seaborn
```

Next, install COMPAS and COMPAS VIEW2 via `conda`. Mind the version of these dependencies:

```bash
conda install -c conda-forge compas<2.0 compas_view2==0.7.0 
```

Clone and install `jax_fdm` from source:

```bash
git clone https://github.com/arpastrana/jax_fdm.git
cd jax_fdm
pip install -e .
```

Finally, clone and install this repository from source:

```bash
git clone https://github.com/arpastrana/neural_fdm.git
cd neural_fdm
pip install -e .
```
Now, go ahead and play. Rock and roll 🎸! 
