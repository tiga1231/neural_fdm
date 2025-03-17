# Real-time design of architectural simulations with differentiable mechanics and neural networks

_Published at the International Conference on Learning Representations (ICLR) 2025_

> This repository is under construction for public access. Please come back a tad bit later. In the meantime, feel free to peruse the repository files.

![Our trained model, deployed in Rhino3D](https://drive.google.com/file/d/1JIbSwx6XO27MxGIGCJlEZK64uYBEKA-1/view?usp=sharing)

## Abstract

Designing mechanically efficient geometry for architectural structures like shells, towers, and bridges, is an expensive iterative process. Existing techniques for solving such inverse problems rely on traditional optimization methods, which are slow and computationally expensive, limiting iteration speed and design exploration. Neural networks would seem to offer a solution via data-driven amortized optimization, but they often require extensive fine-tuning and cannot ensure that important design criteria, such as mechanical integrity, are met. In this work, we combine neural networks with a differentiable mechanics simulator to develop a model that accelerates the solution of shape approximation problems for architectural structures represented as bar systems. This model explicitly guarantees compliance with mechanical constraints while generating designs that closely match target geometries. We validate our approach in two tasks, the design of masonry shells and cable-net towers. Our model achieves better accuracy and generalization than fully neural alternatives, and comparable accuracy to direct optimization but in real time, enabling fast and reliable design exploration. We further demonstrate its advantages by integrating it into 3D modeling software and fabricating a physical prototype. Our work opens up new opportunities for accelerated mechanical design enhanced by neural networks for the built environment.

## Citation

```bibtex
@inproceedings{
    pastrana_2025_diffmechanics,
    title={Real-time design of architectural structures with differentiable mechanics and neural networks},
    author={Rafael Pastrana and Eder Medina and Isabel M. de Oliveira and Sigrid Adriaenssens and Ryan P Adams},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=Tpjq66xwTq}
}
```

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

Next, install COMPAS and COMPAS VIEW2 via `conda`. Please mind the version of these dependencies:

```bash
conda install -c conda-forge compas<2.0 compas_view2==0.7.0 
```

Finally, clone and install this repository from source:

```bash
git clone https://github.com/arpastrana/neural_fdm.git
cd neural_fdm
pip install -e .
```
Now, go ahead and play. Rock and roll 🎸! 
