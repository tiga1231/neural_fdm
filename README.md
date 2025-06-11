# Real-time design of architectural simulations with differentiable mechanics and neural networks

_Published at the International Conference on Learning Representations (ICLR) 2025_

![Our trained model, deployed in Rhino3D](masonry_vault_cad_design.gif)

Designing mechanically efficient geometry for architectural structures like shells, towers, and bridges, is an expensive iterative process. Existing techniques for solving such inverse problems rely on traditional optimization methods, which are slow and computationally expensive, limiting iteration speed and design exploration. Neural networks would seem to offer a solution via data-driven amortized optimization, but they often require extensive fine-tuning and cannot ensure that important design criteria, such as mechanical integrity, are met.

In this work, we combine neural networks with a differentiable mechanics simulator to develop a model that accelerates the solution of shape approximation problems for architectural structures represented as bar systems. This model explicitly guarantees compliance with mechanical constraints while generating designs that closely match target geometries. We validate our approach in two tasks, the design of masonry shells and cable-net towers. Our model achieves better accuracy and generalization than fully neural alternatives, and comparable accuracy to direct optimization but in real time, enabling fast and reliable design exploration. We further demonstrate its advantages by integrating it into 3D modeling software and fabricating a physical prototype.

Our work opens up new opportunities for accelerated mechanical design enhanced by neural networks for the built environment.

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


## Play

This repository contains two folders with the meat of our work: `src` and `scripts`.
The first folder, `src`, defines all the code infrastructure we need to build, train, serialize, and visualize our model and the baselines.
The second one, `scripts`, groups a list of routines to execute the code in `src`, and more importantly, to reproduce our experiments at inference time.
With the scripts, you can even tesselate and 3D print your own masonry vault from one of our model predictions if you fancy!

### Configuration files

Our work focuses on two structural design tasks: compression-only shells and cablenet towers.
We therefore create a `.yml` file with all the configuration hyperparameters per task.
The files are stored in the `scripts` folder as `bezier.yml` and `tower.yml` for the first and the second task, respectively.
The hyperparameters exposed in the configuration files range from choosing a data generator, prescribing the model architecture, and the optimization scheme.
We'll be mingling with them to steer the wheel while we run experiments.

### Data generation



### Building a model

We specify the model architecture in the configuration file.

### Training

We train our model and the baselines.
The training configuration 

### Testing

Blob.

### Direct optimization

Another baseline.

### Visualization

Blah.

## Citation

Consider citing our paper if this work was helpful to your research.
Don't worry, it's free.

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

## Contact

Reach out! If you have questions or find bugs in our code, please open an issue on Github or email the authors at arpastrana@princeton.edu. 