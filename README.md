# Run Note
- download pretrained models to `data/`, then

```
cd scripts
python visualize.py formfinder bezier
```


# Original README
---

# Real-time design of architectural simulations with differentiable mechanics and neural networks

[![arXiv](https://img.shields.io/badge/arXiv-2409.02606-b31b1b.svg)](https://arxiv.org/abs/2409.02606)

> Code for the [paper](https://arxiv.org/abs/2409.02606) published at ICLR 2025.

![Our trained model, deployed in Rhino3D](masonry_vault_cad_design.gif)

## Abstract
Designing mechanically efficient geometry for architectural structures like shells, towers, and bridges, is an expensive iterative process. Existing techniques for solving such inverse problems rely on traditional optimization methods, which are slow and computationally expensive, limiting iteration speed and design exploration. Neural networks would seem to offer a solution via data-driven amortized optimization, but they often require extensive fine-tuning and cannot ensure that important design criteria, such as mechanical integrity, are met. 

In this work, we combine neural networks with a differentiable mechanics simulator to develop a model that accelerates the solution of shape approximation problems for architectural structures represented as bar systems. This model explicitly guarantees compliance with mechanical constraints while generating designs that closely match target geometries. We validate our approach in two tasks, the design of masonry shells and cable-net towers. Our model achieves better accuracy and generalization than fully neural alternatives, and comparable accuracy to direct optimization but in real time, enabling fast and reliable design exploration. We further demonstrate its advantages by integrating it into 3D modeling software and fabricating a physical prototype. Our work opens up new opportunities for accelerated mechanical design enhanced by neural networks for the built environment.

## Table of Contents

- [Pretrained models](#pretrained-models)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Data generation](#data-generation)
  - [Building a model](#building-a-model)
  - [Loss and optimizer](#loss-and-optimizer)
- [Training](#training)
- [Testing](#testing)
- [Visualization](#visualization)
- [Direct optimization](#direct-optimization)
- [Predict then optimize](#predict-then-optimize)
- [Citation](#citation)
- [Contact](#contact)

## Pretrained models

Wan't to skip training? We got you 💪🏽
Our trained model weights are publicly available at this [link](https://drive.google.com/drive/folders/1BL_g5ikNh1s0fxsNp4PzKl84fQFUpm0L?usp=share_link).
Once downloaded, you can [test](#testing) the models at inference time and [display](#visualization) their predictions.

[Table of contents](#table-of-contents) ⬆️

## Repository structure

This repository contains two folders with the meat of our work: `src` and `scripts`.

The first folder, `src`, defines all the code infrastructure we need to build, train, serialize, and visualize our model and the baselines.
The second one, `scripts`, groups a list of routines to execute the code in `src`, and more importantly, to reproduce our experiments at inference time.

With the scripts, you can even tesselate and 3D print your own masonry vault from one of our model predictions if you fancy!

[Table of contents](#table-of-contents) ⬆️

## Installation

>We only support installation on a CPU. Our paper does not use any GPUs. Crazy, right? 🪄

Create a new [Anaconda](https://www.anaconda.com/) environment with Python 3.0 and then activate it:

```bash
conda create -n neural python=3.9
conda activate neural
```

### Basic Installation

1. Install the required Conda dependencies:
```bash
conda install -c conda-forge compas==1.17.10 compas_view2==0.7.0
```

2. Install the package and its pip dependencies:
```bash
pip install -e .
```
The `-e .` flag installs the package in "editable" mode, which means changes to the source code take effect immediately without reinstalling.

### Advanced Installation

If you need additional development tools (testing, formatting, etc.), are interested in making data plots, or want to generate bricks for a shell, follow these steps:

1. Install the necessary and additional Conda dependencies:
```bash
conda install -c conda-forge compas==1.17.10 compas_view2==0.7.0 compas_cgal==0.5.0
```

2. Install the package with development dependencies:
```bash
pip install -e ".[dev]"
```
[Table of contents](#table-of-contents) ⬆️

## Configuration

Our work focuses on two structural design tasks: compression-only shells and cablenet towers.

We thus create a `.yml` file with all the configuration hyperparameters per task.
The files are stored in the `scripts` folder as:
- `bezier.yml`, and
- `tower.yml` 

for the first and the second task, respectively.
The hyperparameters exposed in the configuration files range from choosing a data generator, prescribing the model architecture, and the optimization scheme.
We'll be mingling with them to steer the wheel while we run experiments.


### Data generation

An advantage of our work is that we only need to define target shapes alone, without a vector of force densities to be paired as ground-truth labels.
That would be the case in a fully supervised setting, which is not the case here.
Our model figures these labels out automatically.
This allows us to generate a dataset of target shapes on the fly at train time by specifying a `generator` configuration and a random `seed` to create pseudo-random keys.

#### Shells
The target shell shapes are parametrized by a square Bezier patch.

- `name`: The name of the generator to instantiate. One of `bezier_symmetric_double`, `bezier_symmetric`, `bezier_asymmetric`, and `bezier_lerp`. The first two options constraint the shapes to be symmetric along two or one axis, respectively. The third option does not enforce symmetry. The last option, `bezier_lerp` is used to interpolate linearnly a batch of doubly-symmetric and asymmetric shapes (i.e., shapes generated by `bezier_symmetric_double` and `bezier_asymmetric`).
- `num_points`: the number of control points on one side of the grid that parametrizes a Bezier patch.
- `bounds`: It specifies how to wiggle the control points of the patch on a `num_points x num_points` grid. The option `pillow` only moves the internal control point up and down, while `dome` additionally jitters the two control points on the boundary in and out. `saddle` is an extension of `dome` in that it lets one of the control points on the boundary move up and down too.
- `num_uv`: The number of spans to evaluate on the Bezier along the *u* and *v* directions. A value of `10`, for example, would result in a `10x10` grid of target points. These are the points to be matched during training.
- `size`: The length of the sides of the patch. It defines the scale of the task.
- `lerp_factor`: A scalar factor in [0, 1] to interpolate between two target surfaces. Only employed for `bezier_lerp`.

#### Towers
The target tower shapes are described in turn by a vertical sequence of planar circles.
The tower rings are deformed and rotated depending on the generator `name` and `bounds`.

- `name`: The generator name. Use `tower_ellipse` to make target shapes with elliptical rings, and `tower_circles` to keep the rings as circles.
- `bounds`: Either `straight` or `twisted`. The former only scales the rings on the plane at random. The latter scales and rotates the rings at random.
- `height`: The tower height.
- `radius`: The start radius of the all the generated circles.
- `num_sides`: The number of segments to discretize each circle with.
- `num_levels`: The number of circles to create along the tower's height. Equidistantly spaced.
- `num_rings`: The number of circles to be morphed during training. Must be `>2` since two of these rings are, by default, at the top and bottom of the tower.

### Building a model

We specify the architecture of a model in the configuration file, which for the most part, ressembles an autoencoder.
The configuration scheme is the same for any task.

#### Neural networks

Our experiments use multilayer perceptrons (MLP) for the encoder that maps shapes to simulation parameters, although we are by no means restricted to that.
An MLP too serves as a decoder for our fully neural baselines.
We employ one of the simplest possible neural networks, the MLP, to quantify the benefits of having a physics simulator in a neural network in large-scale mechanical design tasks.
This sets a baseline from which we can build upon with beefier architectures like graph neural networks, transformers, and beyond.

The encoder hyperparameters are:
- `shift`: The lower bound shift in output of the last layer of the encoder. This is what we call `tau` in the [paper](https://arxiv.org/abs/2409.02606).
- `hidden_layer_size`: The width of every fully-connected hidden layer. We restrict the size to `256` in all the experiments.
- `hidden_layer_num`: The number of hidden layers, output layer included.
- `activation_fn_name`: The name of the activation function after each hidden layer. We typically resort to `elu`.
- `final_activation_fn_name`: The activation function name after the output layer. We use `softplus` to ensure a strictly positive output, as needed by the simulator decoder.

The neural decoder's setup mirrors the encoder's, except for the `include_params_xl` flag.
If set to `True`, then the decoder expects the latents and boundary conditions as inputs.
Otherwise, it only decodes the latents.
We fix this hyperparameter to `True` in the [paper](https://arxiv.org/abs/2409.02606).

#### Simulator

For the simulator, the force density method (FDM), we only have `load` as a hyperparameter, which sets the magnitude of a vertical **area** load applied to the structures in the direction of gravity (hello Isaac Newton! 🍎).

If this value is nonzero, then the model will convert the area load into point loads to be compatible with our physics simulator.

### Loss and Optimizer

The training setup is also defined in the configuration file of the task, including the `loss` function to optimize for, the `optimizer` that updates the model parameters, and the `training` schedule that pretty much allocates the compute budget.

The `loss` function is the sum of multiple terms, that for the most part are a shape loss and a physics loss, as we explain in the [paper](https://arxiv.org/abs/2409.02606).
We allow for more refined control on the scaling of each loss term in the file:
- `include`: Whether or not to include the loss term during training. If set to `False`, then the value of the loss term is not calculated, saving some computation resources. By default, `include=True`.
- `weight`: The scalar weight of the loss term used for callibrating model performance, called `kappa` in the [paper](https://arxiv.org/abs/2409.02606). It is particularly useful to tune the scale of the physics loss when training the PINN baseline. The `weight=1.0` by default.

The `optimizer` hyperparameters are:
- `name`: the name of the gradient-based optimizer. We currently support `adam` and `sgd` from the `optax` library, but only use `adam` in the [paper](https://arxiv.org/abs/2409.02606).
- `learning_rate`: The constant learning rate. The rate is fixed, we ommit schedulers - it is more elegant.
- `clip_norm`: The global norm for gradient clipping. If set to `0.0`, then gradient clipping is ignored.

And for the `training` routine:
- `steps`: The number of optimization steps for model training (i.e., the number of times the model parameters are updated). We mostly train the models for `10000` steps.
- `batch_size`: The batch size of the input data.

[Table of contents](#table-of-contents) ⬆️

## Training

After setting up the config files, now it's time to make that CPU go brrrrr.
Execute the `train.py` script from your terminal:

```bash
python train.py <model_name> <task_name>
```

Where `task_name` is either `bezier` for the shells task or `tower` for the towers task.
Task-specific configuration details are given in the [paper](https://arxiv.org/abs/2409.02606).

The `model_name` is where things get interesting. 
In summary:

- Ours: `formfinder`
- NN and PINN baseline: `autoencoder`

If `autoencoder` is trained with the `residual` (i.e., the physics loss is included and active), this model will become a PINN baseline and will internally be renamed as `autoencoder_pinn` (sorry, naming is hard).

We invite you to check the docstring of the `train.py` script to see all the input options. 
They would allow you to warmstart the training from an existing pretrained model, checkpoint every so often, as well as plot and export the loss history for your inspection.

> A note on hyperparameter tuning. We utilized WandB to run hyperparameter sweeps. The sweeps are in turn handled by the `sweep.py` script in tandem with `sweep_bezier.yml` or `sweep_tower.yml` files, depending on the task. The structure of these sweep files mimics that of the configuration files described herein. We trust you'll be able to find your way around them if you really want to fiddle with them.

[Table of contents](#table-of-contents) ⬆️

## Testing

To evaluate the trained models on a test batch, run:

```bash
python predict.py <model_name> <task_name> --batch_size=<batch_size> --seed=<test_seed>
```
where we set to `--batch_size=100` during inference to match what we do in the [paper](https://arxiv.org/abs/2409.02606).
The test set is created by a generator that follows the same configuration as the train set, except for the random seed. 
We set `test_seed` to `90` in the `bezier` task and `test_seed` to `92` in the `tower` task.
Feel free to specify other seed values to test the model on different test datasets.

[Table of contents](#table-of-contents) ⬆️

## Visualization

An image is worth more than a thousand words, or in this case, more than a thousand numbers in a JAX array.

You can visualize the prediction a model makes, either ours or the baselines, with a dedicated script that lets you take control over the style of the rendered prediction:

```bash
python visualize.py <model_name> <task_name> --shape_index=<shape_index> --seed=<test_seed>
```

The shape to display is selected by inputting its index relative to the batch size with the `<shape_index>` argument.
Check out the docstring of `visualize.py` for the nitty-gritty details of how to control color palettes, linewidths, and arrow scales for making pretty pictures.

[Table of contents](#table-of-contents) ⬆️

## Direct optimization

So far we've only discussed how to create neural models for shape-matching tasks.
Direct gradient-based optimization is another baseline that merits its own section as it is the ground-truth in traditional design optimization in structural engineering.
Take an optimizer for a ride via:

```
python optimize.py <optimizer_name> <task_name> --batch_size=<batch_size> --seed=<test_seed> --blow=<blow> --bup=<bup> --param_init=<param_init> --maxiter=<maxiter> --tol=<tol>
```

We support two constrained gradient-based algorithms as implemented in `jaxopt`.
Select one of them through their `optimizer_name`: 
- `slsqp`: The sequential least squares quadratic programming algorithm.
- `lbfgsb`: The limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm.

The algorithms support box constraints on the simulation parameters.
We take advantage of this feature to constrain their value to a specific sign and to a range of reasonable values, depending on the task.
The lower box constraint is equivalent to the effect that `tau` has on the decoder's output in the [paper](https://arxiv.org/abs/2409.02606), by prescribing a minimum output value.
Both optimizers run for `maxiter=5000` iterations at most and stop early if they hit the convergence tolerance of `tol=1e-6`.

We're in the business of local optimization, so the inialization affects convergence. 
You can pick between two initialization schemes with `param_init` that respect the force density signs of a task (compression or tension):

- If specified as a scalar, it determines the starting constant value of all the simulation parameters.
- If set to `None`, the initialization samples starting parameters between `blow` and `bup` from a uniform distribution.
    
In the shells task, we apply `slsqp`, set `blow=0.0` and `bup=20.0`, and `param_init=None`.

In contrast, the towers task uses `lbfgsb` and `blow=1.0` to match the value of `tau` we used in this task in the paper.
The towers task is more nuanced because we explore three different initialization schemes:
- Randomized: `param_init=None`
- Expert: `param_init=1.0`

The third initialization type relies on the predictions of a pre-trained model and, to use it, we need to invoke a different script.
See [Predict then optimize](#predict-then-optimize) below.

[Table of contents](#table-of-contents) ⬆️

## Predict then optimize

There is enormous potential in combining neural networks with traditional optimization techniques to expedite mechanical design.
An opportunity in this space is to leverage the prediction made by one of our models and refine that prediction with direct optimization to unlock the best-performing designs.

Our key to open a tiny (very tiny) door into this space is the `predict_optimize.py` script:

```
python predict_optimize.py <model_name> <optimizer_name> <task_name> --batch_size=<batch_size> --seed=<test_seed> --blow=<blow> --bup=<bup> --maxiter=<maxiter> --tol=<tol>
```

What is different from the `optimize.py` script is that, now, you will have to specify the name of a trained model via `model_name`.
The predictions will warmstart the optimization, replacing any of the `param_init` schemes described earlier.
The rest of the inputs work the same way as in `optimize.py`.

[Table of contents](#table-of-contents) ⬆️

## Citation

Consider citing our [paper](https://arxiv.org/abs/2409.02606) if this work was helpful to your research.
Don't worry, it's free.

```bibtex
@inproceedings{
    pastrana_2025_diffmechanics,
    title={Real-time design of architectural structures with differentiable mechanics and neural networks},
    author={Rafael Pastrana and Eder Medina and Isabel M. de Oliveira and Sigrid Adriaenssens and Ryan P. Adams},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=Tpjq66xwTq}
}
```

[Table of contents](#table-of-contents) ⬆️

## Contact

Reach out! If you have questions or find bugs in our code, please open an issue on Github or email the authors at arpastrana@princeton.edu.

[Table of contents](#table-of-contents) ⬆️
