"""
Generate diagrams for the cablenet tower prediction task.
"""
import yaml

import jax
from jax import vmap
import jax.numpy as jnp

import jax.random as jrn

from compas.colors import Color
from compas.geometry import Polygon
from compas.geometry import Polyline
from compas.geometry import Plane
from compas.geometry import Frame

from jax_fdm.visualization import Viewer

from neural_fdm.builders import build_data_generator

from neural_fdm.generators import points_on_ellipse


# ===============================================================================
# Globals -- Don't do this at home!
# ===============================================================================

CAMERA_CONFIG_TOWER = {
    "color": (1.0, 1.0, 1.0, 1.0),
    "position": (10.718, 10.883, 14.159),
    "target": (-0.902, -0.873, 3.846),
    "distance": 19.482960680274577,
    "rotation": (1.013, 0.000, 2.362),
}


# ===============================================================================
# Script function
# ===============================================================================

def view_tower_task(seed=None, batch_size=None, slice=(0, -1)):
    """
    View the description of the cablenet tower prediction task on a batch of targets.

    Parameters
    ----------
    seed: `int` or `None`
        The random seed to generate a batch of target shapes.
        If `None`, it defaults to the task hyperparameters file.
    batch_size: `int` or `None`
        The size of the batch of target shapes.
        If `None`, it defaults to the task hyperparameters file.
    slice: `tuple`
        The start and stop indices of the slice of the batch for viewing.
        Default: `(0, -1)`, which means all shapes in the batch.
    """
    # slice
    START, STOP = slice
    
    # pick camera configuration for task
    task_name = "tower"
    CAMERA_CONFIG = CAMERA_CONFIG_TOWER
    _width = 450

    # load yaml file with hyperparameters
    with open(f"{task_name}.yml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # unpack parameters
    if seed is None:
        seed = config["seed"]
    if batch_size is None:
        training_params = config["training"]
        batch_size = training_params["batch_size"]

    # randomness
    key = jrn.PRNGKey(seed)
    _, generator_key = jax.random.split(key, 2)

    # create data generator
    generator = build_data_generator(config)

    # sample data batch
    xyz_batch = vmap(generator)(jrn.split(generator_key, batch_size))

    # view shapes in sequence
    print("\nViewing shapes in sequence")
    xyz_slice = xyz_batch[START:STOP]
    for i, xyz in enumerate(xyz_slice):

        # get shape from batch
        xyz = xyz[None, :]

        # create viewer
        viewer = Viewer(
            width=_width,
            height=900,
            show_grid=False,
            viewmode="lighted"
        )

        # modify view
        viewer.view.camera.position = CAMERA_CONFIG["position"]
        viewer.view.camera.target = CAMERA_CONFIG["target"]
        viewer.view.camera.distance = CAMERA_CONFIG["distance"]
        _rotation = CAMERA_CONFIG.get("rotation")
        if _rotation:
            viewer.view.camera.rotation = _rotation

        # draw rings
        rings = jnp.reshape(xyz, generator.shape_tube)[generator.levels_rings_comp, :, :]

        for ring in rings:

            ring = ring.tolist()
            polygon = Polygon(ring)
            
            viewer.add(polygon, opacity=0.5)
            viewer.add(
                Polyline(ring + ring[:1]),
                linewidth=4.0,
                color=Color.black().lightened()
            )

        # draw planes, transparent, thick-ish boundary
        heights = jnp.linspace(0.0, generator.height, generator.num_levels)        
        
        for i, height in enumerate(heights):

            origin_pt = [0.0, 0.0, height]
            plane = Plane(origin_pt, [0.0, 0.0, 1.0])
            

            circle = points_on_ellipse(
                generator.radius,
                generator.radius,
                height,
                generator.num_sides,
            )
            circle = circle.tolist()
            # circle = Polygon.from_sides_and_radius_xy(
            #         generator.num_sides,
            #         generator.radius,
            #     )
            # circle.transform(Translation.from_vector(origin_pt))
            # circle = circle.points

            if i in generator.levels_rings_comp:

                viewer.add(
                    Polyline(circle + circle[:1]),
                    linewidth=2.0,
                    color=Color.grey().lightened()
                )

                # viewer.add(Line(origin_pt, circle[0]), linewidth=2.0)
                # viewer.add(Line(origin_pt, rings[counter][0]), linewidth=2.0)
                # counter += 1
                continue

            size = 1.0
            viewer.add(
                plane,
                size=size,
                linewidth=0.1,
                color=Color.grey().lightened(10),
                opacity=0.1)

            # frame = Frame(origin_pt, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            # square = [frame.to_world_coordinates([-size, -size, 0]),
            #           frame.to_world_coordinates([size, -size, 0]),
            #           frame.to_world_coordinates([size, size, 0]),
            #           frame.to_world_coordinates([-size, size, 0])]

            # viewer.add(
            #     Polyline(square + square[:1]),
            #     linewidth=1.0,
            #     color=Color.black()  # .lightened()
            # )

        # show la crème
        viewer.show()


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(view_tower_task)
