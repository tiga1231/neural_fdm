"""
Generate bricks on the faces of a mesh.
"""
import os
from random import randint

from compas.colors import Color

from compas.datastructures import Mesh
from compas.datastructures import mesh_dual
from compas.datastructures import mesh_thicken
from compas.datastructures import mesh_offset
from compas.datastructures import mesh_delete_duplicate_vertices

from compas.geometry import Sphere
from compas.geometry import Box
from compas.geometry import centroid_points
from compas.geometry import oriented_bounding_box_numpy
from compas.geometry import oriented_bounding_box_xy_numpy
from compas.geometry import Transformation
from compas.geometry import Scale
from compas.geometry import Frame
from compas.geometry import Plane
from compas.geometry import scale_vector
from compas.geometry import add_vectors
from compas.geometry import area_triangle
from compas.geometry import normal_triangle
from compas.geometry import circle_from_points

from compas.utilities import pairwise

from compas_cgal.booleans import boolean_union_mesh_mesh
from compas_cgal.booleans import boolean_difference_mesh_mesh
from compas_cgal.meshing import mesh_remesh

from jax_fdm.datastructures import FDMesh
from jax_fdm.visualization import Viewer

from neural_fofin import DATA

from text_2_mesh import text_2_mesh


# ===============================================================================
# Helper functions
# ===============================================================================

def triangulate_face_quad(face, reverse=False):
    """
    Triangulate a mesh quad face.
    """
    a, b, c, d = face
    if not reverse:
        face_a = [a, b, d]
        face_b = [b, c, d]
    else:
        face_a = [a, b, c]
        face_b = [c, d, a]

    return [face_a, face_b]


def triangulate_face_ngon(face, vertices):
    """
    Triangulate a mesh ngon face.
    """
    midpoint = centroid_points([vertices[vkey] for vkey in face])
    vertices.append(midpoint)
    ckey = len(vertices) - 1

    # create new faces
    new_faces = []
    for a, b in pairwise(face + face[:1]):
        _face = [a, b, ckey]
        new_faces.append(_face)

    return new_faces


def triangulate_face(face, vertices, reverse=False):
    """
    Triangulates a mesh face.
    The face is a list of indices pointing to a list with the vertices xyz coordinates.
    """
    assert len(face) > 2

    new_faces = []
    # triangle
    if len(face) == 3:
        new_faces = [face]
    # quad
    elif len(face) == 4:
        new_faces = triangulate_face_quad(face, reverse)
    # ngon
    else:
        new_faces = triangulate_face_ngon(face, vertices)

    return new_faces


def generate_bricks(mesh, thickness):
    """
    Generate a solid brick per mesh face.
    """
    half_thick = thickness / 2.0

    mesh_bottom = mesh_offset(mesh, half_thick)
    mesh_top = mesh_offset(mesh, half_thick * -1.0)

    bricks = {}
    halfedges_visited = set()

    for i, fkey in enumerate(mesh.faces()):

        # adding side faces
        faces = []

        num_vertices = len(mesh.face_vertices(fkey))
        face_bottom = list(range(num_vertices))
        face_top = list(range(num_vertices, 2 * num_vertices))

        xyz_bottom = mesh_bottom.face_coordinates(fkey)
        xyz_top = mesh_top.face_coordinates(fkey)
        vertices = xyz_bottom + xyz_top

        halfedges = mesh.face_halfedges(fkey)
        iterable = zip(
            pairwise(face_top + face_top[:1]),
            pairwise(face_bottom + face_bottom[:1]),
            halfedges
        )

        for edge_1, edge_2, halfedge in iterable:
            a, b = edge_1
            d, c = edge_2
            face = [a, b, c, d]
            # triangulate
            is_visited = halfedge in halfedges_visited
            tri_faces = triangulate_face_quad(face, is_visited)
            faces.extend(tri_faces)

        # face half edges
        halfedges_visited.update(halfedges)
        halfedges_reversed = [(v, u) for u, v in halfedges]
        halfedges_visited.update(halfedges_reversed)

        # triangulate top and bottom faces
        tri_faces = triangulate_face(face_bottom, vertices)
        faces.extend(tri_faces)

        tri_faces = triangulate_face(face_top, vertices)
        faces.extend([face[::-1] for face in tri_faces])

        # make mesh from face
        brick = Mesh.from_vertices_and_faces(vertices, faces)

        # store brick
        bricks[fkey] = brick

    return bricks, (mesh_bottom, mesh_top)


def add_text_engraving(fkey, bricks, mesh_bottom, text, depth=1):
    """
    Engrave the underside of a brick mesh with text.
    """
    # generate text mesh
    text_mesh = text_2_mesh(text)
    vertices, _ = text_mesh.to_vertices_and_faces()

    # calculate bounding box and properties
    bbox = oriented_bounding_box_numpy(vertices)
    bbox = Box.from_bounding_box(bbox)
    bbox_largest = max([bbox.width, bbox.depth, bbox.height])

    # find face in brick
    brick = bricks[fkey]

    # generate face plane
    vertices = mesh_bottom.face_coordinates(fkey)  # faces_topbottom[fkey][0]  # NOTE: or -1?
    face = list(range(len(vertices)))
    tri_faces = triangulate_face(face, vertices)

    # take face with largest area
    tri_faces = sorted(tri_faces, key=lambda x: area_triangle([vertices[v] for v in x]))
    big_face = tri_faces[-1]
    big_face_vertices = [vertices[v] for v in big_face]
    _, radius = circle_from_points(*big_face_vertices)

    big_face_mesh = Mesh.from_vertices_and_faces(vertices, [big_face])

    center = centroid_points(big_face_vertices)
    normal = normal_triangle(big_face_vertices)
    # normal = scale_vector(normal, -1.0)
    plane = Plane(center, normal)
    brick_frame = Frame.from_plane(plane)

    # box again
    ratio = (0.4 * radius) / bbox_largest
    S = Scale.from_factors(factors=[ratio] * 3)
    bbox.transform(S)

    # text again
    text_origin = [x for x in bbox.frame.point]
    text_frame = Frame(text_origin, [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0])
    text_mesh.transform(S)
    T = Transformation.from_frame_to_frame(text_frame, brick_frame)
    text_mesh.transform(T)

    B = text_mesh.to_vertices_and_faces(triangulated=True)
    B = mesh_remesh(B, radius/30.0, 10)
    text_mesh = Mesh.from_vertices_and_faces(*B)

    text_mesh = mesh_thicken(text_mesh, depth)

    vertices, faces = bbox.to_vertices_and_faces()
    bbox_mesh = Mesh.from_vertices_and_faces(vertices, faces)

    A = brick.to_vertices_and_faces()
    B = text_mesh.to_vertices_and_faces(triangulated=True)

    V, F = boolean_difference_mesh_mesh(A, B)
    brick = Mesh.from_vertices_and_faces(V, F)

    return text_mesh, bbox_mesh, big_face_mesh, brick


def generate_scaffolding(mesh, thickness):
    """
    Generate the scaffolding platform under the bricks.
    """
    # convert
    scaffold_mesh = mesh_offset(mesh, 1.0 * thickness / 2.0)
    vertices, faces = scaffold_mesh.to_vertices_and_faces()

    # triangulate existing faces
    faces_tri = []
    for face in faces:
        faces_tri.extend(triangulate_face(face, vertices))

    # add boundary face
    key_index = scaffold_mesh.key_index()
    face_bnd = [key_index[fkey] for fkey in scaffold_mesh.vertices_on_boundary()][::-1]
    faces_tri.extend(triangulate_face(face_bnd, vertices))

    # create mesh
    return Mesh.from_vertices_and_faces(vertices, faces_tri)


def add_registration_spheres(brick, radius, fkey, mesh):

    count = 0
    for edge in mesh.face_halfedges(fkey):
        xyz = mesh.edge_midpoint(*edge)
        sphere = Sphere(xyz, radius)
        A = brick.to_vertices_and_faces(triangulated=True)
        B = sphere.to_vertices_and_faces(u=16, v=16, triangulated=True)
        B = mesh_remesh(B, radius / 50.0, 50)
        V, F = boolean_union_mesh_mesh(A, B)
        brick = Mesh.from_vertices_and_faces(V, F)
        count += 1

    return brick


def carve_registration_spheres(brick, radius, fkey, mesh):

    count = 0
    for edge in mesh.face_halfedges(fkey):
        xyz = mesh.edge_midpoint(*edge)
        sphere = Sphere(xyz, radius)
        A = brick.to_vertices_and_faces(triangulated=True)
        # A = mesh_remesh(A, radius / 5., 50)
        B = sphere.to_vertices_and_faces(u=16, v=16, triangulated=True)
        # B = mesh_remesh(B, radius / 5., 50)
        V, F = boolean_difference_mesh_mesh(A, B)
        brick = Mesh.from_vertices_and_faces(V, F)
        count += 1

    return brick


# ===============================================================================
# Script function
# ===============================================================================

def brickify(
        name,
        thickness,
        scale=1.0,
        dual=True,
        do_bricks=False,
        do_label=False,
        do_scaffold=False,
        do_support=False,
        save=False
):
    """
    Generate bricks on the faces of a mesh.
    One face = one brick.

    Parameters
    ___________
    name: `str`
        The mesh name (without extension).
    thickness: `float`
        The brick thickness.
    scale: `float`
        The mesh scale.
    dual: `bool`
        If `True`, the script will work on the dual of the input mesh.
    do_bricks: `bool`
        If `True`, engrave the bricks with labels via mesh boolean differences.
    do_label: `bool`
        If `True`, it engraves the bricks with labels via mesh boolean differences.
    do_scaffold: `bool`
        If `True`, it generates scaffolding platform.
    do_support: `bool`
        If `True`, it creates the perimetral support for the bricks.
    save: `bool`
        If `True`, it will save all generated data as both JSON and OBJ files.
    """
    CAMERA_CONFIG = {
        "position": (30.34, 30.28, 42.94),
        "target": (0.956, 0.727, 1.287),
        "distance": 20.0,
    }

    # load mesh
    filepath = os.path.join(DATA, f"{name}.json")
    mesh = FDMesh.from_json(filepath)
    print(mesh)

    # calculate dual mesh
    if dual:
        mesh = mesh_dual(mesh, include_boundary=True)
        mesh_delete_duplicate_vertices(mesh)

    # scale mesh
    if scale != 1.0:
        S = Scale.from_factors(factors=[scale] * 3)
        mesh.transform(S)

    # do bricks
    if do_bricks:
        bricks, meshes = generate_bricks(mesh, thickness)
        mesh_bottom, mesh_top = meshes

    # generate scaffolding
    if do_scaffold:
        scaffold_mesh = generate_scaffolding(mesh, thickness)

    # engrave labels
    if do_bricks and do_label:
        for i, fkey in enumerate(mesh.faces()):
            data = add_text_engraving(
                fkey,
                bricks,
                mesh_bottom,
                f"{i}",
                depth=thickness/2.0
            )
            text_mesh, bbox_mesh, face_mesh, brick = data
            bricks[fkey] = brick

    if save:
        if do_bricks:
            for i, brick in enumerate(bricks.values()):
                filepath = os.path.join(DATA, f"brick_{i}.json")
                brick.to_json(filepath)
                print(f"Saved brick to {filepath}")
                filepath = os.path.join(DATA, f"brick_{i}.obj")
                brick.to_obj(filepath)
                print(f"Saved brick to {filepath}")

        if do_scaffold:
            filepath = os.path.join(DATA, "scaffold.json")
            scaffold_mesh.to_json(filepath)
            print(f"Saved scaffold to {filepath}")
            filepath = os.path.join(DATA, "scaffold.obj")
            scaffold_mesh.to_obj(filepath)
            print(f"Saved scaffold to {filepath}")

    # visualization
    viewer = Viewer(
        width=900,
        height=900,
        show_grid=True,
        viewmode="ghosted"
    )

    # modify view
    viewer.view.camera.position = CAMERA_CONFIG["position"]
    viewer.view.camera.target = CAMERA_CONFIG["target"]
    viewer.view.camera.distance = CAMERA_CONFIG["distance"]

    if do_scaffold:
        viewer.add(scaffold_mesh)

    if do_bricks:
        for fkey, brick in bricks.items():
            r, g, b = [randint(0, 255) for _ in range(3)]
            color = Color.from_rgb255(r, g, b)

            viewer.add(
                brick,
                color=color,
                show_points=False,
                show_edges=True
            )

    # show le crème
    viewer.show()


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(brickify)
