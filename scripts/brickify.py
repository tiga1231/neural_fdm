"""
Generate bricks on the faces of a mesh.
"""
# import open3d as o3d
from open3d.t.geometry import TriangleMesh

import os
from random import randint

from compas.colors import Color

from compas.datastructures import Mesh
from compas.datastructures import mesh_dual
from compas.datastructures import mesh_thicken
from compas.datastructures import mesh_unify_cycles
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
# Script function
# ===============================================================================

def brickify(name, thickness, scale=1.0, dual=True, label=False, save=False, save_scaffold=False):
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
        If `True`, it will generate the bricks on the dual of the mesh.
    label: `bool`
        If `True`, it will engrave the bricks with labels via mesh boolean differences.
    save: `bool`
        If `True`, it will save the bricks as independent JSON and OBJ files.
    save_scaffold: `bool`
        If `True`, it will save a scaffolding platform a a JSON and OBJ files.
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

    # make brick meshes
    half_thick = thickness / 2.0
    faces_topbottom = {}
    bricks = {}
    halfedges_visited = set()
    for i, fkey in enumerate(mesh.faces()):

        # top and bottom faces
        vertices = []
        faces_0 = []
        vertices_0 = []

        vkey_counter = 0
        for direction in (1.0, -1.0):

            face = []
            for vkey in mesh.face_vertices(fkey):

                xyz = mesh.vertex_coordinates(vkey)
                normal = mesh.vertex_normal(vkey)
                vector_extr = scale_vector(normal, half_thick * direction)
                xyz_extr = add_vectors(xyz, vector_extr)
                vertices.append(xyz_extr)

                face.append(vkey_counter)
                vkey_counter += 1

            faces_0.append(face)
            vertices_0.append([vertices[v] for v in face])

        # side faces
        faces_topbottom[fkey] = vertices_0
        face_top, face_bottom = faces_0
        face_top = face_top + face_top[:1]
        face_bottom = face_bottom + face_bottom[:1]

        # adding side faces
        faces = []
        halfedges = mesh.face_halfedges(fkey)
        for edge_top, edge_bottom, halfedge in zip(pairwise(face_bottom), pairwise(face_top), halfedges):
            a, b = edge_top
            d, c = edge_bottom
            # triangulate
            if halfedge in halfedges_visited:
                face_a = [a, b, d]
                face_b = [b, c, d]
            else:
                face_a = [a, b, c]
                face_b = [c, d, a]

            faces.append(face_a)
            faces.append(face_b)

        # face half edges
        halfedges_visited.update(halfedges)
        halfedges_reversed = [(v, u) for u, v in halfedges]
        halfedges_visited.update(halfedges_reversed)

        # triangulate top and bottom faces
        for j, face in enumerate(faces_0):
            # triangle
            if len(face) == 3:
                _faces = [face]
            # quad
            elif len(face) == 4:
                a, b, c, d = face
                face_a = [a, b, d]
                face_b = [b, c, d]
                _faces = [face_a, face_b]
            # ngon
            else:
                # triangulate to midpoint
                midpoint = centroid_points([vertices[vkey] for vkey in face])
                vertices.append(midpoint)

                # create new faces
                _faces = []
                for a, b in pairwise(face + face[:1]):
                    _face = [a, b, vkey_counter]
                    _faces.append(_face)

                vkey_counter += 1

            # reverse bottom faces
            for _face in _faces:
                if j == 1:
                    _face = _face[::-1]
                faces.append(_face)

        # make mesh from face
        brick = Mesh.from_vertices_and_faces(vertices, faces)

        # store brick
        bricks[fkey] = brick


    # generate scaffolding
    if save_scaffold:

        # move scaffold
        scaffold_mesh = mesh.copy()

        for vkey in scaffold_mesh.vertices():

            xyz = mesh.vertex_coordinates(vkey)
            normal = mesh.vertex_normal(vkey)
            vector_extr = scale_vector(normal, half_thick * 1.0)
            xyz_extr = add_vectors(xyz, vector_extr)
            scaffold_mesh.vertex_attributes(vkey, "xyz", xyz_extr)

        # convert
        vertices, faces = scaffold_mesh.to_vertices_and_faces()

        # triangulate faces
        vkey_counter = len(vertices)
        faces_tri = []
        for face in faces:
            # triangle
            if len(face) == 3:
                _faces = [face]
            # quad
            elif len(face) == 4:
                a, b, c, d = face
                face_a = [a, b, d]
                face_b = [b, c, d]
                _faces = [face_a, face_b]
            # ngon
            else:
                # triangulate to midpoint
                midpoint = centroid_points([vertices[vkey] for vkey in face])
                vertices.append(midpoint)

                # create new faces
                _faces = []
                for a, b in pairwise(face + face[:1]):
                    _face = [a, b, vkey_counter]
                    _faces.append(_face)

                vkey_counter += 1

            # store faces
            for _face in _faces:
                faces_tri.append(_face)

        # add boundary face
        key_index = scaffold_mesh.key_index()
        face = [key_index[fkey] for fkey in scaffold_mesh.vertices_on_boundary()][::-1]

        # assume it is an ngon
        # triangulate to midpoint
        midpoint = centroid_points([vertices[vkey] for vkey in face])
        vertices.append(midpoint)

        # create new faces
        _faces = []
        for a, b in pairwise(face + face[:1]):
            _face = [a, b, vkey_counter]
            faces_tri.append(_face)

        vkey_counter += 1

        # create mesh
        scaffold_mesh = Mesh.from_vertices_and_faces(vertices, faces_tri)

        # save mesh
        filepath = os.path.join(DATA, f"scaffold.json")
        scaffold_mesh.to_json(filepath)
        print(f"Saved scaffold to {filepath}")
        filepath = os.path.join(DATA, f"scaffold.obj")
        scaffold_mesh.to_obj(filepath)
        print(f"Saved scaffold to {filepath}")

    # add registration spheres

    # test union sphere
    def add_registration_spheres(brick, radius, fkey, mesh):

        count = 0
        for edge in mesh.face_halfedges(fkey):
            # if count > 0:
                # continue
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

    def add_text_engraving(fkey, bricks, faces_topbottom, text, depth=1):
        """
        Engrave a brick with text.
        """
        text_mesh = text_2_mesh(text)
        vertices, _ = text_mesh.to_vertices_and_faces()
        bbox = oriented_bounding_box_numpy(vertices)
        bbox = Box.from_bounding_box(bbox)
        bbox_largest = max([bbox.width, bbox.depth, bbox.height])

        # find face in brick
        brick = bricks[fkey]

        # generate face plane
        vertices = faces_topbottom[fkey][0]  # NOTE: or -1?
        face = list(range(len(vertices)))
        # triangle
        if len(face) == 3:
            _faces = [face]
        # quad
        elif len(face) == 4:
            a, b, c, d = face
            face_a = [a, b, d]
            face_b = [b, c, d]
            _faces = [face_a, face_b]
        # ngon
        else:
            # triangulate to midpoint
            midpoint = centroid_points([vertices[vkey] for vkey in face])
            ckey = len(vertices) - 1
            vertices.append(midpoint)
            # create new faces
            _faces = []
            for a, b in pairwise(face + face[:1]):
                _face = [a, b, ckey]
                _faces.append(_face)

        # take face with largest area
        _faces = sorted(_faces, key=lambda x: area_triangle([vertices[v] for v in x]))
        big_face = _faces[-1]
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
        text_mesh = mesh.from_vertices_and_faces(*B)

        text_mesh = mesh_thicken(text_mesh, depth)

        vertices, faces = bbox.to_vertices_and_faces()
        bbox_mesh = Mesh.from_vertices_and_faces(vertices, faces)

        A = brick.to_vertices_and_faces()
        B = text_mesh.to_vertices_and_faces(triangulated=True)

        V, F = boolean_difference_mesh_mesh(A, B)
        brick = Mesh.from_vertices_and_faces(V, F)

        return text_mesh, bbox_mesh, big_face_mesh, brick


    if label:
        # fkey = list(mesh.faces())[-1]
        # brick = list(bricks.values())[-1]

        # data = add_text_engraving(
        #     fkey,
        #     bricks,
        #     faces_topbottom,
        #     "99",
        #     depth=thickness/2.0
        # )
        for i, fkey in enumerate(mesh.faces()):
             data = add_text_engraving(
                 fkey,
                 bricks,
                 faces_topbottom,
                 f"{i}",
                 depth=thickness/2.0
             )

             text_mesh, bbox_mesh, face_mesh, brick = data
             bricks[fkey] = brick

    if save:
        for i, brick in enumerate(bricks.values()):
            filepath = os.path.join(DATA, f"brick_{i}.json")
            brick.to_json(filepath)
            print(f"Saved brick to {filepath}")

            filepath = os.path.join(DATA, f"brick_{i}.obj")
            brick.to_obj(filepath)
            print(f"Saved brick to {filepath}")

    # radius = thickness * 0.25
    # mesh_unified = carve_registration_spheres(brick, radius, fkey, mesh)

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

    # viewer.add(mesh_unified)
    if label:
        viewer.add(text_mesh, color=Color.blue())
        # viewer.add(bbox_mesh)
        # viewer.add(face_mesh, color=Color.pink())
        viewer.add(brick)

    if save_scaffold:
        viewer.add(scaffold_mesh)

    for fkey, brick in bricks.items():
        r, g, b = [randint(0, 255) for _ in range(3)]
        color = Color.from_rgb255(r, g, b)

        # viewer.add(
        #     brick,
        #     color=color,
        #     show_points=False,
        #     show_edges=True,
        #     opacity=0.5
        # )

    # show le crème
    viewer.show()


# ===============================================================================
# Main
# ===============================================================================

if __name__ == "__main__":

    from fire import Fire

    Fire(brickify)
