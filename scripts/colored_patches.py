import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
import matplotlib.cm as cm


def plot_triangles_with_gradient(vertices, triangles, vertex_colors):
    """
    NOTE: Problem is that shading leaves a line at the triangulation seams.
    """
    # Create a Triangulation object using the provided vertices and triangles
    triangulation = mtri.Triangulation(vertices[:,0], vertices[:,1], triangles)

    # Create the plot
    plt.figure()
    ax = plt.subplot(1, 1, 1)

    # Use tripcolor to create a gradient fill within the triangulation
    tcf = ax.tripcolor(triangulation,
                       vertex_colors,
                       shading="gouraud",
                       cmap=cm.magma,
                       )

    # Optionally, draw the outline of the triangulation.
    # ax.triplot(triangulation, 'ko-')

    # Set the limits for the plot
    ax.set_xlim(vertices[:,0].min() - 1, vertices[:,0].max() + 1)
    ax.set_ylim(vertices[:,1].min() - 1, vertices[:,1].max() + 1)
    ax.set_aspect('equal')

    plt.show()


# Define the vertices of the polygon (or polygons)
vertices = np.array([[0, 0], [2, 0], [1, 2], [3, 2]])

# Define the triangles by indices into the vertices array
triangles = np.array([[0, 1, 2], [1, 2, 3]])

# Define the colors for each vertex (in RGBA format)
vertex_colors = np.array([0.0, 0.5, 1.0, 0.0])

# plot_triangles_with_gradient(vertices, triangles, vertex_colors)


def plot_quads_with_gradient(vertices, color_values):
    # Assuming vertices is a grid of points in the form of (x, y) pairs
    # Here, we organize the grid properly for pcolormesh

    # Reshape vertices into a grid format suitable for X and Y coordinates
    # For this example, we use a 3x3 grid of vertices
    num_cols = 3  # Number of columns in the grid
    num_rows = 3  # Number of rows in the grid

    X = vertices[:, 0].reshape((num_rows, num_cols))
    Y = vertices[:, 1].reshape((num_rows, num_cols))

    # The color values array should match the dimensions of the vertices for 'gouraud' shading
    C = color_values.reshape((num_rows - 1, num_cols - 1))

    plt.figure()
    ax = plt.subplot(1, 1, 1)

    # Use pcolormesh to plot the quads. Ensure 'shading' is set correctly for your data
    mesh = ax.pcolormesh(X, Y, C, shading='gouraud', cmap='viridis')

    plt.colorbar(mesh, ax=ax)

    ax.set_xlim(vertices[:,0].min(), vertices[:,0].max())
    ax.set_ylim(vertices[:,1].min(), vertices[:,1].max())
    ax.set_aspect('equal')

    plt.show()


# Example usage
d = np.arange(100).reshape(10, 10)  # the values to be color-mapped
x, y = np.meshgrid(np.arange(10), np.arange(10))

theta = 0.25*np.pi
xx = x*np.cos(theta) - y*np.sin(theta)  # rotate x by -theta
yy = x*np.sin(theta) + y*np.cos(theta)  # rotate y by -theta

plt.figure()
ax = plt.subplot(1, 1, 1)

# Use pcolormesh to plot the quads. Ensure 'shading' is set correctly for your data
mesh = ax.pcolormesh(xx, yy, d, shading='gouraud', cmap='viridis')

plt.colorbar(mesh, ax=ax)

# ax.set_xlim(vertices[:,0].min(), vertices[:,0].max())
# ax.set_ylim(vertices[:,1].min(), vertices[:,1].max())
ax.set_aspect('equal')

# plt.show()

n = 12

x = np.linspace(-1.5, 1.5, n)
y = np.linspace(-1.5, 1.5, n * 2)
print(x.shape, y.shape)

X, Y = np.meshgrid(x, y)
print(X.shape, Y.shape)

Qx = np.cos(Y) - np.cos(X)
Qz = np.sin(Y) + np.sin(X)
print(Qx.shape, Qz.shape)

Z = np.sqrt(X**2 + Y**2) / 5
Z = (Z - Z.min()) / (Z.max() - Z.min())
print(Z.shape)

plt.figure()
ax = plt.subplot(1, 1, 1)
mesh = ax.pcolormesh(Qx, Qz, Z, shading='gouraud', cmap='magma')

plt.colorbar(mesh, ax=ax)

# ax.set_xlim(vertices[:,0].min(), vertices[:,0].max())
# ax.set_ylim(vertices[:,1].min(), vertices[:,1].max())
ax.set_aspect('equal')

plt.show()

vertices = np.array([[0, 0], [1, 0], [2, 0],
                     [0, 1], [1, 1], [2, 1],
                     [0, 2], [1, 2], [2, 2]])

# Define scalar color values for the grid vertices (should match vertices grid)
color_values = np.array([0.0, 0.5, 1.0,
                         0.5, 1.0, 1.5,
                         1.0, 1.5, 2.0])

# NOTE: currently not working
# plot_quads_with_gradient(vertices, color_values)
