from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
import spharapy.trimesh as trimesh
import spharapy.spharabasis as sb
import spharapy.datasets as sd
import random

from eeg.main import *

"""
    Based on the spharapy package, please note that Delaunay "triangulation" of
    XYZ coords doesn't give triangles, but tetrahedrons. Hence dmesh.convex_hull.

    https://spharapy.readthedocs.io/en/latest/auto_examples/plot_02_sphara_basis_eeg.html
"""

def compute_scalp_eigenfunctions(coords):
    dmesh = create_triangular_mesh(coords)
    mesh_eeg = trimesh.TriMesh(dmesh.convex_hull, dmesh.points)
    sphara_basis_unit = sb.SpharaBasis(mesh_eeg, 'unit')
    basis_functions_unit, natural_frequencies_unit = sphara_basis_unit.basis()
    return basis_functions_unit


def get_electrode_coordinates():
    """ Get raw EEGMI data from website or locally if already downloaded """
    raw_fnames = eegbci.load_data(1, runs) #assume same coords for all subjects
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1020")
    raw.set_montage(montage)
    montage_positions = montage.get_positions()
    xyz_coords = montage_positions['ch_pos']
    points = np.array(list(xyz_coords.values()))
    return points


def create_triangular_mesh(xyz_coords):
    """  Create a mesh using the Delaunay triangulation """
    mesh = Delaunay(xyz_coords)
    return mesh


def compute_cotangent_matrix():
    pass


def compute_area_matrix():
    pass


def plot_mesh(mesh, points):
    """ Expects a Delaunay mesh, not a sphara TriMesh """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
    for simplex in mesh.simplices:
            triangle = np.append(simplex, simplex[0])
            ax.plot(points[triangle, 0], points[triangle, 1], points[triangle, 2], 'b-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_basis_functions(dmesh, coords):
    """ Plot 9 random ones """
    basisfuncs = compute_scalp_eigenfunctions(coords)
    figsb1, axes1 = plt.subplots(nrows=3, ncols=3, figsize=(8, 12),
                                                              subplot_kw={'projection': '3d'})

    for i in range(np.size(axes1)):
        #colors = np.mean(basisfuncs[dmesh.convex_hull, i + 0], axis=1)
        ax = axes1.flat[i]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=60., azim=80.)
        ax.set_aspect('auto')
        trisurfplot = ax.plot_trisurf(dmesh.points[:, 0], dmesh.points[:, 1],
        dmesh.points[:, 2], triangles=dmesh.convex_hull,
        cmap=plt.cm.bwr,
        edgecolor='white', linewidth=0.)
        #trisurfplot.set_array(colors)
        trisurfplot.set_clim(-0.15, 0.15)

    cbar = figsb1.colorbar(trisurfplot, ax=axes1.ravel().tolist(), shrink=0.85,
    orientation='horizontal', fraction=0.05, pad=0.05,
    anchor=(0.5, -4.5))

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=1.0)
    plt.show()



if __name__ == '__main__':
    coords = get_electrode_coordinates()
    dmesh = create_triangular_mesh(coords)
    plot_basis_functions(dmesh, coords)
