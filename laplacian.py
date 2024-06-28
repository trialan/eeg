import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import spharapy.trimesh as tm
from scipy.spatial import Delaunay, ConvexHull
import spharapy.trimesh as trimesh
import spharapy.spharabasis as sb
import spharapy.datasets as sd
import random
import matplotlib.pyplot as plt

from eeg.data import *

"""
    Based on the spharapy package, please note that Delaunay "triangulation" of
    XYZ coords doesn't give triangles, but tetrahedrons. Hence dmesh.convex_hull.

    https://spharapy.readthedocs.io/en/latest/auto_examples/plot_02_sphara_basis_eeg.html
"""


def compute_scalp_eigenmodes(mesh):
    sphara_basis_unit = sb.SpharaBasis(mesh, 'inv_euclidean')
    eigenmodes, eigenvals = sphara_basis_unit.basis()
    return eigenmodes


def get_electrode_coordinates(subject=1):
    """ Get raw EEGMI data from website or locally if already downloaded """
    raw_fnames = eegbci.load_data(subject, runs) #all subjects have same coords
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1020")
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    montage_positions = montage.get_positions()
    xyz_coords = montage_positions['ch_pos']
    points = np.array(list(xyz_coords.values()))
    return points


def create_triangular_dmesh(xyz_coords):
    """  Create a mesh using the Delaunay triangulation """
    mesh = Delaunay(xyz_coords)
    mesh = trimesh.TriMesh(mesh.convex_hull, mesh.points)
    mesh = remove_bottom_of_the_mesh(mesh)
    return mesh


def distance(p1, p2):
    dx =(p1[0] - p2[0])
    dy =(p1[1] - p2[1])
    dz =(p1[2] - p2[2])
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    return distance


def length_of_longest_edge(triangle, vertices):
    p1 = vertices[triangle[0]]
    p2 = vertices[triangle[1]]
    p3 = vertices[triangle[2]]
    el1 = distance(p1, p2)
    el2 = distance(p1, p3)
    el3 = distance(p2, p3)
    return max(el1, el2, el3)


def remove_bottom_of_the_mesh(mesh, N=6):
    """ Visual inspection reveals that the bottom triangles we want to be rid
        of have the longest edgs. So we remove the N triangles with longest edge """
    le =  [length_of_longest_edge(t, mesh.vertlist) for t in mesh.trilist]
    ixs = np.argsort(le)[::-1][:N]
    ntriangles = [mesh.trilist[ix] for ix in range(len(mesh.trilist)) if ix not in ixs]
    mesh.trilist = np.array(ntriangles)
    return mesh


def plot_mesh(mesh):
    """ Expects a sphara TriMesh """
    vertices = np.array(mesh.vertlist)
    triangles = np.array(mesh.trilist)
    fig = plt.figure()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('The triangulated EEG sensor setup')
    ax.view_init(elev=20., azim=80.)
    ax.set_aspect('auto')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=triangles, color='lightblue', edgecolor='black',
                    linewidth=0.5, shade=True, alpha=1)
    plt.show()


def plot_basis_functions(mesh):
    vertices = np.array(mesh.vertlist)
    triangles = np.array(mesh.trilist)
    eigenmodes = compute_scalp_eigenmodes(mesh)

    figsb1, axes1 = plt.subplots(nrows=7, ncols=7, figsize=(8, 12),
                                 subplot_kw={'projection': '3d'})
    for i in range(np.size(axes1)):
        colors = np.mean(eigenmodes[triangles, i + 0], axis=1)
        ax = axes1.flat[i]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=60., azim=80.)
        ax.set_aspect('auto')
        trisurfplot = ax.plot_trisurf(vertices[:, 0], vertices[:, 1],
                                      vertices[:, 2], triangles=triangles,
                                      cmap=plt.cm.bwr,
                                      edgecolor='white', linewidth=0.)
        trisurfplot.set_array(colors)
        trisurfplot.set_clim(-0.15, 0.15)

    cbar = figsb1.colorbar(trisurfplot, ax=axes1.ravel().tolist(), shrink=0.85,
                           orientation='horizontal', fraction=0.05, pad=0.05,
                           anchor=(0.5, -4.5))

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=1.0)
    plt.show()


if __name__ == '__main__':
    xyz_coords = get_electrode_coordinates()
    mesh = create_triangular_dmesh(xyz_coords)
    plot_mesh(mesh)
    plot_basis_functions(mesh)

