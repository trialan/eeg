from mpl_toolkits.mplot3d import Axes3D
import spharapy.trimesh as tm
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
    mesh_eeg = create_triangular_dmesh(coords)
    sphara_basis_unit = sb.SpharaBasis(mesh_eeg, 'unit')
    basis_functions_unit, natural_frequencies_unit = sphara_basis_unit.basis()
    return basis_functions_unit


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
    mesh_eeg = trimesh.TriMesh(mesh.convex_hull, mesh.points)
    return mesh_eeg


def plot_mesh(mesh):
    """ Expects a sphara TriMesh """
    vertlist = mesh._vertlist
    trilist = mesh._trilist
    fig = plt.figure()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('The triangulated EEG sensor setup')
    ax.view_init(elev=20., azim=80.)
    ax.set_aspect('auto')
    ax.plot_trisurf(vertlist[:, 0], vertlist[:, 1], vertlist[:, 2],
                    triangles=trilist, color='lightblue', edgecolor='black',
                    linewidth=0.5, shade=True)
    plt.show()


def plot_basis_functions(dmesh, coords):
    """ Plot 9 first ones """
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

    #cbar = figsb1.colorbar(trisurfplot, ax=axes1.ravel().tolist(), shrink=0.85,
        #orientation='horizontal', fraction=0.05, pad=0.05,
        #anchor=(0.5, -4.5))

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=1.0)
    plt.show()


if __name__ == '__main__':
    coords = get_electrode_coordinates()
    dmesh = create_triangular_dmesh(coords)
    plot_mesh(dmesh)
    #plot_basis_functions(dmesh, coords)

