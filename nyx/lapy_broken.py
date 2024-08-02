from lapy import Solver, TriaMesh, TetMesh
import vtk
import numpy as np
from tqdm import tqdm


"""
    Getting eigenmodes worked for second subject Nyx sent on FB. Doesn't work
    for the first subject.

    Red flags:
        - the mask isn't the same length as "v" for the first subject.
        - there's a whole in the masked mesh for this subject

    Warning: I renamed the files hence 'p1_mesh_v2.vtk'
"""


def convert_new_vtk_to_old_vtk(new_vtk_filename, new_name):
    """ inputs: a new-format vtk file, saves an old-format vtk file new_name """
    points, offsets, connectivity = read_new_vtk(new_vtk_filename)
    points, polygons = convert_to_old_format(points, offsets, connectivity)
    write_old_vtk(new_name, points, polygons)


def read_new_vtk(filename):

    with open(filename, 'r') as file:
        lines = file.readlines()

    # Find the start of each section
    points_start = next(i for i, line in enumerate(lines) if line.startswith('POINTS'))
    polygons_start = next(i for i, line in enumerate(lines) if line.startswith('POLYGONS'))
    offsets_start = polygons_start + 2
    connectivity_start = next(i for i, line in enumerate(lines) if line.startswith('CONNECTIVITY'))

    # Read points
    num_points = int(lines[points_start].split()[1])
    points = []
    for line in lines[points_start+1:polygons_start]:
        points.extend(map(float, line.split()))
    points = [points[i:i+3] for i in range(0, len(points), 3)]

    # Read offsets
    offsets = []
    for line in lines[offsets_start:connectivity_start]:
        offsets.extend(map(int, line.split()))

    # Read connectivity
    connectivity = []
    for line in lines[connectivity_start+1:]:
        connectivity.extend(map(int, line.split()))

    return points, offsets, connectivity


def convert_to_old_format(points, offsets, connectivity):
    old_format_polygons = []
    for i in range(len(offsets) - 1):
        start = offsets[i]
        end = offsets[i+1]
        polygon = [3] + connectivity[start:end]
        old_format_polygons.append(polygon)
    return points, old_format_polygons


def write_old_vtk(filename, points, polygons):
    with open(filename, 'w') as file:
        # Write header
        file.write("# vtk DataFile Version 5.1\n")
        file.write("vtk output\n")
        file.write("ASCII\n")
        file.write("DATASET POLYDATA\n")

        # Write points
        file.write(f"POINTS {len(points)} float\n")
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

        # Write polygons
        total_values = sum(len(polygon) for polygon in polygons)
        file.write(f"POLYGONS {len(polygons)} {total_values}\n")
        for polygon in polygons:
            file.write(" ".join(map(str, polygon)) + "\n")


def remap_indices(old_indices, good_vertices):
    def remap(idx):
        return index_map.get(idx, -1)
    # Use vectorize to apply the remap function to the entire array
    return np.vectorize(remap)(old_indices)


if __name__ == '__main__':
    # Read the mesh and mask
    mesh = TriaMesh.read_vtk('p1_mesh_v2.vtk')
    v = mesh.v
    t = mesh.t

    mask = np.loadtxt('p3_mask.txt', dtype=int)
    good_vertices = np.where(mask == 1)[0]
    bad_vertices = np.where(mask == 0)[0]
    good_v = v[good_vertices]

    # Remap triangles
    index_map = {old: new for new, old in enumerate(good_vertices)}
    good_t = []
    for triangle in tqdm(t):
        if not any(vertex in bad_vertices for vertex in triangle):
            remapped_triangle = remap_indices(triangle, good_vertices)
            if not np.any(remapped_triangle == -1):
                good_t.append(remapped_triangle)

    good_t = np.array(good_t)

    # Create new mesh
    new_mesh = TriaMesh(good_v, good_t)

    import spharapy.trimesh as tm
    from eeg.laplacian import plot_mesh
    spymesh = tm.TriMesh(mesh.t, mesh.v)
    plot_mesh(spymesh)

    num_modes = 10
    fem = Solver(mesh)
    evals, emodes = fem.eigs(k=num_modes)


