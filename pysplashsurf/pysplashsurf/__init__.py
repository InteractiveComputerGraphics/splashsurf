from .pysplashsurf import *
from . import bgeo
import numpy as np
import sys


def run_pysplashsurf():
    run_splashsurf(sys.argv)


def push_point_attribute(self, name: str, data: np.ndarray, real_type):
    """Add a point attribute to the mesh"""
    if data.ndim == 2:
        return self.push_point_attribute_vector_real(name, data)

    elif data.ndim == 1:
        if data.dtype == np.uint64:
            return self.push_point_attribute_scalar_u64(name, data)

        elif data.dtype == real_type:
            return self.push_point_attribute_scalar_real(name, data)

        else:
            raise ValueError(
                "Not a valid data type, try explicitly specifying uint64 or float64"
            )

    else:
        raise ValueError("Not a valid data array")


def push_cell_attribute(self, name: str, data: np.ndarray, real_type):
    """Add a cell attribute to the mesh"""
    if data.ndim == 2:
        return self.push_cell_attribute_vector_real(name, data)

    elif data.ndim == 1:
        if data.dtype == np.uint64:
            return self.push_cell_attribute_scalar_u64(name, data)

        elif data.dtype == real_type:
            return self.push_cell_attribute_scalar_real(name, data)

        else:
            raise ValueError(
                "Not a valid data type, try explicitly specifying uint64 or float64"
            )

    else:
        raise ValueError("Not a valid data array")


def write_to_file(mesh, filename, file_format=None, consume_object=False):
    """Write the mesh and its attributes to a file using meshio

    Parameters
    ----------
    mesh: TriMesh3d | MixedTriQuadMesh3d | MeshWithData
        Mesh object to write

    filename: Any
        File path for the output file

    file_format: str | None
        File format for the output file, generally also derived from filename

    consume_object: bool
        Flag for specifying whether the MeshWithData object should be consumed for a faster execution.
        Only consumes the mesh field.
    """
    try:
        import meshio
    except ImportError:
        raise ImportError(
            "meshio is not installed, please install it with with `pip install meshio` to use this function"
        )

    point_data = (
        mesh.point_attributes if type(mesh) is pysplashsurf.MeshWithData else {}
    )
    cell_data = mesh.cell_attributes if type(mesh) is pysplashsurf.MeshWithData else {}
    mesh = mesh.mesh if type(mesh) is pysplashsurf.MeshWithData else mesh

    if type(mesh) is pysplashsurf.TriMesh3d:
        meshio.write_points_cells(
            filename,
            mesh.vertices,
            [("triangle", mesh.triangles)],
            point_data=point_data,
            cell_data=cell_data,
            file_format=file_format,
        )
    elif type(mesh) is pysplashsurf.MixedTriQuadMesh3d:
        cells = [
            ("triangle", mesh.get_triangles()),
            ("quad", mesh.get_quads()),
        ]
        meshio.write_points_cells(
            filename,
            mesh.vertices,
            cells,
            point_data=point_data,
            cell_data=cell_data,
            file_format=file_format,
        )
    else:
        raise TypeError("unsupported mesh type")


def create_mesh_with_data_object(mesh):
    """Create the corresponding mesh with data object to a mesh object

    Parameters
    ----------
    mesh: TriMesh3dF64 | TriMesh3dF32 | MixedTriQuadMesh3dF64 | MixedTriQuadMesh3dF32
        Mesh object to convert

    Returns
    -------
    TriMeshWithDataF64 | TriMeshWithDataF32 | MixedTriQuadMeshWithDataF64 | MixedTriQuadMeshWithDataF32
        Mesh with data object
    """

    if type(mesh) is TriMesh3dF64:
        return TriMeshWithDataF64(mesh)
    elif type(mesh) is TriMesh3dF32:
        return TriMeshWithDataF32(mesh)
    elif type(mesh) is MixedTriQuadMesh3dF64:
        return MixedTriQuadMeshWithDataF64(mesh)
    elif type(mesh) is MixedTriQuadMesh3dF32:
        return MixedTriQuadMeshWithDataF32(mesh)
    else:
        raise ValueError("Invalid mesh type")


def create_sph_interpolator_object(
    particle_positions, particle_densities, particle_rest_mass, compact_support_radius
):
    """Create the corresponding SPH interpolator object to a set of particle data

    Parameters
    ----------
    particle_positions: np.ndarray
        2-dimensional array containing all particle positions [[ax, ay, az], [bx, by, bz], ...]

    particle_densities: np.ndarray
        1-dimensional array containing all particle densities

    particle_rest_mass: float
        Rest mass of the particles

    compact_support_radius: float
        Compact support radius of the SPH kernel

    Returns
    -------
    SphInterpolatorF32 | SphInterpolatorF64
        SphInterpolator object
    """

    if particle_positions.dtype == "float32":
        return SphInterpolatorF32(
            particle_positions,
            particle_densities,
            particle_rest_mass,
            compact_support_radius,
        )
    elif particle_positions.dtype == "float64":
        return SphInterpolatorF64(
            particle_positions,
            particle_densities,
            particle_rest_mass,
            compact_support_radius,
        )
    else:
        raise ValueError(
            "Invalid data type (only float32 and float64 are supported, consider explicitly specifying the dtype for particle_positions)"
        )


def create_aabb_object(aabb_min, aabb_max):
    """Create the corresponding AABB object to a set of min and max values

    Parameters
    ----------
    aabb_min: np.ndarray
        Smallest corner of the axis-aligned bounding box

    aabb_max: np.ndarray
        Largest corner of the axis-aligned bounding box

    Returns
    -------
    Aabb3dF32 | Aabb3dF64
        Aabb object
    """

    if aabb_min.dtype == "float32":
        return Aabb3dF32(aabb_min, aabb_max)
    elif aabb_min.dtype == "float64":
        return Aabb3dF64(aabb_min, aabb_max)
    else:
        raise ValueError(
            "Invalid data type (only float32 and float64 are supported, consider explicitly specifying the dtype for aabb_min and aabb_max)"
        )


def create_aabb_object_from_points(points):
    """Create the corresponding AABB object to a set of points

    Parameters
    ----------
    points: np.ndarray
        2-dimensional array containing all point positions [[ax, ay, az], [bx, by, bz], ...]

    Returns
    -------
    Aabb3dF32 | Aabb3dF64
        Aabb object
    """

    if points.dtype == "float32":
        return Aabb3dF32.from_points(points)
    elif points.dtype == "float64":
        return Aabb3dF64.from_points(points)
    else:
        raise ValueError(
            "Invalid data type (only float32 and float64 are supported, consider explicitly specifying the dtype for points)"
        )
