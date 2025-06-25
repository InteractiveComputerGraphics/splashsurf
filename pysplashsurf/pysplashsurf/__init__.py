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
            raise ValueError("Not a valid data type, try explicitly specifying uint64 or float64")
        
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
            raise ValueError("Not a valid data type, try explicitly specifying uint64 or float64")
        
    else:
        raise ValueError("Not a valid data array")

TriMeshWithDataF64.push_point_attribute = lambda self, name, data: push_point_attribute(self, name, data, np.float64)
TriMeshWithDataF64.push_point_attribute.__doc__ = push_point_attribute.__doc__
TriMeshWithDataF32.push_point_attribute = lambda self, name, data: push_point_attribute(self, name, data, np.float32)
TriMeshWithDataF32.push_point_attribute.__doc__ = push_point_attribute.__doc__

TriMeshWithDataF64.push_cell_attribute = lambda self, name, data: push_cell_attribute(self, name, data, np.float64)
TriMeshWithDataF64.push_cell_attribute.__doc__ = push_cell_attribute.__doc__
TriMeshWithDataF32.push_cell_attribute = lambda self, name, data: push_cell_attribute(self, name, data, np.float32)
TriMeshWithDataF32.push_cell_attribute.__doc__ = push_cell_attribute.__doc__

MixedTriQuadMeshWithDataF64.push_point_attribute = lambda self, name, data: push_point_attribute(self, name, data, np.float64)
MixedTriQuadMeshWithDataF64.push_point_attribute.__doc__ = push_point_attribute.__doc__
MixedTriQuadMeshWithDataF32.push_point_attribute = lambda self, name, data: push_point_attribute(self, name, data, np.float32)
MixedTriQuadMeshWithDataF32.push_point_attribute.__doc__ = push_point_attribute.__doc__

MixedTriQuadMeshWithDataF64.push_cell_attribute = lambda self, name, data: push_cell_attribute(self, name, data, np.float64)
MixedTriQuadMeshWithDataF64.push_cell_attribute.__doc__ = push_cell_attribute.__doc__
MixedTriQuadMeshWithDataF32.push_cell_attribute = lambda self, name, data: push_cell_attribute(self, name, data, np.float32)
MixedTriQuadMeshWithDataF32.push_cell_attribute.__doc__ = push_cell_attribute.__doc__

def write_to_file(mesh_with_data, filename, file_format=None, consume_object=False):
    """Write the mesh and its attributes to a file using meshio
    
    Parameters
    ----------
    mesh: TriMeshWithDataF64 | TriMeshWithDataF32 | MixedTriQuadMeshWithDataF64 | MixedTriQuadMeshWithDataF32
        Mesh with data object to write
    
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
        raise ImportError("meshio is not installed, please install it with with `pip install meshio` to use this function")
    
    mesh = mesh_with_data.take_mesh() if consume_object else mesh_with_data.mesh

    point_data = mesh_with_data.get_point_attributes()
    cell_data = mesh_with_data.get_cell_attributes()

    if type(mesh) is pysplashsurf.TriMesh3dF64 or type(mesh) is pysplashsurf.TriMesh3dF32:
        verts, tris = mesh.take_vertices_and_triangles() if consume_object else (mesh.vertices, mesh.triangles)
        meshio.write_points_cells(filename, verts, [("triangle", tris)], point_data=point_data, cell_data=cell_data, file_format=file_format)
            
    else:
        verts, cells = mesh.take_vertices_and_cells() if consume_object else (mesh.vertices, mesh.cells)
        cells = [("triangle", list(filter(lambda x: len(x) == 3, cells))), ("quad", list(filter(lambda x: len(x) == 4, cells)))]
        meshio.write_points_cells(filename, verts, cells, point_data=point_data, cell_data=cell_data, file_format=file_format)


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

def create_sph_interpolator_object(particle_positions, particle_densities, particle_rest_mass, compact_support_radius):
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
    
    if particle_positions.dtype == 'float32':
        return SphInterpolatorF32(particle_positions, particle_densities, particle_rest_mass, compact_support_radius) 
    elif particle_positions.dtype == 'float64':
        return SphInterpolatorF64(particle_positions, particle_densities, particle_rest_mass, compact_support_radius)
    else:
        raise ValueError("Invalid data type (only float32 and float64 are supported, consider explicitly specifying the dtype for particle_positions)")

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
    
    if aabb_min.dtype == 'float32':
        return Aabb3dF32(aabb_min, aabb_max) 
    elif aabb_min.dtype == 'float64':
        return Aabb3dF64(aabb_min, aabb_max)
    else:
        raise ValueError("Invalid data type (only float32 and float64 are supported, consider explicitly specifying the dtype for aabb_min and aabb_max)")

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
    
    if points.dtype == 'float32':
        return Aabb3dF32.from_points(points) 
    elif points.dtype == 'float64':
        return Aabb3dF64.from_points(points)
    else:
        raise ValueError("Invalid data type (only float32 and float64 are supported, consider explicitly specifying the dtype for points)")

def reconstruct_surface(
    particles, *, 
    particle_radius: float = 0.025, 
    rest_density: float = 1000.0,
    smoothing_length: float = 2.0,
    cube_size: float = 0.5,
    iso_surface_threshold: float = 0.6,
    enable_multi_threading: bool = False,
    global_neighborhood_list: bool = False,
    subdomain_grid: bool = False,
    subdomain_num_cubes_per_dim: int = 64,
    aabb_min = None,
    aabb_max = None,
):
    """Reconstruct the surface from only particle positions
    
    Performs a marching cubes surface construction of the fluid represented by the given particle positions
    
    Parameters
    ----------
    particles: np.ndarray
        2-dimensional array containing all particle positions [[ax, ay, az], [bx, by, bz], ...]
        
    particle_radius: float, optional (default=0.025)
        Particle radius
    
    rest_density: float
        Rest density of the fluid
    
    smoothing_length: float
        Smoothing length of the fluid
    
    cube_size: float
        Size of the cubes used in the uniform grid
    
    iso_surface_threshold: float
        Threshold for the iso surface
    
    enable_multi_threading: bool
        Multi-threading flag
    
    global_neighborhood_list: bool
        Global neighborhood list flag
    
    subdomain_grid: bool
        Enable spatial decomposition using a regular grid-based approach
        
    subdomain_num_cubes_per_dim: int
        Each subdomain will be a cube consisting of this number of MC cube cells along each coordinate axis
    
    aabb_min: np.ndarray
        Smallest corner of the axis-aligned bounding box
    
    aabb_max: np.ndarray
        Largest corner of the axis-aligned bounding box
    
    Returns
    -------
    SurfaceReconstructionF32 | SurfaceReconstructionF64
        SurfaceReconstruction object containing the reconstructed mesh and used grid
    
    """
    
    if particles.dtype == 'float32':
        return reconstruct_surface_f32(particles, particle_radius=particle_radius, rest_density=rest_density, 
                                smoothing_length=smoothing_length, cube_size=cube_size, iso_surface_threshold=iso_surface_threshold, 
                                enable_multi_threading=enable_multi_threading, global_neighborhood_list=global_neighborhood_list, 
                                use_custom_grid_decomposition=subdomain_grid, subdomain_num_cubes_per_dim=subdomain_num_cubes_per_dim, 
                                aabb_min=aabb_min, aabb_max=aabb_max)   
    elif particles.dtype == 'float64':
        return reconstruct_surface_f64(particles, particle_radius=particle_radius, rest_density=rest_density, 
                                smoothing_length=smoothing_length, cube_size=cube_size, iso_surface_threshold=iso_surface_threshold, 
                                enable_multi_threading=enable_multi_threading, global_neighborhood_list=global_neighborhood_list, 
                                use_custom_grid_decomposition=subdomain_grid, subdomain_num_cubes_per_dim=subdomain_num_cubes_per_dim, 
                                aabb_min=aabb_min, aabb_max=aabb_max)
    else:
        raise ValueError("Invalid data type (only float32 and float64 are supported, consider explicitly specifying the dtype for particles)")

def marching_cubes_cleanup(
    mesh,
    grid,
    max_iter: int = 5, 
    keep_vertices: bool = False
):
    """Mesh simplification designed for marching cubes surfaces meshes inspired by the "Compact Contouring"/"Mesh displacement" approach by Doug Moore and Joe Warren

    See Moore and Warren: `"Mesh Displacement: An Improved Contouring Method for Trivariate Data" <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.5214&rep=rep1&type=pdf>`_ (1991)
    or Moore and Warren: "Compact Isocontours from Sampled Data" in "Graphics Gems III" (1992).
    
    Parameters
    ----------
    mesh: TriMesh3dF32 | TriMesh3dF64 | TriMeshWithDataF32 | TriMeshWithDataF64
        Mesh object to simplify
    
    grid: UniformGridF32 | UniformGridF64
        Uniform grid object that was used to construct the mesh
    
    max_iter: int
        Maximum number of iterations
    
    keep_vertices: bool
        Flag to keep vertices
    
    Returns
    -------
    list
        vertex connectivity list of the simplified mesh
    """
    if type(mesh) is TriMesh3dF32 or type(mesh) is TriMeshWithDataF32:
        return marching_cubes_cleanup_f32(mesh, grid, max_iter=max_iter, keep_vertices=keep_vertices)
    
    elif type(mesh) is TriMesh3dF64 or type(mesh) is TriMeshWithDataF64:
        return marching_cubes_cleanup_f64(mesh, grid, max_iter=max_iter, keep_vertices=keep_vertices)
    
    else:
        raise ValueError("Invalid mesh type")
    
def decimation(
    mesh,
    keep_vertices: bool = False
):
    """Barnacle decimation
    
    For details see “Weighted Laplacian Smoothing for Surface Reconstruction of Particle-based Fluids” (Löschner, Böttcher, Jeske, Bender; 2023).
    
    Parameters
    ----------
    mesh: TriMesh3dF32 | TriMesh3dF64 | TriMeshWithDataF32 | TriMeshWithDataF64
        Mesh object to simplify
    
    keep_vertices: bool
        Flag to keep vertices
    
    Returns
    -------
    list
        vertex connectivity list of the simplified mesh
    """
    if type(mesh) is TriMesh3dF32 or type(mesh) is TriMeshWithDataF32:
        return decimation_f32(mesh, keep_vertices=keep_vertices)
    
    elif type(mesh) is TriMesh3dF64 or type(mesh) is TriMeshWithDataF64:
        return decimation_f64(mesh, keep_vertices=keep_vertices)
    
    else:
        raise ValueError("Invalid mesh type")

def par_laplacian_smoothing_inplace(
    mesh,
    vertex_connectivity: list[list[int]],
    iterations: int,
    beta: float,
    weights: list[float]
):
    """Laplacian Smoothing with feature weights
    
    Move each vertex towards the mean position of its neighbors.\n
    Factor beta in [0;1] proportional to amount of smoothing (for beta=1 each vertex is placed at the mean position).\n
    Additionally, feature weights can be specified to apply a varying amount of smoothing over the mesh.
    
    Parameters
    ----------
    mesh: TriMesh3dF32 | TriMesh3dF64 | TriMeshWithDataF32 | TriMeshWithDataF64
        Mesh object to smooth
        
    vertex_connectivity: list[list[int]]
        Vertex connectivity list
        
    iterations: int
        Number of iterations
    
    beta: float
        Smoothing factor
    
    weights: list[float]
        Feature weights for the vertices
    """
    
    if type(mesh) is TriMesh3dF32 or type(mesh) is TriMeshWithDataF32:
        par_laplacian_smoothing_inplace_f32(mesh, vertex_connectivity, iterations, beta, weights)
    
    elif type(mesh) is TriMesh3dF64 or type(mesh) is TiMeshWithDataF64:
        par_laplacian_smoothing_inplace_f64(mesh, vertex_connectivity, iterations, beta, weights)
    
    else:
        raise ValueError("Invalid mesh type")
    
def par_laplacian_smoothing_normals_inplace(
    normals: np.ndarray,
    vertex_connectivity: list[list[int]],
    iterations: int
):
    """Laplacian smoothing of a normal field
    
    Parameters
    ----------
    normals: np.ndarray
        2D-Array of vertex normals to smooth
        
    vertex_connectivity: list[list[int]]
        Vertex connectivity list
        
    iterations: int
        Number of iterations
    """
    
    if normals.dtype == 'float32':
        par_laplacian_smoothing_normals_inplace_f32(normals, vertex_connectivity, iterations)
    
    elif normals.dtype == 'float64':
        par_laplacian_smoothing_normals_inplace_f64(normals, vertex_connectivity, iterations)
    
    else:
        raise ValueError("Invalid mesh type")

def neighborhood_search_spatial_hashing_parallel(
    domain,
    particle_positions: np.ndarray,
    search_radius: float
):
    """Performs a neighborhood search (multi-threaded implementation)
    
    Returns the indices of all neighboring particles in the given search radius per particle as a `list[list[int]]`.

    Parameters
    ----------
    domain: Aabb3dF32 | Aabb3dF64
        Axis-aligned bounding box of the domain
        
    particle_positions: np.ndarray
        2D-Array of particle positions
        
    search_radius: float
        Search radius
    """
    
    if type(domain) is Aabb3dF32:
        return neighborhood_search_spatial_hashing_parallel_f32(domain, particle_positions, search_radius)
    
    elif type(domain) is Aabb3dF64:
        return neighborhood_search_spatial_hashing_parallel_f64(domain, particle_positions, search_radius)
    
    else:
        raise ValueError("Invalid domain type")

def check_mesh_consistency(
    grid,
    mesh, *,
    check_closed: bool,
    check_manifold: bool,
    debug: bool,
):
    """Checks the consistency of the mesh (currently checks for holes, non-manifold edges and vertices) and returns a string with debug information in case of problems
    
    Parameters
    ----------
    grid: UniformGridF32 | UniformGridF64
        Uniform grid object
        
    mesh: TriMesh3dF32 | TriMesh3dF64 | TriMeshWithDataF32 | TriMeshWithDataF64
        Triangular mesh object
        
    check_closed: bool
        Flag to check for closed mesh
        
    check_manifold: bool
        Flag to check for manifold mesh
        
    debug: bool
        Flag to enable debug output
    """
    
    if type(grid) is UniformGridF32 and (type(mesh) is TriMesh3dF32 or type(mesh) is TriMeshWithDataF32):
        return check_mesh_consistency_f32(grid, mesh, check_closed=check_closed, check_manifold=check_manifold, debug=debug)
    
    elif type(grid) is UniformGridF64 and (type(mesh) is TriMesh3dF64 or type(mesh) is TriMeshWithDataF64):
        return check_mesh_consistency_f64(grid, mesh, check_closed=check_closed, check_manifold=check_manifold, debug=debug)
    
    else:
        raise ValueError("Invalid grid or mesh type")

def convert_tris_to_quads(
    mesh, *,
    non_squareness_limit: float,
    normal_angle_limit_rad: float,
    max_interior_angle: float,
):
    """Merges triangles sharing an edge to quads if they fulfill the given criteria
    
    Parameters
    ----------
    mesh: TriMesh3dF32 | TriMesh3dF64 | TriMeshWithDataF32 | TriMeshWithDataF64
        Triangular mesh object\n
        When called with a MeshWithData Object, the resulting MixedTriQuadMeshWithData won't inherit the cell attributes from the input.
        
    non_squareness_limit: float
        Non-squareness limit
        
    normal_angle_limit_rad: float
        Normal angle limit in radians
        
    max_interior_angle: float
        Maximum interior angle in radians
        
    Returns
    -------
    MixedTriQuadMesh3dF32 | MixedTriQuadMesh3dF64 | MixedTriQuadMeshWithDataF32 | MixedTriQuadMeshWithDataF64
        Mixed triangular and quadrilateral mesh object
    """
    
    if type(mesh) is TriMesh3dF32 or type(mesh) is TriMeshWithDataF32:
        return convert_tris_to_quads_f32(mesh, non_squareness_limit=non_squareness_limit, normal_angle_limit_rad=normal_angle_limit_rad, max_interior_angle=max_interior_angle)
    
    elif type(mesh) is TriMesh3dF64 or type(mesh) is TriMeshWithDataF64:
        return convert_tris_to_quads_f64(mesh, non_squareness_limit=non_squareness_limit, normal_angle_limit_rad=normal_angle_limit_rad, max_interior_angle=max_interior_angle)
    
    else:
        raise ValueError("Invalid mesh type")
    

def reconstruction_pipeline(
    particles, *, attributes_to_interpolate={}, particle_radius,
    rest_density=1000.0, smoothing_length=2.0, cube_size,
    iso_surface_threshold=0.6, enable_multi_threading=True,
    check_mesh_closed=False, check_mesh_manifold=False,
    check_mesh_orientation=False, check_mesh_debug=False,
    mesh_smoothing_weights=False, sph_normals=False,
    mesh_smoothing_weights_normalization=13.0, mesh_smoothing_iters=None, normals_smoothing_iters=None,
    mesh_cleanup=False, mesh_cleanup_snap_dist=None, decimate_barnacles=False, keep_vertices=False,
    compute_normals=False, output_raw_normals=False, output_raw_mesh=False, output_mesh_smoothing_weights=False, mesh_aabb_clamp_vertices=False,
    subdomain_grid=True, subdomain_num_cubes_per_dim=64, aabb_min=None, aabb_max=None, mesh_aabb_min=None, mesh_aabb_max=None,
    generate_quads=False, quad_max_edge_diag_ratio=1.75, quad_max_normal_angle=10.0, quad_max_interior_angle=135.0
):
    """Surface reconstruction based on particle positions with subsequent post-processing
    
    Parameters
    ----------
    particles: np.ndarray
        2-dimensional array containing all particle positions [[ax, ay, az], [bx, by, bz], ...]
    
    attributes_to_interpolate: dict
        Dictionary containing all attributes to interpolate. The keys are the attribute names and the values are the corresponding 1D/2D arrays.\n
        The arrays must have the same length as the number of particles. \n
        Supported array types are 2D float32/float64 arrays for vector attributes and 1D uint64/float32/float64 arrays for scalar attributes.
        
    particle_radius: float
        Particle radius
        
    rest_density: float
        Rest density of the fluid
        
    smoothing_length: float
        Smoothing length of the fluid in multiples of the particle radius (compact support radius of SPH kernel will be twice the smoothing length)
        
    cube_size: float
        Size of the cubes used for the marching cubes grid in multiples of the particle radius
        
    iso_surface_threshold: float
        Threshold for the iso surface
        
    enable_multi_threading: bool
        Multi-threading flag
        
    check_mesh_closed: bool
        Enable checking the final mesh for holes
        
    check_mesh_manifold: bool
        Enable checking the final mesh for non-manifold edges and vertices
    
    check_mesh_orientation: bool
        Enable checking the final mesh for inverted triangles (compares angle between vertex normals and adjacent face normals)
    
    check_mesh_debug: bool
        Enable additional debug output for the check-mesh operations (has no effect if no other check-mesh option is enabled)

    sph_normals: bool
        Flag to compute normals using SPH interpolation instead of geometry-based normals.
        
    mesh_smoothing_weights: bool
        Flag to compute mesh smoothing weights\n
        This implements the method from “Weighted Laplacian Smoothing for Surface Reconstruction of Particle-based Fluids” (Löschner, Böttcher, Jeske, Bender; 2023).
        
    mesh_smoothing_weights_normalization: float
        Normalization factor for the mesh smoothing weights
        
    mesh_smoothing_iters: int
        Number of iterations for the mesh smoothing
    
    normals_smoothing_iters: int
        Number of iterations for the normal smoothing
        
    mesh_cleanup: bool
        Flag to perform mesh cleanup\n
        This implements the method from “Compact isocontours from sampled data” (Moore, Warren; 1992)

    mesh_cleanup_snap_dist: float
        If MC mesh cleanup is enabled, vertex snapping can be limited to this distance relative to the MC edge length (should be in range of [0.0,0.5])
        
    decimate_barnacles: bool
        Flag to perform barnacle decimation\n
        For details see “Weighted Laplacian Smoothing for Surface Reconstruction of Particle-based Fluids” (Löschner, Böttcher, Jeske, Bender; 2023).
        
    keep_vertices: bool
        Flag to keep any vertices without connectivity resulting from mesh cleanup or decimation step
        
    compute_normals: bool
        Flag to compute normals\n
        If set to True, the normals will be computed and stored in the mesh.
    
    output_mesh_smoothing_weights: bool
        Flag to store the mesh smoothing weights if smoothing weights are computed.
    
    output_raw_normals: bool
        Flag to output the raw normals in addition to smoothed normals if smoothing of normals is enabled
    
    output_raw_mesh: bool
        When true, also return the SurfaceReconstruction object with no post-processing applied
    
    mesh_aabb_clamp_vertices: bool
        Flag to clamp the vertices of the mesh to the AABB
        
    subdomain_grid: bool
        Enables spatial decomposition using a regular grid-based approach
    
    subdomain_num_cubes_per_dim: int
        Each subdomain will be a cube consisting of this number of MC cube cells along each coordinate axis
        
    aabb_min: np.ndarray
        Smallest corner of the axis-aligned bounding box
    
    aabb_max: np.ndarray
        Largest corner of the axis-aligned bounding box
        
    mesh_aabb_min: np.ndarray
        Smallest corner of the axis-aligned bounding box for the mesh
        
    mesh_aabb_max: np.ndarray
        Largest corner of the axis-aligned bounding box for the mesh
    
    generate_quads: bool
        Enable trying to convert triangles to quads if they meet quality criteria
        
    quad_max_edge_diag_ratio: float
        Maximum allowed ratio of quad edge lengths to its diagonals to merge two triangles to a quad (inverse is used for minimum)
    
    quad_max_normal_angle: float
        Maximum allowed angle (in degrees) between triangle normals to merge them to a quad
    
    quad_max_interior_angle: float
        Maximum allowed vertex interior angle (in degrees) inside a quad to merge two triangles to a quad
        
    Returns
    -------
    tuple[TriMeshWithDataF32 | TriMeshWithDataF64 | MixedTriQuadMeshWithDataF32 | MixedTriQuadMeshWithDataF64, Optional[SurfaceReconstructionF32] | Optional[SurfaceReconstructionF64]]
        Mesh with data object and SurfaceReconstruction object containing the reconstructed mesh and used grid
    """
    if particles.dtype == 'float32':
        tri_mesh, tri_quad_mesh, reconstruction = reconstruction_pipeline_f32(particles, attributes_to_interpolate=attributes_to_interpolate, particle_radius=particle_radius, rest_density=rest_density, 
                                                                              smoothing_length=smoothing_length, cube_size=cube_size, iso_surface_threshold=iso_surface_threshold, 
                                                                              aabb_min=aabb_min, aabb_max=aabb_max, enable_multi_threading=enable_multi_threading, 
                                                                              use_custom_grid_decomposition=subdomain_grid, subdomain_num_cubes_per_dim=subdomain_num_cubes_per_dim,
                                                                              check_mesh_closed=check_mesh_closed, check_mesh_manifold=check_mesh_manifold, check_mesh_orientation=check_mesh_orientation, check_mesh_debug=check_mesh_debug,
                                                                              mesh_cleanup=mesh_cleanup, max_rel_snap_dist=mesh_cleanup_snap_dist, decimate_barnacles=decimate_barnacles,
                                                                              keep_vertices=keep_vertices, compute_normals=compute_normals, sph_normals=sph_normals, normals_smoothing_iters=normals_smoothing_iters,
                                                                              mesh_smoothing_iters=mesh_smoothing_iters, mesh_smoothing_weights=mesh_smoothing_weights, mesh_smoothing_weights_normalization=mesh_smoothing_weights_normalization,
                                                                              output_mesh_smoothing_weights=output_mesh_smoothing_weights, output_raw_normals=output_raw_normals, output_raw_mesh=output_raw_mesh,
                                                                              mesh_aabb_min=mesh_aabb_min, mesh_aabb_max=mesh_aabb_max, mesh_aabb_clamp_vertices=mesh_aabb_clamp_vertices,
                                                                              generate_quads=generate_quads, quad_max_edge_diag_ratio=quad_max_edge_diag_ratio, quad_max_normal_angle=quad_max_normal_angle, quad_max_interior_angle=quad_max_interior_angle)  
        
        if tri_mesh == None:
            return (tri_quad_mesh, reconstruction)
        else:
            return (tri_mesh, reconstruction)
        
    elif particles.dtype == 'float64':
        tri_mesh, tri_quad_mesh, reconstruction = reconstruction_pipeline_f64(particles, attributes_to_interpolate=attributes_to_interpolate, particle_radius=particle_radius, rest_density=rest_density,
                                                                              smoothing_length=smoothing_length, cube_size=cube_size, iso_surface_threshold=iso_surface_threshold,
                                                                              aabb_min=aabb_min, aabb_max=aabb_max, enable_multi_threading=enable_multi_threading,
                                                                              use_custom_grid_decomposition=subdomain_grid, subdomain_num_cubes_per_dim=subdomain_num_cubes_per_dim,
                                                                              check_mesh_closed=check_mesh_closed, check_mesh_manifold=check_mesh_manifold, check_mesh_orientation=check_mesh_orientation, check_mesh_debug=check_mesh_debug,
                                                                              mesh_cleanup=mesh_cleanup, max_rel_snap_dist=mesh_cleanup_snap_dist, decimate_barnacles=decimate_barnacles,
                                                                              keep_vertices=keep_vertices, compute_normals=compute_normals, sph_normals=sph_normals, normals_smoothing_iters=normals_smoothing_iters,
                                                                              mesh_smoothing_iters=mesh_smoothing_iters, mesh_smoothing_weights=mesh_smoothing_weights, mesh_smoothing_weights_normalization=mesh_smoothing_weights_normalization,
                                                                              output_mesh_smoothing_weights=output_mesh_smoothing_weights, output_raw_normals=output_raw_normals, output_raw_mesh=output_raw_mesh,
                                                                              mesh_aabb_min=mesh_aabb_min, mesh_aabb_max=mesh_aabb_max, mesh_aabb_clamp_vertices=mesh_aabb_clamp_vertices,
                                                                              generate_quads=generate_quads, quad_max_edge_diag_ratio=quad_max_edge_diag_ratio, quad_max_normal_angle=quad_max_normal_angle, quad_max_interior_angle=quad_max_interior_angle)

        if tri_mesh == None:
            return (tri_quad_mesh, reconstruction)
        else:
            return (tri_mesh, reconstruction)
    else:
        raise ValueError("Invalid data type (only float32 and float64 are supported, consider explicitly specifying the dtype for particles)")
