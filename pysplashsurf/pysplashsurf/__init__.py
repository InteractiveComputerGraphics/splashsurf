from .pysplashsurf import *
import numpy as np

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

PyTriMeshWithDataF64.push_point_attribute = lambda self, name, data: push_point_attribute(self, name, data, np.float64)
PyTriMeshWithDataF64.push_point_attribute.__doc__ = push_point_attribute.__doc__
PyTriMeshWithDataF32.push_point_attribute = lambda self, name, data: push_point_attribute(self, name, data, np.float32)
PyTriMeshWithDataF32.push_point_attribute.__doc__ = push_point_attribute.__doc__

PyTriMeshWithDataF64.push_cell_attribute = lambda self, name, data: push_cell_attribute(self, name, data, np.float64)
PyTriMeshWithDataF64.push_cell_attribute.__doc__ = push_cell_attribute.__doc__
PyTriMeshWithDataF32.push_cell_attribute = lambda self, name, data: push_cell_attribute(self, name, data, np.float32)
PyTriMeshWithDataF32.push_cell_attribute.__doc__ = push_cell_attribute.__doc__

PyMixedTriQuadMeshWithDataF64.push_point_attribute = lambda self, name, data: push_point_attribute(self, name, data, np.float64)
PyMixedTriQuadMeshWithDataF64.push_point_attribute.__doc__ = push_point_attribute.__doc__
PyMixedTriQuadMeshWithDataF32.push_point_attribute = lambda self, name, data: push_point_attribute(self, name, data, np.float32)
PyMixedTriQuadMeshWithDataF32.push_point_attribute.__doc__ = push_point_attribute.__doc__

PyMixedTriQuadMeshWithDataF64.push_cell_attribute = lambda self, name, data: push_cell_attribute(self, name, data, np.float64)
PyMixedTriQuadMeshWithDataF64.push_cell_attribute.__doc__ = push_cell_attribute.__doc__
PyMixedTriQuadMeshWithDataF32.push_cell_attribute = lambda self, name, data: push_cell_attribute(self, name, data, np.float32)
PyMixedTriQuadMeshWithDataF32.push_cell_attribute.__doc__ = push_cell_attribute.__doc__


def create_mesh_with_data_object(mesh):
    """Create the corresponding mesh with data object to a mesh object
    
    Parameters
    ----------
    mesh: PyTriMesh3dF64 | PyTriMesh3dF32 | PyMixedTriQuadMesh3dF64 | PyMixedTriQuadMesh3dF32
        Mesh object to convert
        
    Returns
    -------
    PyTriMeshWithDataF64 | PyTriMeshWithDataF32 | PyMixedTriQuadMeshWithDataF64 | PyMixedTriQuadMeshWithDataF32
        Mesh with data object
    """
    
    if type(mesh) is PyTriMesh3dF64:
        return PyTriMeshWithDataF64(mesh)
    elif type(mesh) is PyTriMesh3dF32:
        return PyTriMeshWithDataF32(mesh)
    elif type(mesh) is PyMixedTriQuadMesh3dF64:
        return PyMixedTriQuadMeshWithDataF64(mesh)
    elif type(mesh) is PyMixedTriQuadMesh3dF32:
        return PyMixedTriQuadMeshWithDataF32(mesh)
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
    PySphInterpolatorF32 | PySphInterpolatorF64
        SphInterpolator object
    """
    
    if particle_positions.dtype == 'float32':
        return PySphInterpolatorF32(particle_positions, particle_densities, particle_rest_mass, compact_support_radius) 
    elif particle_positions.dtype == 'float64':
        return PySphInterpolatorF64(particle_positions, particle_densities, particle_rest_mass, compact_support_radius)
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
    PyAabb3dF32 | PyAabb3dF64
        Aabb object
    """
    
    if aabb_min.dtype == 'float32':
        return PyAabb3dF32(aabb_min, aabb_max) 
    elif aabb_min.dtype == 'float64':
        return PyAabb3dF64(aabb_min, aabb_max)
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
    PyAabb3dF32 | PyAabb3dF64
        Aabb object
    """
    
    if points.dtype == 'float32':
        return PyAabb3dF32.from_points(points) 
    elif points.dtype == 'float64':
        return PyAabb3dF64.from_points(points)
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
    PySurfaceReconstructionF32 | PySurfaceReconstructionF64
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

    See Moore and Warren: ["Mesh Displacement: An Improved Contouring Method for Trivariate Data"](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.5214&rep=rep1&type=pdf) (1991)
    or Moore and Warren: "Compact Isocontours from Sampled Data" in "Graphics Gems III" (1992).
    
    Parameters
    ----------
    mesh: PyTriMesh3dF32 | PyTriMesh3dF64 | PyTriMeshWithDataF32 | PyTriMeshWithDataF64
        Mesh object to simplify
    
    grid: PyUniformGridF32 | PyUniformGridF64
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
    if type(mesh) is PyTriMesh3dF32 or type(mesh) is PyTriMeshWithDataF32:
        return marching_cubes_cleanup_f32(mesh, grid, max_iter=max_iter, keep_vertices=keep_vertices)
    
    elif type(mesh) is PyTriMesh3dF64 or type(mesh) is PyTriMeshWithDataF64:
        return marching_cubes_cleanup_f64(mesh, grid, max_iter=max_iter, keep_vertices=keep_vertices)
    
    else:
        raise ValueError("Invalid mesh type")
    
def decimation(
    mesh,
    keep_vertices: bool = False
):
    """Barnacle decimation
    
    Parameters
    ----------
    mesh: PyTriMesh3dF32 | PyTriMesh3dF64 | PyTriMeshWithDataF32 | PyTriMeshWithDataF64
        Mesh object to simplify
    
    keep_vertices: bool
        Flag to keep vertices
    
    Returns
    -------
    list
        vertex connectivity list of the simplified mesh
    """
    if type(mesh) is PyTriMesh3dF32 or type(mesh) is PyTriMeshWithDataF32:
        return decimation_f32(mesh, keep_vertices=keep_vertices)
    
    elif type(mesh) is PyTriMesh3dF64 or type(mesh) is PyTriMeshWithDataF64:
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
    
    Move each vertex towards the mean position of its neighbors.
    Factor beta in [0;1] proportional to amount of smoothing (for beta=1 each vertex is placed at the mean position).
    Additionally, feature weights can be specified to apply a varying amount of smoothing over the mesh.
    
    Parameters
    ----------
    mesh: PyTriMesh3dF32 | PyTriMesh3dF64 | PyTriMeshWithDataF32 | PyTriMeshWithDataF64
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
    
    if type(mesh) is PyTriMesh3dF32 or type(mesh) is PyTriMeshWithDataF32:
        par_laplacian_smoothing_inplace_f32(mesh, vertex_connectivity, iterations, beta, weights)
    
    elif type(mesh) is PyTriMesh3dF64 or type(mesh) is PyTiMeshWithDataF64:
        par_laplacian_smoothing_inplace_f64(mesh, vertex_connectivity, iterations, beta, weights)
    
    else:
        raise ValueError("Invalid mesh type")
    
def par_laplacian_smoothing_normals_inplace(
    normals: np.ndarray,
    vertex_connectivity: list[list[int]],
    iterations: int
):
    """Laplacian Smoothing of the vertex normals
    
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
    
    Returns the indices of all neighboring particles in the given search radius per particle as a `Vec<Vec<usize>>`.

    Parameters
    ----------
    domain: PyAabb3dF32 | PyAabb3dF64
        Axis-aligned bounding box of the domain
        
    particle_positions: np.ndarray
        2D-Array of particle positions
        
    search_radius: float
        Search radius
    """
    
    if type(domain) is PyAabb3dF32:
        return neighborhood_search_spatial_hashing_parallel_f32(domain, particle_positions, search_radius)
    
    elif type(domain) is PyAabb3dF64:
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
    """Check mesh consistency
    
    Parameters
    ----------
    grid: PyUniformGridF32 | PyUniformGridF64
        Uniform grid object
        
    mesh: PyTriMesh3dF32 | PyTriMesh3dF64 | PyTriMeshWithDataF32 | PyTriMeshWithDataF64
        Triangular mesh object
        
    check_closed: bool
        Flag to check for closed mesh
        
    check_manifold: bool
        Flag to check for manifold mesh
        
    debug: bool
        Flag to enable debug output
    """
    
    if type(grid) is PyUniformGridF32 and (type(mesh) is PyTriMesh3dF32 or type(mesh) is PyTriMeshWithDataF32):
        return check_mesh_consistency_f32(grid, mesh, check_closed=check_closed, check_manifold=check_manifold, debug=debug)
    
    elif type(grid) is PyUniformGridF64 and (type(mesh) is PyTriMesh3dF64 or type(mesh) is PyTriMeshWithDataF64):
        return check_mesh_consistency_f64(grid, mesh, check_closed=check_closed, check_manifold=check_manifold, debug=debug)
    
    else:
        raise ValueError("Invalid grid or mesh type")

def convert_tris_to_quads(
    mesh, *,
    non_squareness_limit: float,
    normal_angle_limit_rad: float,
    max_interior_angle: float,
):
    """Converts triangles to quads
    
    Parameters
    ----------
    mesh: PyTriMesh3dF32 | PyTriMesh3dF64
        Triangular mesh object
        
    non_squareness_limit: float
        Non-squareness limit
        
    normal_angle_limit_rad: float
        Normal angle limit in radians
        
    max_interior_angle: float
        Maximum interior angle in radians
        
    Returns
    -------
    PyMixedTriQuadMesh3dF32 | PyMixedTriQuadMesh3dF64
        Mixed triangular and quadrilateral mesh object
    """
    
    if type(mesh) is PyTriMesh3dF32:
        return convert_tris_to_quads_f32(mesh, non_squareness_limit=non_squareness_limit, normal_angle_limit_rad=normal_angle_limit_rad, max_interior_angle=max_interior_angle)
    
    elif type(mesh) is PyTriMesh3dF64:
        return convert_tris_to_quads_f64(mesh, non_squareness_limit=non_squareness_limit, normal_angle_limit_rad=normal_angle_limit_rad, max_interior_angle=max_interior_angle)
    
    else:
        raise ValueError("Invalid mesh type")
    

def reconstruction_pipeline(
    particles, *, particle_radius=0.025, 
    rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
    iso_surface_threshold=0.6, enable_multi_threading=True, mesh_smoothing_weights=False, sph_normals=False, 
    mesh_smoothing_weights_normalization=13.0, mesh_smoothing_iters=5, normals_smoothing_iters=5,
    mesh_cleanup=False, decimate_barnacles=False, keep_vertices=False,
    compute_normals=False, output_raw_normals=False, output_mesh_smoothing_weights=False, mesh_aabb_clamp_vertices=False,
    subdomain_grid=False, subdomain_num_cubes_per_dim=64, aabb_min=None, aabb_max=None, mesh_aabb_min=None, mesh_aabb_max=None
):
    if particles.dtype == 'float32':
        return reconstruction_pipeline_f32(particles, particle_radius=particle_radius, rest_density=rest_density, 
                                smoothing_length=smoothing_length, cube_size=cube_size, iso_surface_threshold=iso_surface_threshold, 
                                aabb_min=aabb_min, aabb_max=aabb_max, enable_multi_threading=enable_multi_threading, 
                                use_custom_grid_decomposition=subdomain_grid, subdomain_num_cubes_per_dim=subdomain_num_cubes_per_dim,
                                global_neighborhood_list=False, mesh_cleanup=mesh_cleanup, decimate_barnacles=decimate_barnacles,
                                keep_vertices=keep_vertices, compute_normals=compute_normals, sph_normals=sph_normals, normals_smoothing_iters=normals_smoothing_iters,
                                mesh_smoothing_iters=mesh_smoothing_iters, mesh_smoothing_weights=mesh_smoothing_weights, mesh_smoothing_weights_normalization=mesh_smoothing_weights_normalization,
                                output_mesh_smoothing_weights=output_mesh_smoothing_weights, output_raw_normals=output_raw_normals, mesh_aabb_min=mesh_aabb_min, mesh_aabb_max=mesh_aabb_max, mesh_aabb_clamp_vertices=mesh_aabb_clamp_vertices)   
    elif particles.dtype == 'float64':
        return reconstruction_pipeline_f64(particles, particle_radius=particle_radius, rest_density=rest_density, 
                                smoothing_length=smoothing_length, cube_size=cube_size, iso_surface_threshold=iso_surface_threshold, 
                                aabb_min=aabb_min, aabb_max=aabb_max, enable_multi_threading=enable_multi_threading, 
                                use_custom_grid_decomposition=subdomain_grid, subdomain_num_cubes_per_dim=subdomain_num_cubes_per_dim,
                                global_neighborhood_list=False, mesh_cleanup=mesh_cleanup, decimate_barnacles=decimate_barnacles,
                                keep_vertices=keep_vertices, compute_normals=compute_normals, sph_normals=sph_normals, normals_smoothing_iters=normals_smoothing_iters,
                                mesh_smoothing_iters=mesh_smoothing_iters, mesh_smoothing_weights=mesh_smoothing_weights, mesh_smoothing_weights_normalization=mesh_smoothing_weights_normalization,
                                output_mesh_smoothing_weights=output_mesh_smoothing_weights, output_raw_normals=output_raw_normals, mesh_aabb_min=mesh_aabb_min, mesh_aabb_max=mesh_aabb_max, mesh_aabb_clamp_vertices=mesh_aabb_clamp_vertices)   
    else:
        raise ValueError("Invalid data type (only float32 and float64 are supported, consider explicitly specifying the dtype for particles)")