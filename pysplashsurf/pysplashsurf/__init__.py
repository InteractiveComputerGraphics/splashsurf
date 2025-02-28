from .pysplashsurf import *
import numpy as np

def marching_cubes_cleanup(
    mesh,
    grid,
    max_iter: int = 5, 
    keep_vertices: bool = False
) -> list:
    """Mesh simplification designed for marching cubes surfaces meshes inspired by the "Compact Contouring"/"Mesh displacement" approach by Doug Moore and Joe Warren

    See Moore and Warren: ["Mesh Displacement: An Improved Contouring Method for Trivariate Data"](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.5214&rep=rep1&type=pdf) (1991)
    or Moore and Warren: "Compact Isocontours from Sampled Data" in "Graphics Gems III" (1992).
    
    Parameters
    ----------
    mesh: PyTriMesh3dF32 | PyTriMesh3dF64
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
    if type(mesh) is PyTriMesh3dF32:
        return marching_cubes_cleanup_f32(mesh, grid, max_iter=max_iter, keep_vertices=keep_vertices)
    
    elif type(mesh) is PyTriMesh3dF64:
        return marching_cubes_cleanup_f64(mesh, grid, max_iter=max_iter, keep_vertices=keep_vertices)
    
    else:
        raise ValueError("Invalid mesh type")

def post_processing(
    particles, reconstruction, *, 
    particle_radius: float = 0.025, 
    rest_density: float = 1000.0,
    smoothing_length: float = 2.0,
    cube_size: float = 0.5,
    iso_surface_threshold: float = 0.6,
    enable_multi_threading: bool = False,
    global_neighborhood_list: bool = False,
    use_custom_grid_decomposition: bool = False,
    subdomain_num_cubes_per_dim: int = 64,
    aabb_min = None,
    aabb_max = None
):
    """Post processing of a surface reconstruction
    
    Parameters
    ----------
    particles: np.ndarray
        2-dimensional array containing all particle positions [[ax, ay, az], [bx, by, bz], ...]
    
    reconstruction: PySurfaceReconstructionF32 | PySurfaceReconstructionF64
        Surface reconstruction object
        
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
    
    use_custom_grid_decomposition: bool
        Custom grid decomposition flag
    
    subdomain_num_cubes_per_dim: int
        Number of cubes per dimension per grid cell
    
    aabb_min: np.ndarray
        Smallest corner of the axis-aligned bounding box
    
    aabb_max: np.ndarray
        Largest corner of the axis-aligned bounding box
    
    Returns
    -------
    PyTriMesh3dF32 | PyTriMesh3dF64
    """
    if type(reconstruction) is PySurfaceReconstructionF32:
        return post_processing_f32(particles, reconstruction, 
                                   particle_radius=particle_radius,
                                   rest_density=rest_density,
                                   smoothing_length=smoothing_length,
                                   cube_size=cube_size,
                                   iso_surface_threshold=iso_surface_threshold,
                                   enable_multi_threading=enable_multi_threading,
                                   global_neighborhood_list=global_neighborhood_list,
                                   use_custom_grid_decomposition=use_custom_grid_decomposition,
                                   subdomain_num_cubes_per_dim=subdomain_num_cubes_per_dim,
                                   aabb_min=aabb_min,
                                   aabb_max=aabb_max)
        
    elif type(reconstruction) is PySurfaceReconstructionF64:
        return post_processing_f64(particles, reconstruction, 
                                   particle_radius=particle_radius,
                                   rest_density=rest_density,
                                   smoothing_length=smoothing_length,
                                   cube_size=cube_size,
                                   iso_surface_threshold=iso_surface_threshold,
                                   enable_multi_threading=enable_multi_threading,
                                   global_neighborhood_list=global_neighborhood_list,
                                   use_custom_grid_decomposition=use_custom_grid_decomposition,
                                   subdomain_num_cubes_per_dim=subdomain_num_cubes_per_dim,
                                   aabb_min=aabb_min,
                                   aabb_max=aabb_max)
        
    else:
        raise ValueError("Invalid reconstruction type")