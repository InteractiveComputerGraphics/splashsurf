import numpy as np

def reconstruct_surface(
    particles: np.ndarray, *, 
    particle_radius: float, 
    rest_density: float,
    smoothing_length: float,
    cube_size: float,
    iso_surface_threshold: float,
    enable_multi_threading: bool,
    global_neighborhood_list: bool,
    use_custom_grid_decomposition: bool,
    subdomain_num_cubes_per_dim: int,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray
) -> PySurfaceReconstructionF32 | PySurfaceReconstructionF64:
    """Reconstruct the surface from only particle positions
    
    Performs a marching cubes surface construction of the fluid represented by the given particle positions
    
    Function defers float precision based on the inputted particle positions (32bit or 64bit)
    
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
    PySurfaceReconstructionF32 | PySurfaceReconstructionF64
        SurfaceReconstruction object containing the reconstructed mesh and used grid
    
    """
    ...