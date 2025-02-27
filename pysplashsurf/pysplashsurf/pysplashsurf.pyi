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
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Reconstruct the surface from only particle positions
    
    Performs a marching cubes surface construction of the fluid represented by the given particle positions
    
    Function defers float precision based on the inputted particle positions (32bit or 64bit)
    
    Parameters
    ----------
    particles: np.ndarray
        2-dimensional array containing all particle positions [[ax, ay, az], [bx, by, bz], ...]
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, tuple]
        tuple of triangles and vertices of the reconstructed surface mesh as well as another tuple containing information of the used grid
    
    """
    ...

def marching_cubes_cleanup(
    triangles: np.ndarray,
    vertices: np.ndarray,
    grid_info: tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray], *,
    max_iter: int,
    keep_vertices: bool
) -> np.ndarray:
    """Mesh simplification designed for marching cubes surfaces meshes inspired by the "Compact Contouring"/"Mesh displacement" approach by Doug Moore and Joe Warren

    See Moore and Warren: ["Mesh Displacement: An Improved Contouring Method for Trivariate Data"](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.5214&rep=rep1&type=pdf) (1991)
    or Moore and Warren: "Compact Isocontours from Sampled Data" in "Graphics Gems III" (1992).
    
    Parameters
    ----------
    triangles: np.ndarray
    
    vertices: np.ndarray
    
    grid_info: tuple[[float], [float], float, [int], [int]]
    
    max_iter: int
    
    keep_vertices: bool
    
    Returns
    -------
    np.ndarray
    
    """