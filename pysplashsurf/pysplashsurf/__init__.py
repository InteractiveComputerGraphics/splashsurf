from .pysplashsurf import *
import numpy as np

def marching_cubes_cleanup(mesh, grid, max_iter=5, keep_vertices=False):
    if type(mesh) is PyTriMesh3dF32:
        print("F32")
        return marching_cubes_cleanup_f32(mesh, grid, max_iter=max_iter, keep_vertices=keep_vertices)
    elif type(mesh) is PyTriMesh3dF64:
        print("F64")
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
    if type(reconstruction) is PySurfaceReconstructionF32:
        print("F32")
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
        print("F64")
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