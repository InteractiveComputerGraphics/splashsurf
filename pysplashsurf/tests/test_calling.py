import pysplashsurf
import numpy as np
import meshio

def test_reconstruct_surface():
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points, dtype=np.float64)
    print(particles)
    tris, vertices, grid_info = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=True, particle_radius=0.025, 
                                                                 rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                                                                 iso_surface_threshold=0.6, aabb_min=np.array([0.0, 0.0, 0.0]), aabb_max=np.array([2.0, 2.0, 2.0]))
    print(f"Number of tris {len(tris)}, number of vertices {len(vertices)}")
    print(grid_info)
    meshio.write_points_cells("test.vtk", vertices, [("triangle", tris)])
    
#test_reconstruct_surface()

def test_post_processing():
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points, dtype=np.float32)
    print(particles)
    tris, vertices = pysplashsurf.post_processing_f32(particles, enable_multi_threading=True, particle_radius=0.025)
    print(f"Number of tris {len(tris)}, number of vertices {len(vertices)}")
    meshio.write_points_cells("test.vtk", vertices, [("triangle", tris)])

#test_post_processing()

def test_marching_cubes_cleanup(): 
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points, dtype=np.float32)
    print(particles)
    tris, vertices, grid_info = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=True, particle_radius=0.025, 
                                                                 rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                                                                 iso_surface_threshold=0.6, aabb_min=np.array([0.0, 0.0, 0.0]), aabb_max=np.array([2.0, 2.0, 2.0]))
    print(f"Number of tris {len(tris)}, number of vertices {len(vertices)}")
    res = pysplashsurf.marching_cubes_cleanup(tris, vertices, grid_info, max_iter=5, keep_vertices=False)
    print(res)

test_marching_cubes_cleanup()
