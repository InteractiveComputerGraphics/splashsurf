import pysplashsurf
import numpy as np
import meshio

def test_reconstruct_surface():
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points)
    print(particles)
    tris, vertices, grid_info = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=True, particle_radius=0.025)
    print(f"Number of tris {len(tris)}, number of vertices {len(vertices)}")
    print(grid_info)
    meshio.write_points_cells("test.vtk", vertices, [("triangle", tris)])
    
#test_reconstruct_surface()

def test_marching_cubes_cleanup(): 
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points)
    print(particles)
    tris, vertices, grid_info = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=True, particle_radius=0.025)
    print(f"Number of tris {len(tris)}, number of vertices {len(vertices)}")
    res = pysplashsurf.marching_cubes_cleanup(tris, vertices, grid_info, max_iter=5, keep_vertices=False)
    print(res)

test_marching_cubes_cleanup()

# def test_4d_array():
#     grid_res = 50
#     grid_p = np.linspace(-1, 1, grid_res)
#     sample_points = np.array(np.meshgrid(grid_p, grid_p, grid_p, indexing="ij")).transpose(1, 2, 3, 0)
#     sample_points = np.ascontiguousarray(sample_points)
#     sdf_samples = np.ones(shape=(grid_res, grid_res, grid_res))
#     sdf_samples[np.linalg.norm(sample_points, axis=-1) < 0.5] = -1.

#     tris, vertices = pysplashsurf.reconstruct_from_uniform_grid_f64(sdf_samples, sample_points[0,0,0], sample_points[1,1,1,0] - sample_points[0,0,0,0], 0.0)
#     print(f"Number of tris {len(tris)}, number of vertices {len(vertices)}")
#     # meshio.write_points_cells("test.vtk", vertices, [("triangle", tris)])


# def test_4d_array_float():
#     grid_res = 50
#     grid_p = np.linspace(-1, 1, grid_res)
#     sample_points = np.array(np.meshgrid(grid_p, grid_p, grid_p, indexing="ij"), dtype=np.float32).transpose(1, 2, 3, 0)
#     sample_points = np.ascontiguousarray(sample_points)
#     sdf_samples = np.ones(shape=(grid_res, grid_res, grid_res), dtype=np.float32)
#     sdf_samples[np.linalg.norm(sample_points, axis=-1) < 0.5] = -1.

#     tris, vertices = pysplashsurf.reconstruct_from_uniform_grid_f32(sdf_samples, sample_points[0,0,0], sample_points[1,1,1,0] - sample_points[0,0,0,0], 0.0)
#     print(f"Number of tris {len(tris)}, number of vertices {len(vertices)}")
#     # meshio.write_points_cells("test.obj", vertices, [("triangle", tris)])

# def test_worst_case():
#     grid_res = 50
#     grid_p = np.linspace(-1, 1, grid_res)
#     sample_points = np.array(np.meshgrid(grid_p, grid_p, grid_p, indexing="ij")).transpose(1, 2, 3, 0)
#     sample_points = np.ascontiguousarray(sample_points)
#     sdf_samples = np.ascontiguousarray(np.random.rand(grid_res, grid_res, grid_res) * 2. - 1.)

#     tris, vertices = pysplashsurf.reconstruct_from_uniform_grid_f64(sdf_samples, sample_points[0,0,0], sample_points[1,0,0,0] - sample_points[0,0,0,0], 0.0)
#     print(f"Number of tris {len(tris)}, number of vertices {len(vertices)}")
#     # meshio.write_points_cells("test.vtk", vertices, [("triangle", tris)])

# def test_worst_case_float():
#     grid_res = 50
#     grid_p = np.linspace(-1, 1, grid_res)
#     sample_points = np.array(np.meshgrid(grid_p, grid_p, grid_p, indexing="ij"), dtype=np.float32).transpose(1, 2, 3, 0)
#     sample_points = np.ascontiguousarray(sample_points)
#     sdf_samples = np.ascontiguousarray(np.random.rand(grid_res, grid_res, grid_res) * 2. - 1.).astype(np.float32)

#     tris, vertices = pysplashsurf.reconstruct_from_uniform_grid_f32(sdf_samples, sample_points[0,0,0], sample_points[1,0,0,0] - sample_points[0,0,0,0], 0.0)
#     print(f"Number of tris {len(tris)}, number of vertices {len(vertices)}")
#     # meshio.write_points_cells("test.vtk", vertices, [("triangle", tris)])
