import pysplashsurf
import numpy as np
import meshio

def test_reconstruct_surface():
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points, dtype=np.float64)
    reconstruction = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=True, particle_radius=0.025, 
                                                                 rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                                                                 iso_surface_threshold=0.6, aabb_min=np.array([0.0, 0.0, 0.0]), aabb_max=np.array([2.0, 2.0, 2.0]))
    mesh = reconstruction.mesh
    grid = reconstruction.grid
    print(f"Mesh: {mesh}, Grid: {grid}")
    tris = mesh.triangles
    verts = mesh.vertices
    print(f"Number of tris {len(tris)}, number of vertices {len(verts)}")
    meshio.write_points_cells("test.vtk", verts, [("triangle", tris)])
    
test_reconstruct_surface()

def test_post_processing():
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points, dtype=np.float32)
    reconstruction = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=True, particle_radius=0.025, 
                                                                 rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                                                                 iso_surface_threshold=0.6)
    mesh = pysplashsurf.post_processing(particles, reconstruction, enable_multi_threading=True, particle_radius=0.025)
    print(f"Number of tris {len(mesh.triangles)}, number of vertices {len(mesh.vertices)}")
    meshio.write_points_cells("test.vtk", mesh.vertices, [("triangle", mesh.triangles)])

test_post_processing()

def test_marching_cubes_cleanup(): 
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points, dtype=np.float32)
    reconstruction = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=True, particle_radius=0.025, 
                                                                 rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                                                                 iso_surface_threshold=0.6, aabb_min=np.array([0.0, 0.0, 0.0]), aabb_max=np.array([2.0, 2.0, 2.0]))
    mesh = reconstruction.mesh
    grid = reconstruction.grid
    print(f"Number of tris {len(mesh.triangles)}, number of vertices {len(mesh.vertices)}")
    res = pysplashsurf.marching_cubes_cleanup(mesh, grid, max_iter=5, keep_vertices=False)
    print(f"Number of tris {len(mesh.triangles)}, number of vertices {len(mesh.vertices)}")
    print(len(res))

test_marching_cubes_cleanup()
