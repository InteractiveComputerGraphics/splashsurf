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
    

def test_post_processing():
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points, dtype=np.float32)
    reconstruction = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=True, particle_radius=0.025, 
                                                                 rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                                                                 iso_surface_threshold=0.6)
    mesh = pysplashsurf.post_processing(particles, reconstruction, enable_multi_threading=True, particle_radius=0.025)
    print(f"Number of tris {len(mesh.triangles)}, number of vertices {len(mesh.vertices)}")
    meshio.write_points_cells("test.vtk", mesh.vertices, [("triangle", mesh.triangles)])


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


def test_decimation():
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points, dtype=np.float32)
    reconstruction = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=True, particle_radius=0.025, 
                                                                 rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                                                                 iso_surface_threshold=0.6, aabb_min=np.array([0.0, 0.0, 0.0]), aabb_max=np.array([2.0, 2.0, 2.0]))
    mesh = reconstruction.mesh
    print(f"Number of tris {len(mesh.triangles)}, number of vertices {len(mesh.vertices)}")
    res = pysplashsurf.decimation(mesh, keep_vertices=False)
    print(f"Number of tris {len(mesh.triangles)}, number of vertices {len(mesh.vertices)}")
    print(len(res))
    meshio.write_points_cells("test.vtk", mesh.vertices, [("triangle", mesh.triangles)])
   
def test_laplacian_smoothing():
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points, dtype=np.float32)
    reconstruction = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=True, particle_radius=0.025, 
                                                                 rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                                                                 iso_surface_threshold=0.6, aabb_min=np.array([0.0, 0.0, 0.0]), aabb_max=np.array([2.0, 2.0, 2.0]))
    mesh = reconstruction.mesh
    grid = reconstruction.grid
    res = pysplashsurf.marching_cubes_cleanup(mesh, grid, max_iter=5, keep_vertices=False)
    print(f"Number of tris {len(mesh.triangles)}, number of vertices {len(mesh.vertices)}")
    pysplashsurf.par_laplacian_smoothing_inplace(mesh, res, 5, 1.0, [1.0 for _ in range(len(mesh.vertices))])
    print(f"Number of tris {len(mesh.triangles)}, number of vertices {len(mesh.vertices)}")
    meshio.write_points_cells("test.vtk", mesh.vertices, [("triangle", mesh.triangles)])

def test_mesh_with_data():
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points, dtype=np.float32)
    reconstruction = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=True, particle_radius=0.025, 
                                                                 rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                                                                 iso_surface_threshold=0.6, aabb_min=np.array([0.0, 0.0, 0.0]), aabb_max=np.array([2.0, 2.0, 2.0]))
    mesh = reconstruction.mesh
    grid = reconstruction.grid
    
    vertex_connectivity = pysplashsurf.marching_cubes_cleanup(mesh, grid, max_iter=5, keep_vertices=False)
    
    mesh_with_data = pysplashsurf.PyMeshWithDataF32(mesh)
    pysplashsurf.calculate_smoothed_normals(mesh_with_data, vertex_connectivity, smoothing_iters=5)
    
    ex_scalars = np.array([i for i in range(len(mesh.vertices))], dtype=np.uint64)
    ex_reals = np.array([i for i in range(len(mesh.vertices))], dtype=np.float32)
    ex_vectors = np.array([[i, i, i] for i in range(len(mesh.vertices))], dtype=np.float32)
    
    mesh_with_data.push_point_attribute("test_scalar", ex_scalars)
    mesh_with_data.push_point_attribute("test_reals", ex_reals)
    mesh_with_data.push_point_attribute("test_vector", ex_vectors)
    
    assert((mesh_with_data.get_point_attribute("test_scalar") == ex_scalars).all())
    assert((mesh_with_data.get_point_attribute("test_reals") == ex_reals).all())
    assert((mesh_with_data.get_point_attribute("test_vector") == ex_vectors).all())
    
    ex_cell_scalars = np.array([i for i in range(1000)], dtype=np.uint64)
    mesh_with_data.push_cell_attribute("test_cell_scalar", ex_cell_scalars)
    assert((mesh_with_data.get_cell_attribute("test_cell_scalar") == ex_cell_scalars).all())

    normals = mesh_with_data.get_point_attribute("normals")
    
    print(normals)
    
#test_reconstruct_surface()
#test_post_processing()
#test_marching_cubes_cleanup() 
#test_decimation()
#test_laplacian_smoothing()
test_mesh_with_data()