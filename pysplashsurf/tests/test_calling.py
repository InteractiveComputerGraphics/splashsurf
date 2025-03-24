import pysplashsurf
import numpy as np
import math
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
    
    mesh_with_data = pysplashsurf.PyMeshWithDataF64(mesh)
    
    print(f"Number of tris {len(mesh.triangles)}, number of vertices {len(mesh.vertices)}")
    pysplashsurf.par_laplacian_smoothing_inplace(mesh_with_data, res, 5, 1.0, [1.0 for _ in range(len(mesh.vertices))])
    print(f"Number of tris {len(mesh.triangles)}, number of vertices {len(mesh.vertices)}")
    meshio.write_points_cells("test.vtk", mesh.vertices, [("triangle", mesh.triangles)])

def test_mesh_with_data():
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points, dtype=np.float64)
    reconstruction = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=True, particle_radius=0.025, 
                                                                 rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                                                                 iso_surface_threshold=0.6, aabb_min=np.array([0.0, 0.0, 0.0]), aabb_max=np.array([2.0, 2.0, 2.0]))
    mesh = reconstruction.mesh
    grid = reconstruction.grid
    
    vertex_connectivity = pysplashsurf.marching_cubes_cleanup(mesh, grid, max_iter=5, keep_vertices=False)
    
    normals = mesh.par_vertex_normals()
    pysplashsurf.par_laplacian_smoothing_normals_inplace(normals, vertex_connectivity, 5)
    
    mesh_with_data = pysplashsurf.PyMeshWithDataF64(mesh)
    mesh_with_data.push_point_attribute("normals", normals)
    
    ex_scalars = np.array([i for i in range(len(mesh.vertices))], dtype=np.uint64)
    ex_reals = np.array([i for i in range(len(mesh.vertices))], dtype=np.float64)
    ex_vectors = np.array([[i, i, i] for i in range(len(mesh.vertices))], dtype=np.float64)
    
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

def reconstruction_pipeline(*, enable_multi_threading=True, particle_radius=0.025, 
                            rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                            iso_surface_threshold=0.6, mesh_smoothing_weights=True, sph_normals=True, 
                            mesh_smoothing_weights_normalization=100.0, mesh_smoothing_iters=5, normals_smoothing_iters=5,
                            mesh_aabb = None, mesh_cleanup=True, decimate_barnacles=True, keep_vertices=False,
                            compute_normals=True, output_raw_normals=False, mesh_aabb_clamp_vertices=True,
                            check_mesh_closed=True, check_mesh_manifold=True, check_mesh_debug=False,
                            generate_quads=True, quad_max_edge_diag_ratio=1.75, quad_max_normal_angle=10.0, quad_max_interior_angle=135.0):
    compact_support_radius = 2.0 * particle_radius * smoothing_length
    
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points, dtype=np.float64)
    reconstruction = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=enable_multi_threading, particle_radius=particle_radius, 
                                                                 rest_density=rest_density, smoothing_length=smoothing_length, cube_size=cube_size, 
                                                                 iso_surface_threshold=iso_surface_threshold)
    
    grid = reconstruction.grid
    mesh = reconstruction.mesh
    
    # Mesh Cleanup
    if mesh_cleanup:
        vertex_connectivity = pysplashsurf.marching_cubes_cleanup(mesh, grid, max_iter=5, keep_vertices=keep_vertices)
    
    # Decimate mesh barnacles
    if decimate_barnacles:
        vertex_connectivity = pysplashsurf.decimation(mesh, keep_vertices=keep_vertices)
    
    # Initialize SPH Interpolator if required
    interpolator_required = mesh_smoothing_weights or sph_normals
    if interpolator_required:
        particle_rest_volume = 4.0/3.0 * np.pi * particle_radius**3
        particle_rest_mass = rest_density * particle_rest_volume
        particle_densities = reconstruction.particle_densities()
        
        interpolator = pysplashsurf.PySphInterpolatorF64(particles, particle_densities, particle_rest_mass, compact_support_radius)
    
    # Compute mesh vertex-vertex connectivity map if not done already and required
    if vertex_connectivity is None and (mesh_smoothing_iters > 0 or normals_smoothing_iters > 0):
        vertex_connectivity = mesh.vertex_vertex_connectivity()
    
    if mesh_smoothing_weights:
        # Compute smoothing weights
        # First do global neighborhood search
        nl = reconstruction.particle_neighbors()
        if nl is None:
            search_radius = compact_support_radius
            aabb = pysplashsurf.PyAabb3dF64.from_points(particles)
            aabb.grow_uniformly(search_radius)
            
            nl = pysplashsurf.neighborhood_search_spatial_hashing_parallel(aabb, particles, search_radius)
        
        # Compute weighted neighbor count
        squared_r = compact_support_radius * compact_support_radius
        weighted_ncounts = [sum([1 - max(0, min(1, (np.dot(particles[i]-particles[j], particles[i]-particles[j]) / squared_r))) for j in nll]) for i, nll in enumerate(nl)]
        vertex_weighted_num_neighbors = np.array(interpolator.interpolate_scalar_quantity(weighted_ncounts, mesh.vertices, True))
        
        # Create Mesh with Data object
        mesh_with_data = pysplashsurf.PyMeshWithDataF64(mesh)
        
        # Now compute the weights
        offset = 0
        normalization = mesh_smoothing_weights_normalization - offset
        
        smoothing_weights = [max(n-offset, 0) for n in vertex_weighted_num_neighbors]
        smoothing_weights = [min(n/normalization, 1.0) for n in smoothing_weights]
        smoothing_weights = np.array([6*pow(x,5) - 15*pow(x,4) + 10*pow(x,3) for x in smoothing_weights])
        
        mesh_with_data.push_point_attribute("wnn", vertex_weighted_num_neighbors)
        mesh_with_data.push_point_attribute("sw", smoothing_weights)
    
    # Perform smoothing
    if smoothing_weights is None:
        smoothing_weights = [1.0 for _ in range(len(mesh.vertices))]
    
    pysplashsurf.par_laplacian_smoothing_inplace(mesh_with_data, vertex_connectivity, mesh_smoothing_iters, 1.0, smoothing_weights)
    
    mesh = mesh_with_data.mesh
    
    # Compute normals
    if compute_normals:
        if sph_normals:
            normals = interpolator.interpolate_normals(mesh.vertices)
        else:
            normals = mesh.par_vertex_normals()
        
        # Smooth normals
        smoothed_normals = normals.copy()
        pysplashsurf.par_laplacian_smoothing_normals_inplace(smoothed_normals, vertex_connectivity, mesh_smoothing_iters)
        
        # Add normals to mesh
        mesh_with_data.push_point_attribute("normals", smoothed_normals)
        
        if output_raw_normals:
            mesh_with_data.push_point_attribute("raw_normals", normals)
    
    # Left out: Interpolate attributes if requested 
    
    # Remove and clamp cells outside AABB
    if mesh_aabb is not None:
        mesh_with_data = mesh_with_data.par_clamp_with_aabb(mesh_aabb, mesh_aabb_clamp_vertices, keep_vertices)
        
    mesh = mesh_with_data.mesh
    
    # Convert triangles to quads
    if generate_quads:
        mesh = pysplashsurf.convert_tris_to_quads(mesh, non_squareness_limit=quad_max_edge_diag_ratio, normal_angle_limit_rad=math.radians(quad_max_normal_angle), max_interior_angle=math.radians(quad_max_interior_angle))
    
    if type(mesh) is pysplashsurf.PyTriMesh3dF64:
        meshio.write_points_cells("test.vtk", mesh.vertices, [("triangle", mesh.triangles)])
        
        # Mesh checks
        pysplashsurf.check_mesh_consistency(grid, mesh, check_closed=check_mesh_closed, check_manifold=check_mesh_manifold, debug=check_mesh_debug)
        
    else:
        cells = mesh.cells
        cells = [("triangle", list(filter(lambda x: len(x) == 3, cells))), ("quad", list(filter(lambda x: len(x) == 4, cells)))]
        meshio.write_points_cells("test.vtk", mesh.vertices, cells)
    
    
    # Left out: Mesh orientation check
    

#test_reconstruct_surface()
#test_post_processing()
#test_marching_cubes_cleanup() 
#test_decimation()
#test_laplacian_smoothing()
#test_mesh_with_data()
reconstruction_pipeline()