import pysplashsurf
import numpy as np
import math
import meshio
import subprocess
import time

BINARY_PATH = "splashsurf"

def test_memory_access():
    print("\nTesting memory copy vs take")
    
    particles = np.array(meshio.read("./ParticleData_Fluid_5.vtk").points, dtype=np.float64)
    reconstruction = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=True, particle_radius=0.025, 
                                                                 rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                                                                 iso_surface_threshold=0.6, aabb_min=np.array([0.0, 0.0, 0.0]), aabb_max=np.array([2.0, 2.0, 2.0]))
    mesh = reconstruction.mesh
    
    start = time.time()
    triangles_copy = mesh.triangles
    vertices_copy = mesh.vertices
    copy_time = time.time() - start
    print("Copy time:", copy_time)
    
    start = time.time()
    vertices, triangles = mesh.take_vertices_and_triangles()
    take_time = time.time() - start
    print("Take time:", take_time)
    
    print("Copy time / Take time (Speedup):", copy_time / take_time)
    
    assert(np.allclose(vertices, vertices_copy))
    assert(np.allclose(triangles, triangles_copy))

def reconstruction_pipeline(input_file, output_file, *, enable_multi_threading=True, particle_radius=0.025, 
                            rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                            iso_surface_threshold=0.6, mesh_smoothing_weights=False, sph_normals=False, 
                            mesh_smoothing_weights_normalization=13.0, mesh_smoothing_iters=5, normals_smoothing_iters=5,
                            mesh_aabb = None, mesh_cleanup=False, decimate_barnacles=False, keep_vertices=False,
                            compute_normals=False, output_raw_normals=False, mesh_aabb_clamp_vertices=False,
                            check_mesh_closed=True, check_mesh_manifold=True, check_mesh_debug=False,
                            generate_quads=False, quad_max_edge_diag_ratio=1.75, quad_max_normal_angle=10.0, quad_max_interior_angle=135.0,
                            subdomain_grid=False, subdomain_num_cubes_per_dim=64):
    
    compact_support_radius = np.float64(2.0) * particle_radius * smoothing_length
    
    particles = np.array(meshio.read(input_file).points, dtype=np.float64)
    reconstruction = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=enable_multi_threading, particle_radius=particle_radius, 
                                                                 rest_density=rest_density, smoothing_length=smoothing_length, cube_size=cube_size, 
                                                                 iso_surface_threshold=iso_surface_threshold, subdomain_grid=subdomain_grid,
                                                                 subdomain_num_cubes_per_dim=subdomain_num_cubes_per_dim)
    
    grid = reconstruction.grid
    mesh = reconstruction.mesh
    
    vertex_connectivity = None
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
    
    # Create Mesh with Data object
    mesh_with_data = pysplashsurf.PyMeshWithDataF64(mesh)
    
    smoothing_weights = None
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
        
        # Now compute the weights
        offset = 0
        normalization = mesh_smoothing_weights_normalization - offset
        
        smoothing_weights = [max(n-offset, 0) for n in vertex_weighted_num_neighbors]
        smoothing_weights = [min(n/normalization, 1.0) for n in smoothing_weights]
        smoothing_weights = np.array([6*pow(x,5) - 15*pow(x,4) + 10*pow(x,3) for x in smoothing_weights])
        
        mesh_with_data.push_point_attribute("wnn", vertex_weighted_num_neighbors)
        mesh_with_data.push_point_attribute("sw", smoothing_weights)
    
    # Perform smoothing
    if mesh_smoothing_iters > 0:
        if smoothing_weights is None:
            smoothing_weights = [1.0 for _ in range(len(mesh.vertices))]
        
        pysplashsurf.par_laplacian_smoothing_inplace(mesh_with_data, vertex_connectivity, mesh_smoothing_iters, 1.0, smoothing_weights)
    
    # Compute normals
    if compute_normals:
        
        mesh = mesh_with_data.mesh
        
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
        
    mesh = mesh_with_data.take_mesh()
    
    # Convert triangles to quads
    if generate_quads:
        mesh = pysplashsurf.convert_tris_to_quads(mesh, non_squareness_limit=quad_max_edge_diag_ratio, normal_angle_limit_rad=math.radians(quad_max_normal_angle), max_interior_angle=math.radians(quad_max_interior_angle))
    
    if type(mesh) is pysplashsurf.PyTriMesh3dF64:
        verts, tris = mesh.take_vertices_and_triangles()
        meshio.write_points_cells(output_file, verts, [("triangle", tris)])
        
        # Mesh checks
        if check_mesh_closed or check_mesh_manifold:
            pysplashsurf.check_mesh_consistency(grid, mesh, check_closed=check_mesh_closed, check_manifold=check_mesh_manifold, debug=check_mesh_debug)
        
    else:
        verts, cells = mesh.take_vertices_and_cells()
        cells = [("triangle", list(filter(lambda x: len(x) == 3, cells))), ("quad", list(filter(lambda x: len(x) == 4, cells)))]
        meshio.write_points_cells(output_file, verts, cells)
    
    
    # Left out: Mesh orientation check
    


def test_no_post_processing():
    start = time.time()
    subprocess.run([BINARY_PATH] + "reconstruct ./ParticleData_Fluid_5.vtk -o test_bin.vtk -r=0.025 -l=2.0 -c=0.5 -t=0.6 -d=on --subdomain-grid=on --mesh-cleanup=off --mesh-smoothing-weights=off --mesh-smoothing-iters=0 --normals=off --normals-smoothing-iters=0".split(), check=True)
    print("Binary done in", time.time() - start)
    
    start = time.time()
    reconstruction_pipeline("./ParticleData_Fluid_5.vtk", "test.vtk", particle_radius=np.float64(0.025), smoothing_length=np.float64(2.0), 
                            cube_size=np.float64(0.5), iso_surface_threshold=np.float64(0.6), mesh_smoothing_weights=False, 
                            mesh_smoothing_weights_normalization=np.float64(13.0), mesh_smoothing_iters=0, normals_smoothing_iters=0, 
                            generate_quads=False, mesh_cleanup=False, compute_normals=False, subdomain_grid=True)
    print("Python done in", time.time() - start)
    
    binary_mesh = meshio.read("test_bin.vtk")
    python_mesh = meshio.read("test.vtk")
    
    binary_verts = np.array(binary_mesh.points, dtype=np.float64)
    python_verts = np.array(python_mesh.points, dtype=np.float64)
    
    print("# of vertices binary:", len(binary_verts))
    print("# of vertices python:", len(python_verts))
    
    assert(len(binary_verts) == len(python_verts))
    
    binary_verts.sort(axis=0)
    python_verts.sort(axis=0)
    
    assert(np.allclose(binary_verts, python_verts))
    
def test_with_post_processing():
    start = time.time()
    subprocess.run([BINARY_PATH] + "reconstruct ./ParticleData_Fluid_5.vtk -o test_bin.vtk -r=0.025 -l=2.0 -c=0.5 -t=0.6 -d=on --subdomain-grid=on --decimate-barnacles=on --mesh-cleanup=on --mesh-smoothing-weights=on --mesh-smoothing-iters=25 --normals=on --normals-smoothing-iters=10".split(), check=True)
    print("Binary done in", time.time() - start)
    
    start = time.time()
    reconstruction_pipeline("./ParticleData_Fluid_5.vtk", "test.vtk", particle_radius=np.float64(0.025), smoothing_length=np.float64(2.0), 
                            cube_size=np.float64(0.5), iso_surface_threshold=np.float64(0.6), mesh_smoothing_weights=True, 
                            mesh_smoothing_weights_normalization=np.float64(13.0), mesh_smoothing_iters=25, normals_smoothing_iters=10, 
                            generate_quads=False, mesh_cleanup=True, compute_normals=True, subdomain_grid=True, decimate_barnacles=True)
    print("Python done in", time.time() - start)
    
    binary_mesh = meshio.read("test_bin.vtk")
    python_mesh = meshio.read("test.vtk")
    
    binary_verts = np.array(binary_mesh.points, dtype=np.float64)
    python_verts = np.array(python_mesh.points, dtype=np.float64)
    
    print("# of vertices binary:", len(binary_verts))
    print("# of vertices python:", len(python_verts))
    
    assert(len(binary_verts) == len(python_verts))
    
    binary_verts.sort(axis=0)
    python_verts.sort(axis=0)
    
    print("Binary verts:", binary_verts)
    print("Python verts:", python_verts)
    
    assert(np.allclose(binary_verts, python_verts))

test_with_post_processing()
# test_memory_access()