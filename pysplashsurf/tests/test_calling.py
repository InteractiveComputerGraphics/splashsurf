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
                            mesh_aabb_min=None, mesh_aabb_max=None, mesh_cleanup=False, decimate_barnacles=False, keep_vertices=False,
                            compute_normals=False, output_raw_normals=False, mesh_aabb_clamp_vertices=False,
                            check_mesh_closed=True, check_mesh_manifold=True, check_mesh_debug=False,
                            generate_quads=False, quad_max_edge_diag_ratio=1.75, quad_max_normal_angle=10.0, quad_max_interior_angle=135.0,
                            subdomain_grid=False, subdomain_num_cubes_per_dim=64):
    
    particles = np.array(meshio.read(input_file).points, dtype=np.float64)
    
    mesh_with_data = pysplashsurf.reconstruction_pipeline(particles, enable_multi_threading=enable_multi_threading, particle_radius=particle_radius,
                                                          rest_density=rest_density, smoothing_length=smoothing_length, cube_size=cube_size, iso_surface_threshold=iso_surface_threshold,
                                                          mesh_smoothing_weights=mesh_smoothing_weights, sph_normals=sph_normals,
                                                          mesh_smoothing_weights_normalization=mesh_smoothing_weights_normalization,
                                                          mesh_smoothing_iters=mesh_smoothing_iters, normals_smoothing_iters=normals_smoothing_iters,
                                                          mesh_aabb_min=mesh_aabb_min, mesh_aabb_max=mesh_aabb_max, mesh_cleanup=mesh_cleanup, decimate_barnacles=decimate_barnacles,
                                                          keep_vertices=keep_vertices, compute_normals=compute_normals, output_raw_normals=output_raw_normals,
                                                          mesh_aabb_clamp_vertices=mesh_aabb_clamp_vertices, subdomain_grid=subdomain_grid, subdomain_num_cubes_per_dim=subdomain_num_cubes_per_dim)
        
    mesh = mesh_with_data.take_mesh()
    
    # Convert triangles to quads
    if generate_quads:
        mesh = pysplashsurf.convert_tris_to_quads(mesh, non_squareness_limit=quad_max_edge_diag_ratio, normal_angle_limit_rad=math.radians(quad_max_normal_angle), max_interior_angle=math.radians(quad_max_interior_angle))
    
    if type(mesh) is pysplashsurf.PyTriMesh3dF64:
        verts, tris = mesh.take_vertices_and_triangles()
        meshio.write_points_cells(output_file, verts, [("triangle", tris)])
        
        # Mesh checks
        # if check_mesh_closed or check_mesh_manifold:
        #     pysplashsurf.check_mesh_consistency(grid, mesh, check_closed=check_mesh_closed, check_manifold=check_mesh_manifold, debug=check_mesh_debug)
        
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
