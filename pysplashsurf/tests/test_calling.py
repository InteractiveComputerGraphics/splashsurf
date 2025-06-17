import pysplashsurf
import numpy as np
import math
import meshio
import subprocess
import time
import trimesh
import pathlib

BINARY_PATH = "splashsurf"
DIR = pathlib.Path(__file__).parent.resolve()
BGEO_PATH = DIR.joinpath("../ParticleData_Fluid_50.bgeo")
VTK_PATH = DIR.joinpath("../ParticleData_Fluid_5.vtk")


def test_bgeo():
    particles = np.array(meshio.read(BGEO_PATH).points, dtype=np.float32)
    
    assert(len(particles) == 4732)

def test_aabb_class():
    print("\nTesting AABB class")
    
    aabb = pysplashsurf.Aabb3dF64.par_from_points(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.5, 4.2]]))
    
    assert(aabb.min() == np.array([0.0, 0.0, 0.0])).all()
    assert(aabb.max() == np.array([2.0, 1.0, 4.2])).all()
    
    aabb.join_with_point([3.0, 2.0, 1.0])
    
    assert(aabb.min() == np.array([0.0, 0.0, 0.0])).all()
    assert(aabb.max() == np.array([3.0, 2.0, 4.2])).all()
    
    assert(aabb.contains_point([1.0, 1.0, 4.1]))
    assert(aabb.contains_point([0.0, 0.0, 0.0]))
    assert(not aabb.contains_point([4.0, 2.0, 1.0]))
    assert(not aabb.contains_point([1.0, -1.0, 5.0]))

def test_marching_cubes_calls():
    print("\nTesting marching cubes calls")
    
    particles = np.array(meshio.read(VTK_PATH).points, dtype=np.float32)
    reconstruction = pysplashsurf.reconstruct_surface(particles, enable_multi_threading=True, particle_radius=0.025,
                                                                 rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                                                                 iso_surface_threshold=0.6)
    mesh = reconstruction.mesh
    verts_before = len(mesh.vertices)
    print("# of vertices before:", verts_before)
    
    mesh_with_data = pysplashsurf.create_mesh_with_data_object(mesh)
    pysplashsurf.marching_cubes_cleanup(mesh_with_data, reconstruction.grid)
    
    verts_after = len(mesh_with_data.mesh.vertices)
    print("# of vertices after:", verts_after)
    assert(verts_after < verts_before)

def test_memory_access():
    print("\nTesting memory copy vs take")
    
    particles = np.array(meshio.read(VTK_PATH).points, dtype=np.float64)
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

def reconstruction_pipeline(input_file, output_file, *, attributes_to_interpolate=[], enable_multi_threading=True, particle_radius=0.025, 
                            rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, 
                            iso_surface_threshold=0.6, mesh_smoothing_weights=False, output_mesh_smoothing_weights=False, sph_normals=False, 
                            mesh_smoothing_weights_normalization=13.0, mesh_smoothing_iters=5, normals_smoothing_iters=5,
                            mesh_aabb_min=None, mesh_aabb_max=None, mesh_cleanup=False, decimate_barnacles=False, keep_vertices=False,
                            compute_normals=False, output_raw_normals=False, output_raw_mesh=False, mesh_aabb_clamp_vertices=False,
                            check_mesh_closed=False, check_mesh_manifold=False, check_mesh_orientation=False, check_mesh_debug=False,
                            generate_quads=False, quad_max_edge_diag_ratio=1.75, quad_max_normal_angle=10.0, quad_max_interior_angle=135.0,
                            subdomain_grid=False, subdomain_num_cubes_per_dim=64):
    
    mesh = meshio.read(input_file)
    particles = np.array(mesh.points, dtype=np.float64)
    
    attrs = {}
    for attr in attributes_to_interpolate:
        if attr in mesh.point_data:
            if mesh.point_data[attr].dtype.kind == 'f':
                attrs[attr] = mesh.point_data[attr].astype(np.float64)
            else:
                attrs[attr] = mesh.point_data[attr].astype(np.int64)
    
    mesh_with_data, reconstruction = pysplashsurf.reconstruction_pipeline(particles, attributes_to_interpolate=attrs, enable_multi_threading=enable_multi_threading, particle_radius=particle_radius,
                                                          rest_density=rest_density, smoothing_length=smoothing_length, cube_size=cube_size, iso_surface_threshold=iso_surface_threshold,
                                                          mesh_smoothing_weights=mesh_smoothing_weights, sph_normals=sph_normals,
                                                          mesh_smoothing_weights_normalization=mesh_smoothing_weights_normalization,
                                                          mesh_smoothing_iters=mesh_smoothing_iters, normals_smoothing_iters=normals_smoothing_iters,
                                                          mesh_aabb_min=mesh_aabb_min, mesh_aabb_max=mesh_aabb_max, mesh_cleanup=mesh_cleanup, decimate_barnacles=decimate_barnacles,
                                                          keep_vertices=keep_vertices, compute_normals=compute_normals, output_raw_normals=output_raw_normals, output_raw_mesh=output_raw_mesh,
                                                          mesh_aabb_clamp_vertices=mesh_aabb_clamp_vertices, subdomain_grid=subdomain_grid, subdomain_num_cubes_per_dim=subdomain_num_cubes_per_dim, output_mesh_smoothing_weights=output_mesh_smoothing_weights,
                                                          check_mesh_closed=check_mesh_closed, check_mesh_manifold=check_mesh_manifold, check_mesh_orientation=check_mesh_orientation, check_mesh_debug=check_mesh_debug,
                                                          generate_quads=generate_quads, quad_max_edge_diag_ratio=quad_max_edge_diag_ratio, quad_max_normal_angle=quad_max_normal_angle, quad_max_interior_angle=quad_max_interior_angle)

    pysplashsurf.write_to_file(mesh_with_data, output_file, consume_object=True)


def test_no_post_processing():
    start = time.time()
    subprocess.run([BINARY_PATH] + f"reconstruct {VTK_PATH} -o {DIR.joinpath("test_bin.vtk")} -r=0.025 -l=2.0 -c=0.5 -t=0.6 -d=on --subdomain-grid=on --mesh-cleanup=off --mesh-smoothing-weights=off --mesh-smoothing-iters=0 --normals=off --normals-smoothing-iters=0".split(), check=True)
    print("Binary done in", time.time() - start)
    
    start = time.time()
    reconstruction_pipeline(VTK_PATH, DIR.joinpath("test.vtk"), particle_radius=np.float64(0.025), smoothing_length=np.float64(2.0), 
                            cube_size=np.float64(0.5), iso_surface_threshold=np.float64(0.6), mesh_smoothing_weights=False, 
                            mesh_smoothing_iters=0, normals_smoothing_iters=0, mesh_cleanup=False, compute_normals=False, subdomain_grid=True)
    print("Python done in", time.time() - start)
    
    binary_mesh = meshio.read(DIR.joinpath("test_bin.vtk"))
    python_mesh = meshio.read(DIR.joinpath("test.vtk"))
    
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
    subprocess.run([BINARY_PATH] + f"reconstruct {VTK_PATH} -o {DIR.joinpath("test_bin.vtk")} -r=0.025 -l=2.0 -c=0.5 -t=0.6 -d=on --subdomain-grid=on --interpolate_attribute velocity --decimate-barnacles=on --mesh-cleanup=on --mesh-smoothing-weights=on --mesh-smoothing-iters=25 --normals=on --normals-smoothing-iters=10 --output-smoothing-weights=on --generate-quads=off".split(), check=True)
    print("Binary done in", time.time() - start)
    
    start = time.time()
    reconstruction_pipeline(VTK_PATH, DIR.joinpath("test.vtk"), attributes_to_interpolate=["velocity"], particle_radius=np.float64(0.025), smoothing_length=np.float64(2.0), 
                            cube_size=np.float64(0.5), iso_surface_threshold=np.float64(0.6), mesh_smoothing_weights=True, 
                            mesh_smoothing_weights_normalization=np.float64(13.0), mesh_smoothing_iters=25, normals_smoothing_iters=10, 
                            generate_quads=False, mesh_cleanup=True, compute_normals=True, subdomain_grid=True, decimate_barnacles=True,
                            output_mesh_smoothing_weights=True, output_raw_normals=True)
    print("Python done in", time.time() - start)
    
    binary_mesh = meshio.read(DIR.joinpath("test_bin.vtk"))
    python_mesh = meshio.read(DIR.joinpath("test.vtk"))
    
    # Compare number of vertices
    binary_verts = np.array(binary_mesh.points, dtype=np.float64)
    python_verts = np.array(python_mesh.points, dtype=np.float64)
    
    print("# of vertices binary:", len(binary_verts))
    print("# of vertices python:", len(python_verts))
    
    assert(len(binary_verts) == len(python_verts))
    
    # Compare interpolated attribute
    binary_vels = binary_mesh.point_data["velocity"]
    python_vels = python_mesh.point_data["velocity"]
    
    binary_vels.sort(axis=0)
    python_vels.sort(axis=0)
    
    assert(np.allclose(binary_vels, python_vels))
    
    # Trimesh similarity test
    binary_mesh = trimesh.load_mesh(DIR.joinpath("test_bin.vtk"), "vtk")
    python_mesh = trimesh.load_mesh(DIR.joinpath("test.vtk"), "vtk")
    
    (_, distance_bin, _) = trimesh.proximity.closest_point(binary_mesh, python_verts)
    (_, distance_py, _) = trimesh.proximity.closest_point(python_mesh, binary_verts)
    distance = (np.sum(distance_bin) + np.sum(distance_py)) / (len(distance_bin) + len(python_verts))
    print("Distance:", distance)
    assert(distance < 1e-5)
    
    # Naïve similarity test
    
    binary_verts.sort(axis=0)
    python_verts.sort(axis=0)
    
    print("Binary verts:", binary_verts)
    print("Python verts:", python_verts)
    
    assert(np.allclose(binary_verts, python_verts))

# test_bgeo()
# test_aabb_class()
# test_marching_cubes_calls()
# test_memory_access()
# test_with_post_processing()
