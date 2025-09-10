import pysplashsurf
import numpy as np
import meshio
import subprocess
import time
import trimesh
import pathlib

BINARY_PATH = "splashsurf"
DIR = pathlib.Path(__file__).parent.resolve()
VTK_PATH = DIR.joinpath("ParticleData_Fluid_5.vtk")


def now_s():
    return time.process_time_ns() / (10**9)


def marching_cubes_calls(dtype):
    print("\nTesting marching cubes calls")

    particles = np.array(meshio.read(VTK_PATH).points, dtype=dtype)
    reconstruction = pysplashsurf.reconstruct_surface(
        particles,
        particle_radius=0.025,
        rest_density=1000.0,
        smoothing_length=2.0,
        cube_size=0.5,
        iso_surface_threshold=0.6,
    )
    mesh = reconstruction.mesh
    verts_before = len(mesh.vertices)
    print("# of vertices before:", verts_before)

    mesh_with_data = pysplashsurf.MeshWithData(mesh)
    pysplashsurf.marching_cubes_cleanup(mesh_with_data, reconstruction.grid)

    mesh = mesh_with_data.mesh
    verts_after = len(mesh.vertices)
    print("# of vertices after:", verts_after)
    assert verts_after < verts_before


def test_marching_cubes_calls_f32():
    marching_cubes_calls(np.float32)


def test_marching_cubes_calls_f64():
    marching_cubes_calls(np.float64)


def reconstruction_pipeline(
    input_file,
    output_file,
    *,
    attributes_to_interpolate=None,
    multi_threading=True,
    particle_radius=0.025,
    rest_density=1000.0,
    smoothing_length=2.0,
    cube_size=0.5,
    iso_surface_threshold=0.6,
    mesh_smoothing_weights=False,
    output_mesh_smoothing_weights=False,
    sph_normals=False,
    mesh_smoothing_weights_normalization=13.0,
    mesh_smoothing_iters=5,
    normals_smoothing_iters=5,
    mesh_aabb_min=None,
    mesh_aabb_max=None,
    mesh_cleanup=False,
    decimate_barnacles=False,
    keep_vertices=False,
    compute_normals=False,
    output_raw_normals=False,
    output_raw_mesh=False,
    mesh_aabb_clamp_vertices=False,
    check_mesh_closed=False,
    check_mesh_manifold=False,
    check_mesh_orientation=False,
    check_mesh_debug=False,
    generate_quads=False,
    quad_max_edge_diag_ratio=1.75,
    quad_max_normal_angle=10.0,
    quad_max_interior_angle=135.0,
    subdomain_grid=False,
    subdomain_num_cubes_per_dim=64,
    dtype
):
    mesh = meshio.read(input_file)
    particles = np.array(mesh.points, dtype=dtype)

    if attributes_to_interpolate is None:
        attributes_to_interpolate = []

    # Prepare attributes dictionary
    attrs = {}
    for attr in attributes_to_interpolate:
        if attr in mesh.point_data:
            if mesh.point_data[attr].dtype.kind == "f":
                attrs[attr] = mesh.point_data[attr].astype(dtype)
            else:
                attrs[attr] = mesh.point_data[attr].astype(np.int64)

    mesh_with_data, reconstruction = pysplashsurf.reconstruction_pipeline(
        particles,
        attributes_to_interpolate=attrs,
        multi_threading=multi_threading,
        particle_radius=particle_radius,
        rest_density=rest_density,
        smoothing_length=smoothing_length,
        cube_size=cube_size,
        iso_surface_threshold=iso_surface_threshold,
        mesh_smoothing_weights=mesh_smoothing_weights,
        sph_normals=sph_normals,
        mesh_smoothing_weights_normalization=mesh_smoothing_weights_normalization,
        mesh_smoothing_iters=mesh_smoothing_iters,
        normals_smoothing_iters=normals_smoothing_iters,
        mesh_aabb_min=mesh_aabb_min,
        mesh_aabb_max=mesh_aabb_max,
        mesh_cleanup=mesh_cleanup,
        decimate_barnacles=decimate_barnacles,
        keep_vertices=keep_vertices,
        compute_normals=compute_normals,
        output_raw_normals=output_raw_normals,
        output_raw_mesh=output_raw_mesh,
        mesh_aabb_clamp_vertices=mesh_aabb_clamp_vertices,
        subdomain_grid=subdomain_grid,
        subdomain_num_cubes_per_dim=subdomain_num_cubes_per_dim,
        output_mesh_smoothing_weights=output_mesh_smoothing_weights,
        check_mesh_closed=check_mesh_closed,
        check_mesh_manifold=check_mesh_manifold,
        check_mesh_orientation=check_mesh_orientation,
        check_mesh_debug=check_mesh_debug,
        generate_quads=generate_quads,
        quad_max_edge_diag_ratio=quad_max_edge_diag_ratio,
        quad_max_normal_angle=quad_max_normal_angle,
        quad_max_interior_angle=quad_max_interior_angle,
    )

    mesh_with_data.write_to_file(output_file)


def no_post_processing_test(dtype):
    start = now_s()
    subprocess.run(
        [BINARY_PATH]
        + f"reconstruct {VTK_PATH} -o {DIR.joinpath('test_bin.vtk')} -r=0.025 -l=2.0 -c=0.5 -t=0.6 {"-d=on" if dtype == np.float64 else ""} --subdomain-grid=on --mesh-cleanup=off --mesh-smoothing-weights=off --mesh-smoothing-iters=0 --normals=off --normals-smoothing-iters=0".split(),
        check=True,
    )
    print("Binary done in", now_s() - start)

    start = now_s()
    reconstruction_pipeline(
        VTK_PATH,
        DIR.joinpath("test.vtk"),
        particle_radius=0.025,
        smoothing_length=2.0,
        cube_size=0.5,
        iso_surface_threshold=0.6,
        mesh_smoothing_weights=False,
        mesh_smoothing_iters=0,
        normals_smoothing_iters=0,
        mesh_cleanup=False,
        compute_normals=False,
        subdomain_grid=True,
        dtype=dtype
    )
    print("Python done in", now_s() - start)

    binary_mesh = meshio.read(DIR.joinpath("test_bin.vtk"))
    python_mesh = meshio.read(DIR.joinpath("test.vtk"))

    binary_verts = np.array(binary_mesh.points, dtype=dtype)
    python_verts = np.array(python_mesh.points, dtype=dtype)

    print("# of vertices binary:", len(binary_verts))
    print("# of vertices python:", len(python_verts))

    assert len(binary_verts) == len(python_verts)

    binary_verts.sort(axis=0)
    python_verts.sort(axis=0)

    assert np.allclose(binary_verts, python_verts)


def test_no_post_processing_f32():
    no_post_processing_test(np.float32)


def test_no_post_processing_f64():
    no_post_processing_test(np.float64)


def with_post_processing_test(dtype):
    start = now_s()
    subprocess.run(
        [BINARY_PATH]
        + f"reconstruct {VTK_PATH} -o {DIR.joinpath('test_bin.vtk')} -r=0.025 -l=2.0 -c=0.5 -t=0.6 {"-d=on" if dtype == np.float64 else ""} --subdomain-grid=on --interpolate_attribute velocity --decimate-barnacles=on --mesh-cleanup=on --mesh-smoothing-weights=on --mesh-smoothing-iters=25 --normals=on --normals-smoothing-iters=10 --output-smoothing-weights=on --generate-quads=off".split(),
        check=True,
    )
    print("Binary done in", now_s() - start)

    start = now_s()
    reconstruction_pipeline(
        VTK_PATH,
        DIR.joinpath("test.vtk"),
        attributes_to_interpolate=["velocity"],
        particle_radius=0.025,
        smoothing_length=2.0,
        cube_size=0.5,
        iso_surface_threshold=0.6,
        mesh_smoothing_weights=True,
        mesh_smoothing_weights_normalization=13.0,
        mesh_smoothing_iters=25,
        normals_smoothing_iters=10,
        generate_quads=False,
        mesh_cleanup=True,
        compute_normals=True,
        subdomain_grid=True,
        decimate_barnacles=True,
        output_mesh_smoothing_weights=True,
        output_raw_normals=True,
        dtype=dtype
    )
    print("Python done in", now_s() - start)

    binary_mesh = meshio.read(DIR.joinpath("test_bin.vtk"))
    python_mesh = meshio.read(DIR.joinpath("test.vtk"))

    # Compare number of vertices
    binary_verts = np.array(binary_mesh.points, dtype=dtype)
    python_verts = np.array(python_mesh.points, dtype=dtype)

    print("# of vertices binary:", len(binary_verts))
    print("# of vertices python:", len(python_verts))

    assert len(binary_verts) == len(python_verts)

    # Compare interpolated attribute
    binary_vels = binary_mesh.point_data["velocity"]
    python_vels = python_mesh.point_data["velocity"]

    binary_vels.sort(axis=0)
    python_vels.sort(axis=0)

    assert np.allclose(binary_vels, python_vels)

    # Trimesh similarity test
    # TODO: Replace load_mesh call: the function tries to create temporary files which may fail on some CI runners
    binary_mesh = trimesh.load_mesh(DIR.joinpath("test_bin.vtk"), "vtk")
    python_mesh = trimesh.load_mesh(DIR.joinpath("test.vtk"), "vtk")

    (_, distance_bin, _) = trimesh.proximity.closest_point(binary_mesh, python_verts)
    (_, distance_py, _) = trimesh.proximity.closest_point(python_mesh, binary_verts)
    distance = (np.sum(distance_bin) + np.sum(distance_py)) / (
        len(distance_bin) + len(python_verts)
    )
    print("Distance:", distance)
    assert distance < 1e-5

    # Naïve similarity test

    binary_verts.sort(axis=0)
    python_verts.sort(axis=0)

    print("Binary verts:", binary_verts)
    print("Python verts:", python_verts)

    assert np.allclose(binary_verts, python_verts)


def test_with_post_processing_f32():
    with_post_processing_test(np.float32)


def test_with_post_processing_f64():
    with_post_processing_test(np.float64)
