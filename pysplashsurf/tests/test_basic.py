import pysplashsurf
import numpy as np
import meshio
import os.path
import pathlib
import tempfile

DIR = pathlib.Path(__file__).parent.resolve()
VTK_PATH = DIR.joinpath("ParticleData_Random_1000.vtk")


def test_aabb_class():
    print("\nTesting AABB class")

    aabb = pysplashsurf.Aabb3d.from_min_max(min=[0.0, 0.0, 0.0], max=[1.0, 2.0, 3.0])
    assert (aabb.min == np.array([0.0, 0.0, 0.0])).all()
    assert (aabb.max == np.array([1.0, 2.0, 3.0])).all()

    aabb = pysplashsurf.Aabb3d.from_min_max(
        min=np.array([0.0, 0.0, 0.0]), max=np.array([1.0, 2.0, 3.0])
    )
    assert (aabb.min == np.array([0.0, 0.0, 0.0])).all()
    assert (aabb.max == np.array([1.0, 2.0, 3.0])).all()

    aabb = pysplashsurf.Aabb3d.from_points(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.5, 4.2]])
    )

    print("AABB min:", aabb.min)
    print("AABB max:", aabb.max)

    assert (aabb.min == np.array([0.0, 0.0, 0.0])).all()
    assert (aabb.max == np.array([2.0, 1.0, 4.2])).all()

    assert aabb.contains_point([1.0, 0.9, 4.1])
    assert aabb.contains_point([0.0, 0.0, 0.0])
    assert not aabb.contains_point([2.0, 1.0, 4.2])
    assert not aabb.contains_point([1.0, -1.0, 5.0])


def impl_basic_test(dtype):
    particles = np.array(meshio.read(VTK_PATH).points, dtype=dtype)

    mesh_with_data, reconstruction = pysplashsurf.reconstruction_pipeline(
        particles,
        particle_radius=0.025,
        rest_density=1000.0,
        smoothing_length=2.0,
        cube_size=1.0,
        iso_surface_threshold=0.6,
        mesh_smoothing_iters=5,
        output_mesh_smoothing_weights=True,
    )

    assert type(mesh_with_data) is pysplashsurf.MeshWithData
    assert type(reconstruction) is pysplashsurf.SurfaceReconstruction
    assert type(mesh_with_data.mesh) is pysplashsurf.TriMesh3d

    mesh = mesh_with_data.mesh

    assert mesh_with_data.dtype == mesh.dtype
    assert mesh_with_data.dtype == dtype

    assert type(mesh_with_data.mesh_type) is pysplashsurf.MeshType
    assert mesh_with_data.mesh_type == pysplashsurf.MeshType.Tri3d

    assert mesh.vertices.dtype == dtype
    assert mesh.triangles.dtype in [np.uint32, np.uint64]

    assert mesh_with_data.nvertices == len(mesh.vertices)
    assert mesh_with_data.ncells == len(mesh.triangles)

    assert mesh_with_data.nvertices in range(21000, 25000)
    assert mesh_with_data.ncells in range(45000, 49000)

    assert mesh.vertices.shape == (mesh_with_data.nvertices, 3)
    assert mesh.triangles.shape == (mesh_with_data.ncells, 3)

    assert len(mesh_with_data.point_attributes) == 2
    assert len(mesh_with_data.cell_attributes) == 0

    assert "sw" in mesh_with_data.point_attributes
    assert "wnn" in mesh_with_data.point_attributes

    sw = mesh_with_data.point_attributes["sw"]
    wnn = mesh_with_data.point_attributes["wnn"]

    assert len(sw) == mesh_with_data.nvertices
    assert len(wnn) == mesh_with_data.nvertices

    assert sw.dtype == dtype
    assert wnn.dtype == dtype

    assert sw.shape == (mesh_with_data.nvertices,)
    assert wnn.shape == (mesh_with_data.nvertices,)

    assert sw.min() >= 0.0
    assert sw.max() <= 1.0

    assert wnn.min() >= 0.0


def test_pipeline_f32():
    impl_basic_test(np.float32)


def test_pipeline_f64():
    impl_basic_test(np.float64)


def reconstruct_test(dtype):
    particles = np.array(meshio.read(VTK_PATH).points, dtype=dtype)

    reconstruction = pysplashsurf.reconstruct_surface(
        particles,
        particle_radius=0.025,
        rest_density=1000.0,
        smoothing_length=2.0 * 0.025,
        cube_size=1.0 * 0.025,
        iso_surface_threshold=0.6,
        global_neighborhood_list=True,
    )

    assert type(reconstruction) is pysplashsurf.SurfaceReconstruction
    assert type(reconstruction.mesh) is pysplashsurf.TriMesh3d
    assert type(reconstruction.grid) is pysplashsurf.UniformGrid
    assert type(reconstruction.particle_densities) is np.ndarray
    assert type(reconstruction.particle_inside_aabb) is type(None)
    assert type(reconstruction.particle_neighbors) is pysplashsurf.NeighborhoodLists

    mesh = reconstruction.mesh

    assert mesh.dtype == dtype

    assert reconstruction.particle_densities.dtype == dtype
    assert len(reconstruction.particle_densities) == len(particles)

    assert len(mesh.vertices) in range(25000, 30000)
    assert len(mesh.triangles) in range(49000, 53000)


def test_reconstruct_f32():
    reconstruct_test(np.float32)


def test_reconstruct_f64():
    reconstruct_test(np.float64)


def neighborhood_search_test(dtype):
    particles = np.array(meshio.read(VTK_PATH).points, dtype=dtype)

    reconstruction = pysplashsurf.reconstruct_surface(
        particles,
        particle_radius=0.025,
        rest_density=1000.0,
        smoothing_length=2.0 * 0.025,
        cube_size=1.0 * 0.025,
        iso_surface_threshold=0.6,
        global_neighborhood_list=True,
    )

    neighbors_reconstruct = reconstruction.particle_neighbors.get_neighborhood_lists()

    assert type(neighbors_reconstruct) is list
    assert len(neighbors_reconstruct) == len(particles)

    aabb = reconstruction.grid.aabb

    neighbor_lists = pysplashsurf.neighborhood_search_spatial_hashing_parallel(
        particles, domain=aabb, search_radius=4.0 * 0.025
    )

    assert type(neighbor_lists) is pysplashsurf.NeighborhoodLists

    neighbors = neighbor_lists.get_neighborhood_lists()

    assert type(neighbors) is list
    assert len(neighbors) == len(particles)
    assert len(neighbors) == len(neighbors_reconstruct)

    # TODO: Compare with naive neighbor search


def test_neighborhood_search_f32():
    neighborhood_search_test(np.float32)


def test_neighborhood_search_f64():
    neighborhood_search_test(np.float64)


def check_consistency_test(dtype):
    particles = np.array(meshio.read(VTK_PATH).points, dtype=dtype)

    reconstruction = pysplashsurf.reconstruct_surface(
        particles,
        particle_radius=0.025,
        rest_density=1000.0,
        smoothing_length=2.0 * 0.025,
        cube_size=1.0 * 0.025,
        iso_surface_threshold=0.6,
        global_neighborhood_list=True,
    )
    mesh = reconstruction.mesh

    assert pysplashsurf.check_mesh_consistency(mesh, reconstruction.grid) is None

    mesh_with_data, reconstruction = pysplashsurf.reconstruction_pipeline(
        particles,
        particle_radius=0.025,
        rest_density=1000.0,
        smoothing_length=2.0,
        cube_size=1.0,
        iso_surface_threshold=0.6,
        mesh_smoothing_iters=5,
        output_mesh_smoothing_weights=True,
    )

    assert (
        pysplashsurf.check_mesh_consistency(mesh_with_data, reconstruction.grid) is None
    )

    # TODO: Delete some triangles and check for failure


def test_check_consistency_f32():
    check_consistency_test(np.float32)


def test_check_consistency_f64():
    check_consistency_test(np.float64)


def tris_to_quads_test(dtype):
    particles = np.array(meshio.read(VTK_PATH).points, dtype=dtype)

    mesh_with_data, reconstruction = pysplashsurf.reconstruction_pipeline(
        particles,
        particle_radius=0.025,
        rest_density=1000.0,
        smoothing_length=2.0,
        cube_size=1.0,
        iso_surface_threshold=0.6,
        mesh_smoothing_iters=5,
        output_mesh_smoothing_weights=True,
    )

    mesh_with_data_quads = pysplashsurf.convert_tris_to_quads(mesh_with_data)

    assert type(mesh_with_data_quads.mesh) is pysplashsurf.MixedTriQuadMesh3d
    assert mesh_with_data_quads.mesh_type == pysplashsurf.MeshType.MixedTriQuad3d

    assert mesh_with_data_quads.nvertices == mesh_with_data.nvertices
    assert mesh_with_data_quads.ncells < mesh_with_data.ncells

    tris = mesh_with_data_quads.mesh.get_triangles()
    quads = mesh_with_data_quads.mesh.get_quads()

    assert tris.dtype in [np.uint32, np.uint64]
    assert quads.dtype in [np.uint32, np.uint64]

    assert len(tris) + len(quads) == mesh_with_data_quads.ncells

    assert tris.shape == (len(tris), 3)
    assert quads.shape == (len(quads), 4)

    assert len(tris) in range(35000, 39000)
    assert len(quads) in range(4600, 5000)

    assert len(mesh_with_data.point_attributes) == 2
    assert len(mesh_with_data.cell_attributes) == 0

    assert "sw" in mesh_with_data.point_attributes
    assert "wnn" in mesh_with_data.point_attributes


def test_tris_to_quads_f32():
    tris_to_quads_test(np.float32)


def test_tris_to_quads_f64():
    tris_to_quads_test(np.float64)


def interpolator_test(dtype):
    particles = np.array(meshio.read(VTK_PATH).points, dtype=dtype)

    mesh_with_data, reconstruction = pysplashsurf.reconstruction_pipeline(
        particles,
        particle_radius=0.025,
        rest_density=1000.0,
        smoothing_length=2.0,
        cube_size=1.0,
        iso_surface_threshold=0.6,
        mesh_smoothing_iters=5,
        output_mesh_smoothing_weights=True,
    )

    compact_support = 4.0 * 0.025
    rest_mass = 1000.0 * 0.025**3

    interpolator = pysplashsurf.SphInterpolator(
        particles, reconstruction.particle_densities, rest_mass, compact_support
    )

    assert type(interpolator) is pysplashsurf.SphInterpolator

    mesh = mesh_with_data.mesh
    mesh_densities = interpolator.interpolate_quantity(
        reconstruction.particle_densities, mesh.vertices
    )

    assert type(mesh_densities) is np.ndarray
    assert mesh_densities.dtype == dtype
    assert mesh_densities.shape == (len(mesh.vertices),)
    assert mesh_densities.min() >= 0.0

    mesh_particles = interpolator.interpolate_quantity(particles, mesh.vertices)

    assert type(mesh_particles) is np.ndarray
    assert mesh_particles.dtype == dtype
    assert mesh_particles.shape == (len(mesh.vertices), 3)

    mesh_sph_normals = interpolator.interpolate_normals(mesh.vertices)

    assert type(mesh_sph_normals) is np.ndarray
    assert mesh_sph_normals.dtype == dtype
    assert mesh_sph_normals.shape == (len(mesh.vertices), 3)

    mesh_with_data.add_point_attribute("density", mesh_densities)
    mesh_with_data.add_point_attribute("position", mesh_particles)
    mesh_with_data.add_point_attribute("normal", mesh_sph_normals)

    assert "density" in mesh_with_data.point_attributes
    assert "position" in mesh_with_data.point_attributes
    assert "normal" in mesh_with_data.point_attributes

    assert np.array_equal(mesh_with_data.point_attributes["density"], mesh_densities)
    assert np.array_equal(mesh_with_data.point_attributes["position"], mesh_particles)
    assert np.array_equal(mesh_with_data.point_attributes["normal"], mesh_sph_normals)


def test_interpolator_f32():
    interpolator_test(np.float32)


def test_interpolator_f64():
    interpolator_test(np.float64)
