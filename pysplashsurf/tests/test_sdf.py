import pysplashsurf
import numpy as np


def test_sphere_sdf_mc():
    radius = 1.0
    num_verts = 100

    grid_size = radius * 2.2
    dx = grid_size / (num_verts - 1)

    translation = -0.5 * grid_size

    def make_sdf():
        coords = np.arange(num_verts, dtype=np.float32) * dx + translation
        x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
        sdf = np.sqrt(x**2 + y**2 + z**2) - radius
        return sdf

    sdf = make_sdf()

    # Note: Currently this reconstruction assumes that inside the surface values get bigger (like a density function)
    mesh, grid = pysplashsurf.marching_cubes(
        sdf, cube_size=dx, iso_surface_threshold=0.0, translation=[translation] * 3
    )

    assert len(mesh.vertices) > 0

    norms = np.linalg.norm(mesh.vertices, axis=1)
    assert norms.min() > radius - 1e-4
    assert norms.max() < radius + 1e-4

    assert pysplashsurf.check_mesh_consistency(mesh, grid) is None
