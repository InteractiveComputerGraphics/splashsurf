import pysplashsurf
import numpy as np
import meshio
import pathlib

DIR = pathlib.Path(__file__).parent.resolve()
BGEO_PATH = DIR.joinpath("ParticleData_Fluid_50.bgeo")


def bgeo_test(dtype):
    particles = np.array(meshio.read(BGEO_PATH).points, dtype=dtype)

    assert len(particles) == 4732


def test_bgeo_f32():
    bgeo_test(np.float32)


def test_bgeo_f64():
    bgeo_test(np.float64)
