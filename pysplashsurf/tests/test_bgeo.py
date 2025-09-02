import pysplashsurf
import numpy as np
import meshio
import pathlib

DIR = pathlib.Path(__file__).parent.resolve()
BGEO_PATH = DIR.joinpath("ParticleData_Fluid_50.bgeo")


def test_bgeo():
    particles = np.array(meshio.read(BGEO_PATH).points, dtype=np.float32)

    assert len(particles) == 4732
