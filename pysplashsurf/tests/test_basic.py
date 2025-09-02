import pysplashsurf
import numpy as np
import meshio


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
