from .pysplashsurf import *

def marching_cubes_cleanup(mesh, grid, max_iter=5, keep_vertices=False):
    if type(mesh) is PyTriMesh3dF32:
        print("F32")
        return marching_cubes_cleanup_f32(mesh, grid, max_iter=max_iter, keep_vertices=keep_vertices)
    elif type(mesh) is PyTriMesh3dF64:
        print("F64")
        return marching_cubes_cleanup_f64(mesh, grid, max_iter=max_iter, keep_vertices=keep_vertices)
    else:
        raise ValueError("Invalid mesh type")