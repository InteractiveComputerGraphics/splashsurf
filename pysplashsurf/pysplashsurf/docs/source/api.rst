API Overview
============

.. currentmodule:: pysplashsurf

The main functionality of ``pysplashsurf`` is provided by the :py:func:`reconstruction_pipeline` function which implements all features of the ``splashsurf`` CLI including the surface reconstruction from particles and optional post-processing, and the :py:func:`reconstruct_surface` function which only performs the surface reconstruction itself.

**Data types:** The functions of the package accept Python ``float`` for scalar parameters and Numpy arrays of data-type ``np.float32`` or ``np.float64`` for array inputs (e.g. particle positions).
Outputs will be of the same float precision as the input arrays.
Array-like inputs have to be contiguous (C-order) in memory.
All array-like and object type (e.g. :py:class:`Aabb3d`) inputs to a function call have to use the same float data-type.

Functions
---------

.. autosummary::
    barnacle_decimation
    check_mesh_consistency
    convert_tris_to_quads
    laplacian_smoothing_normals_parallel
    laplacian_smoothing_parallel
    marching_cubes
    marching_cubes_cleanup
    neighborhood_search_spatial_hashing_parallel
    reconstruct_surface
    reconstruction_pipeline

Classes
-------

.. autosummary::
    Aabb3d
    MeshAttribute
    MeshWithData
    MixedTriQuadMesh3d
    NeighborhoodLists
    SphInterpolator
    SurfaceReconstruction
    TriMesh3d
    UniformGrid
    VertexVertexConnectivity

Enums
-----

.. autosummary::
    MeshType
