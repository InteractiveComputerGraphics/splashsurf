# splashsurf_lib
Library for surface reconstruction of SPH particle data

The library is mainly used by the `splashsurf` command-line tool which is [also available](https://crates.io/crates/splashsurf) on crates.io.

This readme is currently a placeholder that will be extended over time. 
For now, check the [GitHub repository](https://github.com/w1th0utnam3/splashsurf) for more information.

## Usage

*TODO: Describe main "entry points" of the library, parameters, inputs/outputs etc.*

The library re-exports `nalgebra` to avoid version conflicts for users of the library.

## Feature flags

By default none of the following features are enabled to reduce the dependencies introduced by this library. The following feature flags are available for `splashsurf_lib`:
 
 - **vtk-extras**: Enables convenience traits and helper functions to convert the mesh types returned by the library to [`vtkio`](https://crates.io/crates/vtkio) data structures (in particular [`UnstructuredGridPiece`](https://docs.rs/vtkio/latest/vtkio/model/struct.UnstructuredGridPiece.html)) that can be used to easily write the meshes to VTK files (e.g. for viewing them with [Paraview](https://www.paraview.org/)). Check out the documentation of `vtkio` or the [corresponding io module](https://github.com/w1th0utnam3/splashsurf/blob/master/splashsurf/src/io/vtk_format.rs) of the `splashsurf` CLI for reference.
 - **profiling**: Enables profiling of the library using [`coarse-prof`](https://crates.io/crates/coarse-prof). Several functions in the library will use the [`profile!`](https://docs.rs/coarse-prof/latest/coarse_prof/macro.profile.html) macro with the function name as an argument to record their runtime. The user of the library can then obtain the profiling data using the functions provided by the `coarse-prof` crate. Note that profiling using this crate might reduce performance for surface reconstructions with a very small number of particles (i.e. only a few hundred).

For each of the features, `splashsurf_lib` re-exports the corresponding dependencies to avoid version conflicts for users of the library.

## The surface reconstruction procedure

*TODO: Describe the surface reconstruction "algorithm" implemented in this crate*
