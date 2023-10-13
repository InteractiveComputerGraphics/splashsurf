# splashsurf_lib
Library for surface reconstruction of SPH particle data

The library is mainly used by the `splashsurf` command-line tool which is [also available](https://crates.io/crates/splashsurf) on crates.io.

For more information about the CLI, check out the [readme in the root of the repository](https://github.com/w1th0utnam3/splashsurf).

**Contents**
- [splashsurf_lib](#splashsurf_lib)
  - [Usage](#usage)
  - [Feature flags](#feature-flags)
  - [The surface reconstruction procedure](#the-surface-reconstruction-procedure)

## Usage

A reconstruction from particle positions is performed using the top-level `reconstruct_surface` function:
```rust
pub fn reconstruct_surface<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    parameters: &Parameters<R>,
) -> Result<SurfaceReconstruction<I, R>, ReconstructionError<I, R>>
```
See the [documentation of the crate](https://docs.rs/splashsurf_lib/latest/splashsurf_lib/) on docs.rs for more information on the usage.

The library re-exports `nalgebra` to avoid version conflicts for users of the library.

## Feature flags

By default none of the following features are enabled to reduce the dependencies introduced by this library. The following feature flags are available for `splashsurf_lib`:
 
 - **vtk-extras**: Enables convenience traits and helper functions to convert the mesh types returned by the library to [`vtkio`](https://crates.io/crates/vtkio) data structures (in particular [`UnstructuredGridPiece`](https://docs.rs/vtkio/latest/vtkio/model/struct.UnstructuredGridPiece.html)) that can be used to easily write the meshes to VTK files (e.g. for viewing them with [Paraview](https://www.paraview.org/)). Check out the documentation of `vtkio` or the [corresponding io module](https://github.com/w1th0utnam3/splashsurf/blob/main/splashsurf/src/io/vtk_format.rs) of the `splashsurf` CLI for reference.
 - **profiling**: Enables profiling of the library using [`coarse-prof`](https://crates.io/crates/coarse-prof). Several functions in the library will use the [`profile!`](https://docs.rs/coarse-prof/latest/coarse_prof/macro.profile.html) macro with the function name as an argument to record their runtime. The user of the library can then obtain the profiling data using the functions provided by the `coarse-prof` crate. Note that profiling using this crate might reduce performance for surface reconstructions with a very small number of particles (i.e. only a few hundred).

For each of the features, `splashsurf_lib` re-exports the corresponding dependencies to avoid version conflicts for users of the library.

## The surface reconstruction procedure

Currently, only one method based on a "spatial hashing" strategy is implemented.

`TODO: This section is missing a description of the domain decomposition for more efficient parallelization`

**Short summary**: The fluid density is evaluated or mapped onto a sparse grid using spatial hashing in the support radius of each fluid particle. This implies that memory is only allocated in areas where the fluid density is non-zero. This is in contrast to a naive approach where the marching cubes background grid is allocated for the whole domain. Finally, the marching cubes reconstruction is performed only in those grid cells where an edge crosses the surface threshold. Cells completely in the interior of the fluid are skipped in the marching cubes phase.

**Individual steps**:
 1. Construct a "virtual background grid" with the desired resolution of the marching cubes algorithm. In the end, the procedure will place a single surface mesh vertex on each edge of this virtual grid, where the fluid surface crosses the edge (or rather, where the fluid density crosses the specified threshold). Virtual means that no storage is actually allocated for this grid yet; only its topology is used implicitly later.
 2. Compute the density of each fluid particle
    - Perform a neighborhood search
    - Per particle, evaluate an SPH sum over the neighbors to compute its density (based on input parameters of kernel radius and particle rest mass)
 3. Optional: filter out (or rather mask as inactive) single particles if the user provided a "splash detection radius". This is done by performing an additional neighborhood search using this splash detection radius instead of the kernel radius.
 4. Compute a "sparse density map": a map from the index of a vertex of the virtual background grid to the corresponding fluid density value. The map will only contain entries for vertices where the fluid density is non-zero. Construction of the map:
    - Iterate over all active particles
    - For each particle evaluate its kernel at all virtual background vertices that can be influenced by it (i.e. vertices inside its kernel radius)
    - Add-assign the corresponding density contribution (kernel value times particle density) to the vertex entry in the density map
 5. Interpolate the density values at the vertices of each virtual background cell to points on the edges where the edge crosses the fluid surface
    - Iterate over all vertices in the sparse density map
    - Skip vertices where the density value is above the threshold to be considered inside of the fluid
    - For each of the remaining vertices, check if any neighboring vertex (i.e. a vertex that is directly connected with it by an edge of the virtual background grid) is above the threshold
    - If this is the case, an edge that crosses the fluid surface was found and the position of the surface on the edge can be calculated using linear interpolation
    - This interpolated position is stored in a new map that maps indices of virtual background cells to structs containing the surface crossing points per edge. Entries are only created for cells with edges that actually cross the surface
 6. Triangulate the points on the edges using a marching cubes case table
    - Iterate over all cells in the cell data map
    - For each cell, look up the corresponding triangulation in a marching cubes LUT
    - Emit the required triangles into the final mesh data structure
