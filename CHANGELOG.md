## Master

## Version 0.7.0

 - Lib: Fix a bug that caused the neighborhood search results to be incomplete
 - Lib: Update to `nalgebra` 0.27. with const generics and removed usage of `nalgebra` types that are deprecated
 - Lib: Add a "naive" `O(n^2)` neighborhood search for testing purposes
 - Lib: Rename other neighborhood search functions to `neighborhood_search_spatial_hashing` and `neighborhood_search_spatial_hashing_parallel`
 - CLI: Add a command line option `--output-normals` to the `reconstruct` sub-command to write mesh normals to the output file. Note that currently the normals are only computed using an area weighted average of triangle normals.
 - CLI: Change the `convert` sub-command to allow both conversion of particle files and conversion of mesh formats. Instead of using the `--input` option you now have to either use the `--particles` or `--mesh` option to specify the input file depending on whether you want to convert a particle or mesh file. For particles `VTK, BGEO, PLY, XYZ, JSON -> VTK, PLY` is supported. For meshes only `VTK, PLY -> OBJ` is supported.
 - CLI: Add JSON files as supported particle input file
 - CLI: Change the command line option to specify a sequence of input files from `--input-sequence-pattern` to `--input-sequence`

## Version 0.6.1

 - Lib: Add docs.rs metadata to manifest to document all features and to build only for a single target

## Version 0.6.0

This release fixes a couple of bugs that may lead to inconsistent surface reconstructions when using domain decomposition (i.e. reconstructions with artificial bumps exactly at the subdomain boundaries, especially on flat surfaces). Currently there are no other known bugs and the domain decomposed approach appears to be really fast and robust.

In addition the CLI now reports more detailed timing statistics for multi-threaded reconstructions.

Otherwise this release contains just some small changes to command line parameters.

 - Lib: Add a `ParticleDensityComputationStrategy` enum to the `SpatialDecompositionParameters` struct. In order for domain decomposition to work consistently, the per particle densities have to be evaluated to a consistent value between domains. This is especially important for the ghost particles. Previously, this resulted inconsistent density values on boundaries if the ghost particle margin was not at least 2x the compact support radius (as this ensures that the inner ghost particles actually have the correct density). This option is now still available as the `IndependentSubdomains` strategy. The preferred way, that avoids the 2x ghost particle margin is the `SynchronizeSubdomains` where the density values of the particles in the subdomains are first collected into a global storage. This can be faster as the previous method as this avoids having to collect a double-width ghost particle layer. In addition there is the "playing it safe" option, the `Global` strategy, where the particle densities are computed in a completely global step before any domain decomposition. This approach however is *really* slow for large quantities of particles. For more information, read the documentation on the `ParticleDensityComputationStrategy` enum.
 - Lib: Fix bug where the workspace storage was not cleared correctly leading to inconsistent results depending on the sequence of processed subdomains
 - Lib: Fix bug in the octree splitting where ghost particles of a parent node were actually classified as non-ghost particles in the child node, leading to many subsequent splits
 - Lib: Change AABB's `contains_point` method such that it now considers the AABB as "open" to its max value, i.e. it checks if the point is in the half-open set `[min, max[`
 - Lib: Implemented an own `profile!` macro that also works in multi-threaded code, i.e. with together with `rayon`
 - CLI: The CLI now also prints detailed profiling/timing output when running in parallel with domain decomposition thanks to the new `profile` macro
 - CLI: Add `--domain-min` and `--domain-max` flags to the `convert` sub-command that allows to filter out particles
 - CLI: Remove the `--splash-detection-radius` flag as it did not work for a couple of releases
 - Lib: Update to [`vtkio`](https://github.com/elrnv/vtkio) 0.6.0

## Version 0.5.1

 - Lib: Fixed bug in triangulation that resulted in at most 4 out 5 possible triangles being emitted. This resulted in holes in surface meshes in some cases.

## Version 0.5.0

The biggest new feature is a domain decomposed approach for the surface reconstruction by performing a spatial decomposition of the particle set with an octree.
The resulting local patches can then be processed in parallel (leaving a single layer of boundary cells per patch untriangulated to avoid incompatible boundaries).
Afterwards, a stitching procedure walks the octree back upwards and merges the octree leaves by averaging density values on the boundaries. 
As the library uses task based parallelism, a task for stitching can be enqueued as soon as all children of an octree node are processed.
Depending on the number of available threads and the particle data, this approach results in a speedup of 4-10x in comparison to the global parallel approach in selected benchmarks.
At the moment, this domain decomposition approach is only available when allowing to parallelize over particles using the `--mt-particles` flag.

 - CLI: The CLI now uses subcommands, the reconstruction is now run using the `reconstruct` subcommand
 - CLI: Added a `convert` subcommand to convert between particle formats. All supported input formats can be read but only output to VTK is currently supported.
 - CLI: Add flags to enable/disable spatial decomposition and control octree usage
 - CLI: Add flag to control number of threads for the worker thread pool
 - CLI: Add flags for quiet/verbose output
 - CLI: Use `on/off` switches for various command line parameters instead of boolean flags that can only be used to set something to `true`
 - CLI: Support for reading uncompressed BGEO files (previously only compressed BGEO files were supported)
 - Lib: Implement domain decomposed reconstruction approach using octree spatial decomposition, the leaves of the octree are processed in parallel.
 - Lib: Implement stitching of local meshes resulting from the domain decomposed reconstruction. 
   The resulting meshes appear to be closed (without boundary) in limited testing with a couple of examples if the old global implementation also resulted also in a closed mesh.
 - Lib: Introduce thread local "workspaces" to reuse allocated memory. The workspace will be stored in the `SurfaceReconstruction` result object and can be reused to reduce allocations in subsequent surface reconstructions.
 - Lib: Fix winding order of triangles generated by marching cubes such that triangle normals will actually point outwards from the surface
 - Lib: Add benchmarks using Criterion
 - Lib: Add functions to approximate the vertex normals and vertex normal directions based on are-weighted averages of the adjacent triangles
 - Lib: Reduce severity of most log messages
 - Lib: Replace term `kernel_radius` with `compact_support_radius` to make it less ambiguous
 - Updated dependencies

### Breaking changes
 - CLI: The CLI now requires specifying subcommands. The reconstruction is now run using the `reconstruct` subcommand
 - CLI: The input file now has to be provided with the `-i` or `--input-file` flag and is not a positional argument anymore
 - CLI: Replace the `--kernel-radius` parameter with the more common term `--smoothing-length`, it holds that `compact_support_radius = 2.0 * smoothing_length`
 - CLI: Other small changes in parameter naming, please refer to the `--help` function
 - Lib: Several library API changes

### Regressions
 - "Splash detection" currently does nothing
 - The density map is currently not returned by the reconstruction

## Version 0.4.0

 - CLI: Add support for reading BGEO files
 - Lib: Allow passing in an existing `SurfaceReconstruction` to reuse allocated memory (currently only memory for the resulting surface mesh is reused) (https://github.com/w1th0utnam3/splashsurf/pull/7)
 - Lib: Add `Default` trait bound to `Index` and `Real` types
 - Updated dependencies

## Version 0.3.0

 - Lib: Move `vtkio` convenience trait implementations in `splashsurf_lib` behind a non-default feature flag (flag `vtk-extras`). Flag is enabled in the CLI crate.
 - Lib: Move `coarse_prof` usage in `splashsurf_lib` behind a non-default feature flag (flag `profiling`). Flag is enabled in the CLI crate.
 - Re-export `vtkio` from lib if feature is enabled and use this from CLI
 - Re-export `coarse_prof` from lib if feature is enabled and use this from CLI 
 - Lib: Instead of directly writing into a global density map (`dashmap`) in parallel, use thread local density maps (`HashMap`s) which are then merged in a second step (in parallel, into a `dashmap`). Improves performance of the test scene in the readme from 20 seconds to 12 seconds.
 - Lib: Fix bug which forced some parts in the library to divided work into 8 chunks, preventing use of more threads

## Version 0.2.0

 - CLI: Add support for reading PLY files (https://github.com/w1th0utnam3/splashsurf/pull/1)
 - Update dependencies
 - Re-export `nalgebra` from lib, use this from CLI

## Version 0.1.0

 - Initial release
