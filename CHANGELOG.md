## Master

 - CLI: Support for reading uncompressed BGEO files (previously only compressed BGEO files were supported)
 - Lib: Add functions to approximate the vertex normals and vertex normal directions based on are-weighted averages of the adjacent triangles 

## Version 0.4.0

 - CLI: Add support for reading BGEO files
 - Lib: Allow passing in an existing `SurfaceReconstruction` to reuse allocated memory (currently only memory for the resulting surface mesh is reused) (https://github.com/w1th0utnam3/splashsurf/pull/7)
 - Lib: Add `Default` trait bound to `Index` and `Real` types
 - Update dependencies

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
