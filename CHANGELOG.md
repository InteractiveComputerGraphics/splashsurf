## Master

## Version 0.3.0

 - Move `vtkio` convenience trait implementations in `splashsurf_lib` behind a non-default feature flag (flag `vtk-extras`). Flag is enabled in the CLI crate.
 - Re-export `vtkio` from lib if feature is enabled and use this from CLI 
 - Move `coarse_prof` usage in `splashsurf_lib` behind a non-default feature flag (flag `profiling`). Flag is enabled in the CLI crate.
 - Re-export `coarse_prof` from lib if feature is enabled and use this from CLI 
 - Instead of directly writing into a global density map (`dashmap`) in parallel, use thread local density maps (`HashMap`s) which are then merged in a second step (in parallel, into a `dashmap`). Improves performance of the test scene in the readme from 20 seconds to 12 seconds.
 - Fix bug which forced some parts in the library to divided work into 8 chunks, preventing use of more threads

## Version 0.2.0

 - Add support for reading PLY files (https://github.com/w1th0utnam3/splashsurf/pull/1)
 - Update dependencies
 - Re-export `nalgebra` from lib, use this from CLI

## Version 0.1.0

 - Initial release
