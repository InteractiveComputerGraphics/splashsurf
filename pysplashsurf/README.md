# Pysplashsurf

![splashsurf logo](https://raw.githubusercontent.com/InteractiveComputerGraphics/splashsurf/main/logos/logo_small.svg "splashsurf")

Pysplashsurf is a Python binding for splashsurf, an open source surface reconstruction library for particle data from SPH simulations. Detailed information on how the library works can be found on the [project website (splashsurf.physics-simulation.org)](https://splashsurf.physics-simulation.org/) or the [main repository](https://github.com/InteractiveComputerGraphics/splashsurf).

## Installation
```
pip install pysplashsurf
```
Requires Python version 3.7+

## Usage
Example to reconstruct the surface from an input file, apply some post processing methods and write the data back to a file:
```python
import meshio
import numpy as np
import pysplashsurf

mesh = meshio.read("input.vtk")
particles = np.array(mesh.points, dtype=np.float64)

mesh_with_data, reconstruction = pysplashsurf.reconstruction_pipeline(particles, particle_radius=0.025, rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, iso_surface_threshold=0.6, mesh_smoothing_weights=True, mesh_smoothing_weights_normalization=13.0, mesh_smoothing_iters=25, normals_smoothing_iters=10, mesh_cleanup=True, decimate_barnacles=True, compute_normals=True, output_raw_normals=True, subdomain_grid=True, subdomain_num_cubes_per_dim=64, output_mesh_smoothing_weights=True)
    
pysplashsurf.write_to_file(mesh_with_data, "output.vtk")
```
It may be necessary to specify the dtype of a function input (as done for `particles` in the example) so that Rust knows what data type to use internally. The extension supports single (`np.float32`) and double precision floats (`np.float64`).

## Build instructions
You can also manually build the package from the source code:
1. Clone the repository
2. cd to the pysplashsurf directory
3. Create an environment from python_environment.yaml and activate it
    - I recommend creating it in a subfolder, e.g.
    ```conda env create --prefix ./envs -f python_environment.yaml```
    - Then activate it using `conda activate ./envs`
4. Now, to build the project, use maturin: `maturin develop`
    - Maturin automatically installs the resulting binary in your python environment
    - Set the release flag `-r` or `--release` to build a more performant binary, however compilation time will be slightly longer

### Documentation Build
To generate the Sphinx documentation, make sure that the package is installed through e.g. maturin, and then run `make html` in the `pysplashsurf/pysplashsurf/docs` directory. The resulting html files will be in `pysplashsurf/pysplashsurf/docs/build/html`.

### Stub File Generation
To automatically generate a stub file for the package, run `cargo run --bin stub_gen` from the root project folder (from `pysplashsurf/`).