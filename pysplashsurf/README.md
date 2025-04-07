# ![splashsurf logo](https://raw.githubusercontent.com/InteractiveComputerGraphics/splashsurf/main/logos/logo_small.svg "splashsurf")

# Pysplashsurf

Pysplashsurf is a Python binding to splashsurf, an open source surface reconstruction library for particle data from SPH simulations. Detailed information on how the library works can be found on the [project website (splashsurf.physics-simulation.org)](https://splashsurf.physics-simulation.org/) or the [main repository](https://github.com/InteractiveComputerGraphics/splashsurf).

## Installation
```
pip install pysplashsurf
```

## Usage
Example to reconstruct the surface from an input file, apply some post processing methods and write the data back to a file
```python
import meshio
import numpy as np
import pysplashsurf

mesh = meshio.read("input.vtk")
particles = np.array(mesh.points, dtype=np.float64)

mesh_with_data, reconstruction = pysplashsurf.reconstruction_pipeline(particles, particle_radius=0.025, rest_density=1000.0, smoothing_length=2.0, cube_size=0.5, iso_surface_threshold=0.6, mesh_smoothing_weights=True, mesh_smoothing_weights_normalization=13.0, mesh_smoothing_iters=25, normals_smoothing_iters=10, mesh_cleanup=True, decimate_barnacles=True, compute_normals=True, output_raw_normals=True, subdomain_grid=True, subdomain_num_cubes_per_dim=64, output_mesh_smoothing_weights=True)
    
mesh = mesh_with_data.take_mesh()

point_data = {}
for key in mesh_with_data.get_point_attribute_keys():
    point_data[key] = mesh_with_data.get_point_attribute(key)

cell_data = {}
for key in mesh_with_data.get_cell_attribute_keys():
    cell_data[key] = mesh_with_data.get_cell_attribute(key)

verts, tris = mesh.take_vertices_and_triangles()
meshio.write_points_cells("output.vtk", verts, [("triangle", tris)], point_data=point_data, cell_data=cell_data)
```

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