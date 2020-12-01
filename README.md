# ![splashsurf logo](logos/logo_small.svg "splashsurf")
[![On crates.io](https://img.shields.io/crates/v/splashsurf)](https://crates.io/crates/splashsurf)

Surface reconstruction for particle data from SPH simulations, written in Rust.

## Introduction

This is a basic but high-performance implementation of a marching cubes based surface reconstruction for SPH fluid simulations (e.g performed with [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)).
The output of this tool is the reconstructed triangle surface mesh of the fluid.
At the moment it does not compute normals or other additional data.
As input, it supports particle positions from .vtk files and binary .xyz files (i.e. files containing a binary dump of a particle position array). In addition, required parameters are the kernel radius and particle radius (to compute the volume of particles) used for the original SPH simulation as well as the surface threshold.

The implementation first computes the density of each particle using the typical SPH approach with a cubic kernel. 
This density is then evaluated or mapped onto a sparse grid using spatial hashing in the support radius of each particle.
This implies that memory is only allocated in areas where the fluid density is non-zero. This is in contrast to a naive approach where the marching cubes background grid is allocated for the whole domain. 
Finally, the marching cubes reconstruction is performed only in those grid cells where the density values cross the surface threshold. Cells completely in the interior of the fluid are skipped.

Due the use of hash maps and multi-threading (if enabled), the output of this implementation is not deterministic.
In the future, flags may be added to switch the internally data structures to use binary trees for debugging.

As shown below, the tool can handle the output of large simulations.
However, it was not tested with a wide range of parameters and may not be totally robust against corner-cases or extreme parameters.
If you experience problems, please report them together with your input data.

## Installation

The command-line tool can be built from this repository but is also available on crates.io.
If you have a [Rust toolchain installed](https://www.rust-lang.org/tools/install) you can install `splashsurf` with the command
```
cargo install splashsurf
```

## Usage

```
USAGE:
    splashsurf.exe [FLAGS] [OPTIONS] <input-file> --cube-size <cube-size> --kernel-radius <kernel-radius> --particle-radius <particle-radius> --surface-threshold <surface-threshold>

FLAGS:
    -h, --help            Prints help information
        --mt-files        Whether to enable multi-threading to process multiple input files in parallel, conflicts with
                          --mt-particles
        --mt-particles    Whether to enable multi-threading for a single input file by processing chunks of particles in
                          parallel, conflicts with --mt-files
    -d                    Whether to enable the use of double precision for all computations (disabled by default)
    -V, --version         Prints version information

OPTIONS:
        --cube-size <cube-size>
            The marching cubes grid size in multiplies of the particle radius

        --domain-max <domain-max> <domain-max> <domain-max>
            Upper corner of the domain where surface reconstruction should be performed, format:domain-
            max=x_max;y_max;z_max (requires domain-min to be specified)
        --domain-min <domain-min> <domain-min> <domain-min>
            Lower corner of the domain where surface reconstruction should be performed, format: domain-
            min=x_min;y_min;z_min (requires domain-max to be specified)
        --kernel-radius <kernel-radius>
            The kernel radius used for building the density map in multiplies of the particle radius

        --output-dir <output-dir>                              Optional base directory for all output files
        --output-dm-grid <output-dm-grid>
            Optional filename for writing the grid representation of the intermediate density map to disk

        --output-dm-points <output-dm-points>
            Optional filename for writing the point cloud representation of the intermediate density map to disk

    -o <output-file>                                           Filename for writing the reconstructed surface to disk
        --particle-radius <particle-radius>                    The particle radius of the input data
        --rest-density <rest-density>                          The rest density of the fluid [default: 1000.0]
        --splash-detection-radius <splash-detection-radius>
            If a particle has no neighbors in this radius (in multiplies of the particle radius) it is considered as a
            free particles
        --surface-threshold <surface-threshold>
            The iso-surface threshold for the density, i.e. value of the reconstructed density that indicates the fluid
            surface

ARGS:
    <input-file>    Path to the input file where the particle positions are stored (supported formats: VTK, binary
                    XYZ, ply)
```
For example:
```
splashsurf "/home/user/canyon.xyz" --output-dir="/home/user/temp" --particle-radius=0.011 --kernel-radius=4.0 --cube-size=1.5 --surface-threshold=0.6 --mt-particles
```
With these parameters, a scene with 13353401 particles is reconstructed in nearly than 25s on a i9 9900K. The output is a mesh with 6016212 triangles.
```
[2020-08-25T15:52:34.515885+02:00][splashsurf::reconstruction][INFO] Loading dataset from "/local/data/temp/canyon.xyz"...
[2020-08-25T15:52:34.655354+02:00][splashsurf::reconstruction][INFO] Loaded dataset with 13353401 particle positions.
[2020-08-25T15:52:34.684734+02:00][splashsurf_lib][INFO] Minimal enclosing bounding box of particles was computed as: AxisAlignedBoundingBox { min: [-25.0060978, -5.0146289, -40.0634613], max: [24.4994926, 18.3062096, 39.7757950] }
[2020-08-25T15:52:34.684748+02:00][splashsurf_lib][INFO] Using a grid with 6002x2828x9679 points and 6001x2827x9678 cells of edge length 0.0165.
[2020-08-25T15:52:34.684753+02:00][splashsurf_lib][INFO] The resulting domain size is: AxisAlignedBoundingBox { min: [-49.7588959, -16.6750488, -79.9830933], max: [49.2576065, 29.9704514, 79.7039032] }
[2020-08-25T15:52:34.684756+02:00][splashsurf_lib][INFO] Starting neighborhood search...
[2020-08-25T15:52:36.570860+02:00][splashsurf_lib][INFO] Computing particle densities...
[2020-08-25T15:52:37.645919+02:00][splashsurf_lib::density_map][INFO] Starting construction of sparse density map for 13353401 particles...
[2020-08-25T15:52:37.653068+02:00][splashsurf_lib::density_map][INFO] To take into account the kernel evaluation radius, the allowed domain of particles was restricted to: AxisAlignedBoundingBox { min: [-49.7093468, -16.6254997, -79.9335403], max: [49.2080574, 29.9209023, 79.6543503] }
[2020-08-25T15:52:55.559939+02:00][splashsurf_lib::density_map][INFO] Sparse density map with 31519986 point data values was constructed.
[2020-08-25T15:52:55.559999+02:00][splashsurf_lib::density_map][INFO] Construction of sparse density map done.
[2020-08-25T15:52:55.560005+02:00][splashsurf_lib::marching_cubes][INFO] Starting interpolation of cell data for marching cubes...
[2020-08-25T15:52:59.118442+02:00][splashsurf_lib::marching_cubes][INFO] Generated cell data for marching cubes with 3009863 cells and 3011516 vertices.
[2020-08-25T15:52:59.118470+02:00][splashsurf_lib::marching_cubes][INFO] Interpolation done.
[2020-08-25T15:52:59.118474+02:00][splashsurf_lib::marching_cubes][INFO] Starting marching cubes triangulation of 3009863 cells...
[2020-08-25T15:52:59.279570+02:00][splashsurf_lib::marching_cubes][INFO] Generated surface mesh with 6016212 triangles and 3011516 vertices.
[2020-08-25T15:52:59.279597+02:00][splashsurf_lib::marching_cubes][INFO] Triangulation done.
[2020-08-25T15:52:59.296979+02:00][splashsurf::reconstruction][INFO] Writing surface mesh to "/home/floeschner/programming/temp/canyon_surface.xyz"...
[2020-08-25T15:52:59.808101+02:00][splashsurf::reconstruction][INFO] Done.
[2020-08-25T15:52:59.879069+02:00][splashsurf][INFO] Finished processing all inputs.
[2020-08-25T15:52:59.879103+02:00][splashsurf][INFO] surface reconstruction cli: 100.00%, 25363.19ms/call @ 0.04Hz
[2020-08-25T15:52:59.879107+02:00][splashsurf][INFO]   loading particle positions: 0.55%, 139.30ms/call @ 0.04Hz
[2020-08-25T15:52:59.879109+02:00][splashsurf][INFO]   reconstruct_surface: 97.15%, 24641.60ms/call @ 0.04Hz
[2020-08-25T15:52:59.879111+02:00][splashsurf][INFO]     compute minimum enclosing aabb: 0.12%, 29.37ms/call @ 0.04Hz
[2020-08-25T15:52:59.879113+02:00][splashsurf][INFO]     neighborhood_search: 7.65%, 1886.10ms/call @ 0.04Hz
[2020-08-25T15:52:59.879115+02:00][splashsurf][INFO]       parallel_generate_cell_to_particle_map: 11.52%, 217.26ms/call @ 0.04Hz
[2020-08-25T15:52:59.879117+02:00][splashsurf][INFO]       get_cell_neighborhoods_par: 2.70%, 50.97ms/call @ 0.04Hz
[2020-08-25T15:52:59.879119+02:00][splashsurf][INFO]       calculate_particle_neighbors_par: 52.80%, 995.77ms/call @ 0.04Hz
[2020-08-25T15:52:59.879121+02:00][splashsurf][INFO]     parallel_compute_particle_densities: 1.12%, 275.39ms/call @ 0.04Hz
[2020-08-25T15:52:59.879144+02:00][splashsurf][INFO]     parallel_generate_sparse_density_map: 72.70%, 17914.09ms/call @ 0.04Hz
[2020-08-25T15:52:59.879146+02:00][splashsurf][INFO]     triangulate_density_map: 15.17%, 3736.97ms/call @ 0.04Hz
[2020-08-25T15:52:59.879148+02:00][splashsurf][INFO]       interpolate_points_to_cell_data: 95.22%, 3558.47ms/call @ 0.04Hz
[2020-08-25T15:52:59.879161+02:00][splashsurf][INFO]         generate_iso_surface_vertices: 69.82%, 2484.44ms/call @ 0.04Hz
[2020-08-25T15:52:59.879164+02:00][splashsurf][INFO]         relative_to_threshold_postprocessing: 29.70%, 1056.71ms/call @ 0.04Hz
[2020-08-25T15:52:59.879167+02:00][splashsurf][INFO]       triangulate: 4.78%, 178.50ms/call @ 0.04Hz
[2020-08-25T15:52:59.879171+02:00][splashsurf][INFO]   write surface mesh to file: 2.02%, 511.21ms/call @ 0.04Hz
```

## Input file formats

### Ply

The ply format expects an element definition 'vertex', with element properties of 'x,y,z'

## License

For license information of this project, see the LICENSE file.
The splashsurf logo is based on two graphics ([1](https://www.svgrepo.com/svg/295647/wave), [2](https://www.svgrepo.com/svg/295652/surfboard-surfboard)) published on SVG Repo under a CC0 ("No Rights Reserved") license.
