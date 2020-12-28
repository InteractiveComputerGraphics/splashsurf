# ![splashsurf logo](https://raw.githubusercontent.com/w1th0utnam3/splashsurf/master/logos/logo_small.svg "splashsurf")
[![On crates.io](https://img.shields.io/crates/v/splashsurf)](https://crates.io/crates/splashsurf)
[![On docs.rs](https://docs.rs/splashsurf_lib/badge.svg)](https://docs.rs/splashsurf_lib)
[![License: MIT](https://img.shields.io/crates/l/splashsurf)](https://github.com/w1th0utnam3/splashsurf/blob/master/LICENSE)
![Build and test GitHub Actions workflow](https://github.com/w1th0utnam3/splashsurf/workflows/Build%20and%20test/badge.svg)

Surface reconstruction library and CLI for particle data from SPH simulations, written in Rust.

**Contents**
- [The `splashsurf` CLI](#the-splashsurf-cli)
  - [Introduction](#introduction)
  - [Notes](#notes)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Basic usage](#basic-usage)
    - [Sequences of files](#sequences-of-files)
  - [Input file formats](#input-file-formats)
    - [VTK](#vtk)
    - [BGEO](#bgeo)
    - [PLY](#ply)
    - [XYZ](#xyz)
  - [Output file formats](#output-file-formats)
- [License](#license)

# The `splashsurf` CLI

The following sections mainly focus on the CLI of `splashsurf`. For more information on the library, see the [corresponding readme](https://github.com/w1th0utnam3/splashsurf/blob/master/splashsurf_lib/README.md) in the `splashsurf_lib` subfolder or the [`splashsurf_lib` crate](https://crates.io/crates/splashsurf_lib) on crates.io.

## Introduction

This is a basic but high-performance implementation of a marching cubes based surface reconstruction for SPH fluid simulations (e.g performed with [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)).
The output of this tool is the reconstructed triangle surface mesh of the fluid.
At the moment it does not compute normals or other additional data.
As input, it supports reading particle positions from `.vtk`, `.bgeo`, `.ply` and binary `.xyz` files (i.e. files containing a binary dump of a particle position array). In addition, required parameters are the kernel radius and particle radius (to compute the volume of particles) used for the original SPH simulation as well as the surface threshold.

The implementation first computes the density of each particle using the typical SPH approach with a cubic kernel. 
This density is then evaluated or mapped onto a sparse grid using spatial hashing in the support radius of each particle.
This implies that memory is only allocated in areas where the fluid density is non-zero. This is in contrast to a naive approach where the marching cubes background grid is allocated for the whole domain. 
Finally, the marching cubes reconstruction is performed only in those grid cells where the density values cross the surface threshold. Cells completely in the interior of the fluid are skipped. For more details, please refer to the [readme of the library]((https://github.com/w1th0utnam3/splashsurf/blob/master/splashsurf_lib/README.md)).

## Notes

Due the use of hash maps and multi-threading (if enabled), the output of this implementation is not deterministic.
In the future, flags may be added to switch the internal data structures to use binary trees for debugging purposes.

Note that for small numbers of fluid particles (i.e. in the low thousands or less) the multi-threaded implementation may have worse performance due to the worker pool overhead and looks on the map data structures (even though the library uses [`dashmap`](https://crates.io/crates/dashmap) which is more optimized for multi-threaded usage).

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

### Basic usage

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
                    XYZ, PLY, BGEO)
```
For example:
```
splashsurf "/home/user/canyon.xyz" --output-dir="/home/user/temp" --particle-radius=0.011 --kernel-radius=4.0 --cube-size=1.5 --surface-threshold=0.6 --mt-particles
```
With these parameters, a scene with 13353401 particles is reconstructed in less than 12 seconds on a Ryzen 9 5950X. The output is a mesh with 6016212 triangles.
```
[2020-12-16T15:38:43.729112+01:00][splashsurf][INFO] splashsurf v0.3.0 (splashsurf)
[2020-12-16T15:38:43.729574+01:00][splashsurf][INFO] Called with command line: target\release\splashsurf.exe U:\Documents\Sciebo\Files\canyon.xyz --output-dir=U:\ --particle-radius=0.011 --kernel-radius=4.0 --cube-size=1.5 --surface-threshold=0.6 --mt-particles
[2020-12-16T15:38:43.729982+01:00][splashsurf::io][INFO] Loading dataset from "U:\Documents\Sciebo\Files\canyon.xyz"...
[2020-12-16T15:38:43.918459+01:00][splashsurf::io][INFO] Loaded dataset with 13353401 particle positions.
[2020-12-16T15:38:43.958381+01:00][splashsurf_lib][INFO] Minimal enclosing bounding box of particles was computed as: AxisAlignedBoundingBox { min: [-25.0060978, -5.0146289, -40.0634613], max: [24.4994926, 18.3062096, 39.7757950] }
[2020-12-16T15:38:43.958419+01:00][splashsurf_lib][INFO] Using a grid with 6002x2828x9679 points and 6001x2827x9678 cells of edge length 0.0165.
[2020-12-16T15:38:43.958428+01:00][splashsurf_lib][INFO] The resulting domain size is: AxisAlignedBoundingBox { min: [-49.7588959, -16.6750488, -79.9830933], max: [49.2576065, 29.9704514, 79.7039032] }
[2020-12-16T15:38:43.958435+01:00][splashsurf_lib][INFO] Starting neighborhood search...
[2020-12-16T15:38:45.670110+01:00][splashsurf_lib][INFO] Computing particle densities...
[2020-12-16T15:38:47.017055+01:00][splashsurf_lib::density_map][INFO] Starting construction of sparse density map for 13353401 particles...
[2020-12-16T15:38:47.017132+01:00][splashsurf_lib::density_map][INFO] To take into account the kernel evaluation radius, the allowed domain of particles was restricted to: AxisAlignedBoundingBox { min: [-49.7093468, -16.6254997, -79.9335403], max: [49.2080574, 29.9209023, 79.6543503] }
[2020-12-16T15:38:47.017146+01:00][splashsurf_lib::density_map][INFO] Splitting particles into 320 chunks (with 41729 particles each) for density map generation
[2020-12-16T15:38:49.576935+01:00][splashsurf_lib::density_map][INFO] Merging 32 thread local density maps to a single global map...
[2020-12-16T15:38:50.653927+01:00][splashsurf_lib::density_map][INFO] Global sparse density map with 31519986 grid point data values was constructed.
[2020-12-16T15:38:50.653958+01:00][splashsurf_lib::density_map][INFO] Construction of sparse density map done.
[2020-12-16T15:38:50.731081+01:00][splashsurf_lib::marching_cubes][INFO] Starting interpolation of cell data for marching cubes...
[2020-12-16T15:38:54.771895+01:00][splashsurf_lib::marching_cubes][INFO] Generated cell data for marching cubes with 3009863 cells and 3011516 vertices.
[2020-12-16T15:38:54.771926+01:00][splashsurf_lib::marching_cubes][INFO] Interpolation done.
[2020-12-16T15:38:54.771941+01:00][splashsurf_lib::marching_cubes][INFO] Starting marching cubes triangulation of 3009863 cells...
[2020-12-16T15:38:54.930684+01:00][splashsurf_lib::marching_cubes][INFO] Generated surface mesh with 6016212 triangles and 3011516 vertices.
[2020-12-16T15:38:54.930724+01:00][splashsurf_lib::marching_cubes][INFO] Triangulation done.
[2020-12-16T15:38:54.982827+01:00][splashsurf::reconstruction][INFO] Writing surface mesh to "U:\canyon_surface.vtk"...
[2020-12-16T15:38:55.362546+01:00][splashsurf::reconstruction][INFO] Done.
[2020-12-16T15:38:55.451486+01:00][splashsurf][INFO] Successfully finished processing all inputs.
[2020-12-16T15:38:55.451544+01:00][splashsurf][INFO] surface reconstruction cli: 100.00%, 11721.50ms/call @ 0.09Hz
[2020-12-16T15:38:55.451551+01:00][splashsurf][INFO]   loading particle positions: 1.61%, 188.46ms/call @ 0.09Hz
[2020-12-16T15:38:55.451557+01:00][splashsurf][INFO]   reconstruct_surface: 94.39%, 11064.34ms/call @ 0.09Hz
[2020-12-16T15:38:55.451562+01:00][splashsurf][INFO]     compute minimum enclosing aabb: 0.36%, 39.93ms/call @ 0.09Hz
[2020-12-16T15:38:55.451567+01:00][splashsurf][INFO]     neighborhood_search: 15.47%, 1711.67ms/call @ 0.09Hz
[2020-12-16T15:38:55.451572+01:00][splashsurf][INFO]       parallel_generate_cell_to_particle_map: 12.23%, 209.30ms/call @ 0.09Hz
[2020-12-16T15:38:55.451577+01:00][splashsurf][INFO]       get_cell_neighborhoods_par: 3.36%, 57.47ms/call @ 0.09Hz
[2020-12-16T15:38:55.451581+01:00][splashsurf][INFO]       calculate_particle_neighbors_par: 49.84%, 853.06ms/call @ 0.09Hz
[2020-12-16T15:38:55.451586+01:00][splashsurf][INFO]     parallel_compute_particle_densities: 1.88%, 208.54ms/call @ 0.09Hz
[2020-12-16T15:38:55.451590+01:00][splashsurf][INFO]     parallel_generate_sparse_density_map: 33.57%, 3713.97ms/call @ 0.09Hz
[2020-12-16T15:38:55.451595+01:00][splashsurf][INFO]       generate thread local maps: 68.92%, 2559.78ms/call @ 0.09Hz
[2020-12-16T15:38:55.451600+01:00][splashsurf][INFO]       merge thread local maps to global map: 31.07%, 1154.11ms/call @ 0.09Hz
[2020-12-16T15:38:55.451604+01:00][splashsurf][INFO]     triangulate_density_map: 38.41%, 4249.48ms/call @ 0.09Hz
[2020-12-16T15:38:55.451608+01:00][splashsurf][INFO]       interpolate_points_to_cell_data: 95.09%, 4040.85ms/call @ 0.09Hz
[2020-12-16T15:38:55.451613+01:00][splashsurf][INFO]         generate_iso_surface_vertices: 75.45%, 3048.72ms/call @ 0.09Hz
[2020-12-16T15:38:55.451618+01:00][splashsurf][INFO]         relative_to_threshold_postprocessing: 24.55%, 992.06ms/call @ 0.09Hz
[2020-12-16T15:38:55.451622+01:00][splashsurf][INFO]       triangulate: 4.91%, 208.61ms/call @ 0.09Hz
[2020-12-16T15:38:55.451627+01:00][splashsurf][INFO]   write surface mesh to file: 3.24%, 379.74ms/call @ 0.09Hz
```

### Sequences of files

*TODO: Describe syntax to reconstruct a sequence of files*

## Input file formats

### VTK

Files with the "`.vtk`" extension are loaded using [`vtkio`](https://crates.io/crates/vtkio). The VTK file is loaded as a big endian binary file and has to contain an "Unstructured Grid" with either `f32` or `f64` vertex coordinates. Any other data or attributes are ignored. Only the first "Unstructured Grid" is loaded, other entities are ignored.

### BGEO

Files with the "`.bgeo`" extension are loaded using a custom parser. Note, that only the "old" `BGEOV` format is supported (which is the format supported by "Partio"). Only points and their implicit position vector attributes are loaded from the file. All other entities (e.g. vertices) and other attributes are ignored/discarded. Notably, the parser supports BGEO files written by [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH) ("Partio export"). 

### PLY

Files with the "`.ply`" extension are loaded using [`ply-rs`](https://crates.io/crates/ply-rs). The PLY file has to contain an element called "`vertex`" with the properties `x`, `y` and `z` of type `f32`/["`Property::Float`"](https://docs.rs/ply-rs/0.1.3/ply_rs/ply/enum.Property.html#variant.Float). Any other properties or elements are ignored.

### XYZ

Files with the "`.xyz`" extension are interpreted as raw bytes of `f32` values in native endianness of the system. Three consecutive `f32`s represent a (x,y,z) coordinate triplet of a fluid particle.

## Output file formats

Currently, only VTK files are supported for output.

# License

For license information of this project, see the LICENSE file.
The splashsurf logo is based on two graphics ([1](https://www.svgrepo.com/svg/295647/wave), [2](https://www.svgrepo.com/svg/295652/surfboard-surfboard)) published on SVG Repo under a CC0 ("No Rights Reserved") license.
