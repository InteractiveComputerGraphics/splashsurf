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

By default, a domain decomposition of the particle set is performed using octree-based subdivision.
The implementation first computes the density of each particle using the typical SPH approach with a cubic kernel. 
This density is then evaluated or mapped onto a sparse grid using spatial hashing in the support radius of each particle.
This implies that memory is only allocated in areas where the fluid density is non-zero. This is in contrast to a naive approach where the marching cubes background grid is allocated for the whole domain. 
The marching cubes reconstruction is performed only in the narrow band of grid cells where the density values cross the surface threshold. Cells completely in the interior of the fluid are skipped. For more details, please refer to the [readme of the library]((https://github.com/w1th0utnam3/splashsurf/blob/master/splashsurf_lib/README.md)).
Finally, all surface patches are stitched together by walking the octree back up, resulting in a closed surface.

## Notes

Due the use of hash maps and multi-threading (if enabled), the output of this implementation is not deterministic.
In the future, flags may be added to switch the internal data structures to use binary trees for debugging purposes.

Note that for small numbers of fluid particles (i.e. in the low thousands or less) the multi-threaded implementation may have worse performance due to the task based parallelism and the additional overhead of domain decomposition and stitching.
In this case, you can try to disable the domain decomposition. The reconstruction will then use a global approach that is parallelized using thread-local hashmaps.
For larger quantities of particles the decomposition approach will be faster however.

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
splashsurf-reconstruct 0.4.0
Reconstruct a surface from particle data

USAGE:
    splashsurf reconstruct [OPTIONS] --cube-size <cube-size> -i <input-file> --particle-radius <particle-radius> --smoothing-length <smoothing-length>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
        --cube-size <cube-size>
            The marching cubes grid size in multiplies of the particle radius

        --domain-max <domain-max> <domain-max> <domain-max>
            Upper corner of the domain where surface reconstruction should be performed, format:domain-
            max=x_max;y_max;z_max (requires domain-min to be specified)
        --domain-min <domain-min> <domain-min> <domain-min>
            Lower corner of the domain where surface reconstruction should be performed, format: domain-
            min=x_min;y_min;z_min (requires domain-max to be specified)
    -i <input-file>
            Path to the input file where the particle positions are stored (supported formats: VTK, binary f32 XYZ, PLY,
            BGEO)
    -n, --num-threads <num-threads>
            Set the number of threads for the worker thread pool

        --octree-ghost-margin-factor <octree-ghost-margin-factor>
            Safety factor applied to the kernel compact support radius when it's used as a margin to collect ghost
            particles in the leaf nodes
        --octree-max-particles <octree-max-particles>
            The maximum number of particles for leaf nodes of the octree, default is to compute it based on number of
            threads and particles
        --output-dir <output-dir>
            Optional base directory for all output files (default: current working directory)

        --output-dm-grid <output-dm-grid>
            Optional filename for writing the grid representation of the intermediate density map to disk

        --output-dm-points <output-dm-points>
            Optional filename for writing the point cloud representation of the intermediate density map to disk

    -o <output-file>
            Filename for writing the reconstructed surface to disk (default: "{original_filename}_surface.vtk")

        --output-octree <output-octree>
            Optional filename for writing the octree used to partition the particles to disk

        --mt-files <parallelize-over-files>
            Flag to enable multi-threading to process multiple input files in parallel [default: off]  [possible values:
            on, off]
        --mt-particles <parallelize-over-particles>
            Flag to enable multi-threading for a single input file by processing chunks of particles in parallel
            [default: on]  [possible values: on, off]
        --particle-radius <particle-radius>                          The particle radius of the input data
        --rest-density <rest-density>                                The rest density of the fluid [default: 1000.0]
        --smoothing-length <smoothing-length>
            The smoothing length radius used for the SPH kernel, the kernel compact support radius will be twice the
            smoothing length (in multiplies of the particle radius)
        --splash-detection-radius <splash-detection-radius>
            If a particle has no neighbors in this radius (in multiplies of the particle radius) it is considered as a
            free particle
        --surface-threshold <surface-threshold>
            The iso-surface threshold for the density, i.e. value of the reconstructed density that indicates the fluid
            surface (in multiplies of the rest density) [default: 0.6]
    -d, --use-double-precision <use-double-precision>
            Whether to enable the use of double precision for all computations [default: off]  [possible values: on,
            off]
        --use-octree <use-octree>
            Whether to enable spatial decomposition using an octree (faster) instead of a global approach [default: on]
            [possible values: on, off]
        --use-stitching <use-stitching>
            Whether to enable stitching of the disjoint subdomain meshes when spatial decomposition is enabled (slower
            but results in closed meshes) [default: on]  [possible values: on, off]
```
For example:
```
splashsurf "/home/user/canyon.xyz" --output-dir="/home/user/temp" --particle-radius=0.011 --kernel-radius=4.0 --cube-size=1.5 --surface-threshold=0.6 --mt-particles
splashsurf reconstruct -i data/canyon_13353401_particles.xyz --output-dir=out --particle-radius=0.011 --smoothing-length=2.0 --cube-size=1.5 --surface-threshold=0.6
```
With these parameters, a scene with 13353401 particles is reconstructed in less than 3.5 seconds on a Ryzen 9 5950X. The output is a mesh with 6022105 triangles.
```
[2021-01-29T15:14:18.738806+01:00][splashsurf][INFO] splashsurf v0.4.0 (splashsurf)
[2021-01-29T15:14:18.738828+01:00][splashsurf][INFO] Called with command line: target/release/splashsurf reconstruct -i data/canyon_13353401_particles.xyz --output-dir=out --particle-radius=0.011 --smoothing-length=2.0 --cube-size=1.5 --surface-threshold=0.6
[2021-01-29T15:14:18.738839+01:00][splashsurf::reconstruction][INFO] Using single precision (f32) for surface reconstruction.
[2021-01-29T15:14:18.738841+01:00][splashsurf::io][INFO] Reading particle dataset from "data/canyon_13353401_particles.xyz"...
[2021-01-29T15:14:18.921825+01:00][splashsurf::io][INFO] Successfully read dataset with 13353401 particle positions.
[2021-01-29T15:14:18.929109+01:00][splashsurf_lib][INFO] Minimal enclosing bounding box of particles was computed as: AxisAlignedBoundingBox { min: [-25.0060978, -5.0146289, -40.0634613], max: [24.4994926, 18.3062096, 39.7757950] }
[2021-01-29T15:14:18.946873+01:00][splashsurf_lib::utils][INFO] Splitting 13353401 particles into 257 chunks (with 52161 particles each) for octree generation
[2021-01-29T15:14:18.946879+01:00][splashsurf_lib::octree::split_criterion][INFO] Building octree with at most 52161 particles per leaf
[2021-01-29T15:14:19.165740+01:00][splashsurf_lib::reconstruction][INFO] Starting triangulation of surface patches.
[2021-01-29T15:14:21.591042+01:00][splashsurf_lib::reconstruction][INFO] Generation of surface patches is done.
[2021-01-29T15:14:21.591056+01:00][splashsurf_lib::reconstruction][INFO] Global mesh has 6022105 triangles and 3015090 vertices.
[2021-01-29T15:14:21.591062+01:00][splashsurf::reconstruction][INFO] Writing surface mesh to "out/canyon_13353401_particles_surface.vtk"...
[2021-01-29T15:14:21.997339+01:00][splashsurf::reconstruction][INFO] Done.
[2021-01-29T15:14:22.007195+01:00][splashsurf::reconstruction][INFO] Successfully finished processing all inputs.
[2021-01-29T15:14:22.007203+01:00][splashsurf][INFO] Timings:
[2021-01-29T15:14:22.007217+01:00][splashsurf][INFO] surface reconstruction cli: 100.00%, 3268.35ms/call @ 0.31Hz
[2021-01-29T15:14:22.007220+01:00][splashsurf][INFO]   loading particle positions: 5.60%, 182.98ms/call @ 0.31Hz
[2021-01-29T15:14:22.007222+01:00][splashsurf][INFO]   reconstruct_surface: 81.67%, 2669.22ms/call @ 0.31Hz
[2021-01-29T15:14:22.007224+01:00][splashsurf][INFO]     compute minimum enclosing aabb: 0.27%, 7.28ms/call @ 0.31Hz
[2021-01-29T15:14:22.007225+01:00][splashsurf][INFO]     octree subdivide_recursively_margin_par: 6.96%, 185.75ms/call @ 0.31Hz
[2021-01-29T15:14:22.007227+01:00][splashsurf][INFO]     parallel domain decomposed surface reconstruction with stitching: 90.86%, 2425.32ms/call @ 0.31Hz
[2021-01-29T15:14:22.007229+01:00][splashsurf][INFO]   write surface mesh to file: 12.43%, 406.29ms/call @ 0.31Hz
```

### Sequences of files

You can either process a single file or let the tool automatically process a sequence of files.
A sequence of files is indicated by specifying a filename with a `{}` placeholder pattern in the name.
The tool will then process files by replacing the placeholder with indices starting with `1` until a file with the given index does not exist anymore.
Note that the tool collects all existing filenames as soon as the command is invoked and does not update the list while running.

By specifying the flag `--mt-files=on`, several files can be processed in parallel.
Note that you should ideally also set `--mt-particles=off` as enabling both will probably degrade performance.

## Input file formats

### VTK

Files with the "`.vtk`" extension are loaded using [`vtkio`](https://crates.io/crates/vtkio). The VTK file is loaded as a big endian binary file and has to contain an "Unstructured Grid" with either `f32` or `f64` vertex coordinates. Any other data or attributes are ignored. Only the first "Unstructured Grid" is loaded, other entities are ignored.

### BGEO

Files with the "`.bgeo`" extension are loaded using a custom parser. Note, that only the "old" `BGEOV` format is supported (which is the format supported by "Partio"). Both uncompressed and (gzip) compressed files are supported. Only points and their implicit position vector attributes are loaded from the file. All other entities (e.g. vertices) and other attributes are ignored/discarded. Notably, the parser supports BGEO files written by [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH) ("Partio export"). 

### PLY

Files with the "`.ply`" extension are loaded using [`ply-rs`](https://crates.io/crates/ply-rs). The PLY file has to contain an element called "`vertex`" with the properties `x`, `y` and `z` of type `f32`/["`Property::Float`"](https://docs.rs/ply-rs/0.1.3/ply_rs/ply/enum.Property.html#variant.Float). Any other properties or elements are ignored.

### XYZ

Files with the "`.xyz`" extension are interpreted as raw bytes of `f32` values in native endianness of the system. Three consecutive `f32`s represent a (x,y,z) coordinate triplet of a fluid particle.

## Output file formats

Currently, only VTK files are supported for output.

# License

For license information of this project, see the LICENSE file.
The splashsurf logo is based on two graphics ([1](https://www.svgrepo.com/svg/295647/wave), [2](https://www.svgrepo.com/svg/295652/surfboard-surfboard)) published on SVG Repo under a CC0 ("No Rights Reserved") license.
