# ![splashsurf logo](https://raw.githubusercontent.com/w1th0utnam3/splashsurf/master/logos/logo_small.svg "splashsurf")
[![On crates.io](https://img.shields.io/crates/v/splashsurf)](https://crates.io/crates/splashsurf)
[![On docs.rs](https://docs.rs/splashsurf_lib/badge.svg)](https://docs.rs/splashsurf_lib)
[![License: MIT](https://img.shields.io/crates/l/splashsurf)](https://github.com/w1th0utnam3/splashsurf/blob/master/LICENSE)
[![Dependency status](https://deps.rs/repo/github/w1th0utnam3/splashsurf/status.svg)](https://deps.rs/repo/github/w1th0utnam3/splashsurf)
![Build and test GitHub Actions workflow](https://github.com/w1th0utnam3/splashsurf/workflows/Build%20and%20test/badge.svg)

Surface reconstruction library and CLI for particle data from SPH simulations, written in Rust.

<p align="center">
<img src="example_particles.png" alt="Image of the original particle data" width="32%"> <img src="example_coarse.png" alt="Image of a coarse reconstructed surface mesh" width="32%"> <img src="example_fine.png" alt="Image of a fine reconstructed surface mesh" width="32%">
</p>

`splashsurf` is a tool to reconstruct surfaces meshes from SPH particle data.
The first image shows the visualization of a set of particles from an SPH fluid simulation from [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH).
The particle radius is `0.025`. As the rendering of a fluid should not look like a ball pit, a surface mesh has to be
reconstructed from this particle data. The next image shows a reconstructed surface mesh of the fluid produced by `splashsurf`
with a "smoothing length" of `2.2` times the particles radius and a cell size of `1.1` times the particle radius. The
third image shows a finer reconstruction with a cell size of `0.45` times the particle radius. These surface meshes can
then be fed into 3D rendering software such as [Blender](https://www.blender.org/) to generate beautiful water animations.
The result might look something like this (please excuse the lack of 3D rendering skills):

<p align="center">
<img src="https://raw.githubusercontent.com/w1th0utnam3/w1th0utnam3.github.io/master/splashsurf.gif" alt="Rendered water animation" width="96%">
</p>

**Contents**
- [The `splashsurf` CLI](#the-splashsurf-cli)
  - [Introduction](#introduction)
  - [Notes](#notes)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Recommended settings](#recommended-settings)
    - [Benchmark example](#benchmark-example)
    - [Sequences of files](#sequences-of-files)
  - [Input file formats](#input-file-formats)
    - [VTK](#vtk)
    - [BGEO](#bgeo)
    - [PLY](#ply)
    - [XYZ](#xyz)
    - [JSON](#json)
  - [Output file formats](#output-file-formats)
  - [All command line options](#all-command-line-options)
    - [The `reconstruct` command](#the-reconstruct-command)
    - [The `convert` subcommand](#the-convert-subcommand)
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

### Recommended settings
Good settings for the surface reconstruction depend on the original simulation and can be influenced by different conventions of different simulators. The following settings appear to work well with [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH):
 - `particle-radius`: should be a bit larger than the particle radius used for the actual simulation. A radius around 1.4 to 1.6 times larger than the original SPH particle radius seems to be appropriate.
 - `smoothing-length`: should be set around `1.2`. Larger values smooth out the iso-surface more but also artificially increase the fluid volume.
 - `surface-threshold`: a good value depends on the selected `particle-radius` and `smoothing-length` and can be used to counteract a fluid volume increase e.g. due to a larger particle radius. In combination with the other recommended values a threshold of `0.6` seemed to work well.
 - `cube-size` i.e. marching cubes resolution of less than `1.0`, e.g. start with `0.5` and increase/decrease it if the result is not smooth enough or the reconstruction takes too long.

### Benchmark example
For example:
```
splashsurf reconstruct -i data/canyon_13353401_particles.xyz --output-dir=out --particle-radius=0.011 --smoothing-length=2.0 --cube-size=1.5 --surface-threshold=0.6
```
With these parameters, a scene with 13353401 particles is reconstructed in less than 3 seconds on a Ryzen 9 5950X. The output is a mesh with 6023244 triangles.
```
[2021-02-09T00:16:11.677571+01:00][splashsurf][INFO] splashsurf v0.6.0 (splashsurf)
[2021-02-09T00:16:11.677595+01:00][splashsurf][INFO] Called with command line: target/release/splashsurf reconstruct -i /home/fabian/programming/canyon_13353401_particles.xyz --output-dir=out --particle-radius=0.011 --smoothing-length=2.0 --cube-size=1.5 --surface-threshold=0.6
[2021-02-09T00:16:11.677609+01:00][splashsurf::reconstruction][INFO] Using single precision (f32) for surface reconstruction.
[2021-02-09T00:16:11.677618+01:00][splashsurf::io][INFO] Reading particle dataset from "/home/fabian/programming/canyon_13353401_particles.xyz"...
[2021-02-09T00:16:11.796790+01:00][splashsurf::io][INFO] Successfully read dataset with 13353401 particle positions.
[2021-02-09T00:16:11.804396+01:00][splashsurf_lib][INFO] Minimal enclosing bounding box of particles was computed as: AxisAlignedBoundingBox { min: [-25.0060978, -5.0146289, -40.0634613], max: [24.4994926, 18.3062096, 39.7757950] }
[2021-02-09T00:16:11.826742+01:00][splashsurf_lib::utils][INFO] Splitting 13353401 particles into 257 chunks (with 52161 particles each) for octree generation
[2021-02-09T00:16:11.826749+01:00][splashsurf_lib::octree::split_criterion][INFO] Building octree with at most 52161 particles per leaf
[2021-02-09T00:16:12.011061+01:00][splashsurf_lib::reconstruction][INFO] Starting computation of particle densities.
[2021-02-09T00:16:12.891286+01:00][splashsurf_lib::reconstruction][INFO] Starting triangulation of surface patches.
[2021-02-09T00:16:14.337897+01:00][splashsurf_lib::reconstruction][INFO] Generation of surface patches is done.
[2021-02-09T00:16:14.337922+01:00][splashsurf_lib::reconstruction][INFO] Global mesh has 6023244 triangles and 3015096 vertices.
[2021-02-09T00:16:14.337929+01:00][splashsurf::reconstruction][INFO] Writing surface mesh to "out/canyon_13353401_particles_surface.vtk"...
[2021-02-09T00:16:14.604527+01:00][splashsurf::reconstruction][INFO] Done.
[2021-02-09T00:16:14.613392+01:00][splashsurf::reconstruction][INFO] Successfully finished processing all inputs.
[2021-02-09T00:16:14.613399+01:00][splashsurf][INFO] Timings:
[2021-02-09T00:16:14.613479+01:00][splashsurf][INFO] surface reconstruction cli: 100.00%, 2935.77ms avg @ 0.34Hz (1 call)
[2021-02-09T00:16:14.613481+01:00][splashsurf][INFO]   loading particle positions: 4.06%, 119.17ms avg @ 0.34Hz (1 call)
[2021-02-09T00:16:14.613483+01:00][splashsurf][INFO]   compute minimum enclosing aabb: 0.26%, 7.61ms avg @ 0.34Hz (1 call)
[2021-02-09T00:16:14.613484+01:00][splashsurf][INFO]   reconstruct_surface_domain_decomposition: 86.30%, 2533.51ms avg @ 0.34Hz (1 call)
[2021-02-09T00:16:14.613485+01:00][splashsurf][INFO]     octree subdivide_recursively_margin_par: 7.28%, 184.32ms avg @ 0.39Hz (1 call)
[2021-02-09T00:16:14.613486+01:00][splashsurf][INFO]     parallel subdomain particle density computation: 33.43%, 847.00ms avg @ 0.39Hz (1 call)
[2021-02-09T00:16:14.613487+01:00][splashsurf][INFO]       visit octree node for density computation: 3116.00%, 23.88ms avg @ 1304.61Hz (1105 calls)
[2021-02-09T00:16:14.613488+01:00][splashsurf][INFO]         compute_particle_densities_and_neighbors: 96.55%, 26.35ms avg @ 36.64Hz (967 calls)
[2021-02-09T00:16:14.613489+01:00][splashsurf][INFO]           neighborhood_search: 88.00%, 23.19ms avg @ 37.95Hz (967 calls)
[2021-02-09T00:16:14.613490+01:00][splashsurf][INFO]             sequential_generate_cell_to_particle_map: 5.92%, 1.37ms avg @ 43.12Hz (967 calls)
[2021-02-09T00:16:14.613491+01:00][splashsurf][INFO]             calculate_particle_neighbors_seq: 78.63%, 18.23ms avg @ 43.12Hz (967 calls)
[2021-02-09T00:16:14.613492+01:00][splashsurf][INFO]           sequential_compute_particle_densities: 11.99%, 3.16ms avg @ 37.95Hz (967 calls)
[2021-02-09T00:16:14.613493+01:00][splashsurf][INFO]         update global density values: 1.38%, 0.38ms avg @ 36.64Hz (967 calls)
[2021-02-09T00:16:14.613494+01:00][splashsurf][INFO]     parallel domain decomposed surf. rec. with stitching: 57.10%, 1446.64ms avg @ 0.39Hz (1 call)
[2021-02-09T00:16:14.613495+01:00][splashsurf][INFO]       visit octree node (reconstruct or stitch): 2812.08%, 36.81ms avg @ 763.84Hz (1105 calls)
[2021-02-09T00:16:14.613496+01:00][splashsurf][INFO]         reconstruct_surface_patch: 96.80%, 50.36ms avg @ 19.22Hz (782 calls)
[2021-02-09T00:16:14.613497+01:00][splashsurf][INFO]           sequential_generate_sparse_density_map_subdomain: 89.21%, 44.92ms avg @ 19.86Hz (782 calls)
[2021-02-09T00:16:14.613498+01:00][splashsurf][INFO]           triangulate_density_map_append: 10.78%, 5.43ms avg @ 19.86Hz (782 calls)
[2021-02-09T00:16:14.613499+01:00][splashsurf][INFO]             interpolate_points_to_cell_data_skip_boundary: 83.44%, 4.53ms avg @ 184.17Hz (782 calls)
[2021-02-09T00:16:14.613500+01:00][splashsurf][INFO]               generate_iso_surface_vertices: 99.97%, 4.53ms avg @ 220.72Hz (782 calls)
[2021-02-09T00:16:14.613501+01:00][splashsurf][INFO]             relative_to_threshold_postprocessing: 7.18%, 0.39ms avg @ 184.17Hz (782 calls)
[2021-02-09T00:16:14.613502+01:00][splashsurf][INFO]             triangulate_with_criterion: 6.73%, 0.37ms avg @ 184.17Hz (782 calls)
[2021-02-09T00:16:14.613503+01:00][splashsurf][INFO]         stitch_surface_patches: 2.85%, 8.39ms avg @ 3.39Hz (138 calls)
[2021-02-09T00:16:14.613504+01:00][splashsurf][INFO]           stitch_children_orthogonal_to: 99.87%, 2.79ms avg @ 357.60Hz (414 calls)
[2021-02-09T00:16:14.613505+01:00][splashsurf][INFO]             stitch_surface_patches: 99.92%, 1.20ms avg @ 835.51Hz (966 calls)
[2021-02-09T00:16:14.613506+01:00][splashsurf][INFO]               interpolate_points_to_cell_data_skip_boundary: 17.32%, 0.21ms avg @ 836.16Hz (966 calls)
[2021-02-09T00:16:14.613507+01:00][splashsurf][INFO]                 generate_iso_surface_vertices: 99.82%, 0.21ms avg @ 4827.46Hz (966 calls)
[2021-02-09T00:16:14.613508+01:00][splashsurf][INFO]               relative_to_threshold_postprocessing: 2.45%, 0.03ms avg @ 836.16Hz (966 calls)
[2021-02-09T00:16:14.613509+01:00][splashsurf][INFO]               triangulate_with_criterion: 2.44%, 0.03ms avg @ 836.16Hz (966 calls)
[2021-02-09T00:16:14.613511+01:00][splashsurf][INFO]   write surface mesh to file: 9.08%, 266.61ms avg @ 0.34Hz (1 call)
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

### JSON

Files with the "`.json`" extension are interpreted as serializations of a `Vec<[f32; 3]>` where each three component array represents a particle position. This corresponds to a JSON file with a structure like this for example:
```json
[
    [1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0],
]
```

## Output file formats

Currently, only VTK and OBJ formats are supported to store the reconstructed surface meshes.

## All command line options

### The `reconstruct` command
```
splashsurf-reconstruct 0.7.0
Reconstruct a surface from particle data

USAGE:
    splashsurf reconstruct [OPTIONS] --cube-size <cube-size> --particle-radius <particle-radius> --smoothing-length <smoothing-length>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
        --check-mesh <check-mesh>
            Whether to check the final mesh for problems such as holes (note that when stitching is disabled this will
            lead to a lot of reported problems) [default: off]  [possible values: on, off]
        --cube-size <cube-size>
            The cube edge length used for marching cubes in multiplies of the particle radius, corresponds to the cell
            size of the implicit background grid
        --domain-max <domain-max> <domain-max> <domain-max>
            Upper corner of the domain where surface reconstruction should be performed, format:domain-
            max=x_max;y_max;z_max (requires domain-min to be specified)
        --domain-min <domain-min> <domain-min> <domain-min>
            Lower corner of the domain where surface reconstruction should be performed, format: domain-
            min=x_min;y_min;z_min (requires domain-max to be specified)
    -d, --double-precision <double-precision>
            Whether to enable the use of double precision for all computations [default: off]  [possible values: on,
            off]
    -i, --input-file <input-file>
            Path to the input file where the particle positions are stored (supported formats: VTK, binary f32 XYZ, PLY,
            BGEO)
    -s, --input-sequence <input-sequence>
            Path to a sequence of particle files that should be processed, use `{}` in the filename to indicate a
            placeholder
    -n, --num-threads <num-threads>
            Set the number of threads for the worker thread pool

        --octree-decomposition <octree-decomposition>
            Whether to enable spatial decomposition using an octree (faster) instead of a global approach [default: on]
            [possible values: on, off]
        --octree-ghost-margin-factor <octree-ghost-margin-factor>
            Safety factor applied to the kernel compact support radius when it's used as a margin to collect ghost
            particles in the leaf nodes when performing the spatial decomposition
        --octree-global-density <octree-global-density>
            Whether to compute particle densities in a global step before domain decomposition (slower) [default: off]
            [possible values: on, off]
        --octree-max-particles <octree-max-particles>
            The maximum number of particles for leaf nodes of the octree, default is to compute it based on the number
            of threads and particles
        --octree-stitch-subdomains <octree-stitch-subdomains>
            Whether to enable stitching of the disconnected local meshes resulting from the reconstruction when spatial
            decomposition is enabled (slower, but without stitching meshes will not be closed) [default: on]  [possible
            values: on, off]
        --octree-sync-local-density <octree-sync-local-density>
            Whether to compute particle densities per subdomain but synchronize densities for ghost-particles (faster,
            recommended). Note: if both this and global particle density computation is disabled the ghost particle
            margin has to be increased to at least 2.0 to compute correct density values for ghost particles [default:
            on]  [possible values: on, off]
        --output-dir <output-dir>
            Optional base directory for all output files (default: current working directory)

        --output-dm-grid <output-dm-grid>
            Optional filename for writing the grid representation of the intermediate density map to disk

        --output-dm-points <output-dm-points>
            Optional filename for writing the point cloud representation of the intermediate density map to disk

    -o <output-file>
            Filename for writing the reconstructed surface to disk (default: "{original_filename}_surface.vtk")

        --output-normals <output-normals>
            Whether to write vertex normals to the output file. Note that currently the normals are only computed using
            an area weighted average of triangle normals [default: off]  [possible values: on, off]
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
        --surface-threshold <surface-threshold>
            The iso-surface threshold for the density, i.e. the normalized value of the reconstructed density level that
            indicates the fluid surface (in multiplies of the rest density) [default: 0.6]
```

### The `convert` subcommand

Allows conversion between particle file formats and between mesh file formats. For particles `VTK, BGEO, PLY, XYZ, JSON -> VTK` 
is supported. For meshes only `VTK, PLY -> OBJ` is supported.

```
splashsurf-convert 0.7.0
Convert particle or mesh files between different file formats

USAGE:
    splashsurf convert [FLAGS] [OPTIONS] -o <output-file>

FLAGS:
    -h, --help         Prints help information
        --overwrite    Whether to overwrite existing files without asking
    -V, --version      Prints version information

OPTIONS:
        --domain-max <domain-max> <domain-max> <domain-max>
            Lower corner of the domain of particles to keep, format:domain-max=x_max;y_max;z_max (requires domain-min to
            be specified)
        --domain-min <domain-min> <domain-min> <domain-min>
            Lower corner of the domain of particles to keep, format: domain-min=x_min;y_min;z_min (requires domain-max
            to be specified)
        --mesh <input-mesh>
            Path to the input file with a surface to read (supported formats: .vtk, .ply)
        --particles <input-particles>
            Path to the input file with particles to read (supported formats: .vtk, .bgeo, .ply, .xyz, .json)

    -o <output-file>
            Path to the output file (supported formats for particles: .vtk, for meshes: .obj)
```

# License

For license information of this project, see the LICENSE file.
The splashsurf logo is based on two graphics ([1](https://www.svgrepo.com/svg/295647/wave), [2](https://www.svgrepo.com/svg/295652/surfboard-surfboard)) published on SVG Repo under a CC0 ("No Rights Reserved") license.
