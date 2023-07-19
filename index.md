[![On crates.io](https://img.shields.io/crates/v/splashsurf)](https://crates.io/crates/splashsurf)
[![On docs.rs](https://docs.rs/splashsurf_lib/badge.svg)](https://docs.rs/splashsurf_lib)
[![Commits since last release](https://img.shields.io/github/commits-since/InteractiveComputerGraphics/splashsurf/latest)](https://github.com/InteractiveComputerGraphics/splashsurf)
[![License: MIT](https://img.shields.io/crates/l/splashsurf)](https://github.com/InteractiveComputerGraphics/splashsurf/blob/main/LICENSE)
[![Dependency status](https://deps.rs/repo/github/InteractiveComputerGraphics/splashsurf/status.svg)](https://deps.rs/repo/github/InteractiveComputerGraphics/splashsurf)
[![Build and test GitHub Actions workflow](https://github.com/InteractiveComputerGraphics/splashsurf/workflows/Build%20and%20test/badge.svg)](https://github.com/InteractiveComputerGraphics/splashsurf/actions/workflows/build.yml)

This project consists of the following crates:
 - `splashsurf`: Binary crate with a CLI (command line interface) to quickly run surface reconstructions of SPH particle data files from the terminal
 - `splashsurf_lib`: Library that implements the reconstruction pipeline used by the CLI. Allows to integrate the reconstruction procedure directly into other Rust applications. Furthermore, it resembles a framework providing access to individual building blocks to create your own surface reconstruction pipeline.

<p align="center">
<img src="assets/example_particles.png" alt="Image of the original particle data" width="32%"> <img src="assets/example_coarse.png" alt="Image of a coarse reconstructed surface mesh" width="32%"> <img src="assets/example_fine.png" alt="Image of a fine reconstructed surface mesh" width="32%">
</p>

`splashsurf` is a tool to reconstruct surfaces meshes from SPH particle data.
The first image shows the visualization of a set of particles from an SPH fluid simulation from [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH).
The particle radius is `0.025`. As the rendering of a fluid should not look like a ball pit, a surface mesh has to be
reconstructed from this particle data. The next image shows a reconstructed surface mesh of the fluid produced by `splashsurf`
with a "smoothing length" of `2.2` times the particles radius and a cell size of `1.1` times the particle radius. The
third image shows a finer reconstruction with a cell size of `0.45` times the particle radius. These surface meshes can
then be fed into 3D rendering software such as [Blender](https://www.blender.org/) to generate beautiful water animations.
The result might look something like this:

<p align="center">
<img src="assets/splashsurf.gif" alt="Rendered water animation" width="96%">
</p>
