[![On crates.io](https://img.shields.io/crates/v/splashsurf)](https://crates.io/crates/splashsurf)
[![On PyPI](https://img.shields.io/pypi/v/pysplashsurf)](https://pypi.org/project/pysplashsurf)
[![On docs.rs](https://docs.rs/splashsurf_lib/badge.svg)](https://docs.rs/splashsurf_lib)
[![Commits since last release](https://img.shields.io/github/commits-since/InteractiveComputerGraphics/splashsurf/latest)](https://github.com/InteractiveComputerGraphics/splashsurf)
[![License: MIT](https://img.shields.io/crates/l/splashsurf)](https://github.com/InteractiveComputerGraphics/splashsurf/blob/main/LICENSE)
[![Dependency status](https://deps.rs/repo/github/InteractiveComputerGraphics/splashsurf/status.svg)](https://deps.rs/repo/github/InteractiveComputerGraphics/splashsurf)
[![Rust library & CLI](https://github.com/InteractiveComputerGraphics/splashsurf/actions/workflows/build.yml/badge.svg)](https://github.com/InteractiveComputerGraphics/splashsurf/actions/workflows/build.yml)
[![Python bindings](https://github.com/InteractiveComputerGraphics/splashsurf/actions/workflows/pysplashsurf_CI.yml/badge.svg)](https://github.com/InteractiveComputerGraphics/splashsurf/actions/workflows/pysplashsurf_CI.yml)

Surface reconstruction library and CLI for particle data from SPH simulations, written in Rust.

---

## News

**2025-06-25**: We published Python bindings for the CLI of splashsurf and most library features 🐍! Pre-built wheels can be installed using PIP: `pip install pysplashsurf`. See the [PyPI entry of pySplashsurf](https://pypi.org/project/pysplashsurf) for more information.

**2023-09-25**: The project now implements the paper ["Weighted Laplacian Smoothing for Surface Reconstruction of Particle-based Fluids" (Löschner, Böttcher, Jeske, Bender; 2023)](https://animation.rwth-aachen.de/publication/0583/). 
It proposes a fast smoothing approach to avoid the typical bumpiness of the surfaces reconstructed from SPH simulations while preventing loss of volume for splashes and droplets that occurs with simple, non-specialized smoothing methods.
The images below show a rendering of a typical surface reconstruction (on the left) with visible bumps due to the SPH particles compared to the same surface reconstruction with weighted smoothing applied (on the right):

<p align="center">
<img src="assets/example_unsmoothed.jpg" alt="Image of the original surface reconstruction without smoothing (bumpy & rough)" width="48%"> <img src="assets/example_smoothed.jpg" alt="Image of the surface reconstruction with weighted smoothing applied (nice & smooth)" width="48%">
</p>

You can see this rendering in motion in [this video](https://youtu.be/2bYvaUXlBQs). For more details see the [paper](https://animation.rwth-aachen.de/publication/0583/) and the [readme](https://github.com/InteractiveComputerGraphics/splashsurf) in the repository.

---

## About

This project consists of the following crates:
 - 🛠️ `splashsurf`: Binary crate with a CLI (command line interface) to quickly run surface reconstructions of SPH particle data files from the terminal. Install with `cargo install splashsurf`.
 - 🧰 `splashsurf_lib`: Rust library that implements the reconstruction method used by the CLI. Allows integrating the reconstruction procedure directly into other Rust applications. Furthermore, it resembles a framework providing access to individual building blocks to create your own surface reconstruction pipeline.
 - 🐍 `pysplashsurf`: Bindings to the CLI and library for Python. Install with `pip install splashsurf` and see the [`README`](pysplashsurf/README.md) for more details.

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

For more information please refer to the [GitHub repository of splashsurf](https://github.com/InteractiveComputerGraphics/splashsurf).

## Acknowledgements

This project contains notable contributions from the following people:
 - [Timna Böttcher](https://animation.rwth-aachen.de/person/80/) ([@timnaboettcher](https://github.com/timnaboettcher)): co-developed the [weighted smoothing approach](https://diglib.eg.org/handle/10.2312/vmv20231245)
 - [Felix Kern](https://github.com/Fek04) ([@Fek04](https://github.com/Fek04)): implemented the Python bindings for `splashsurf`
 - [Fabian Löschner](https://www.floeschner.de/) ([@w1th0utnam3](https://github.com/w1th0utnam3)): implemented most of the surface reconstruction library and CLI including the domain decomposition
