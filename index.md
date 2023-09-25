[![On crates.io](https://img.shields.io/crates/v/splashsurf)](https://crates.io/crates/splashsurf)
[![On docs.rs](https://docs.rs/splashsurf_lib/badge.svg)](https://docs.rs/splashsurf_lib)
[![Commits since last release](https://img.shields.io/github/commits-since/InteractiveComputerGraphics/splashsurf/latest)](https://github.com/InteractiveComputerGraphics/splashsurf)
[![License: MIT](https://img.shields.io/crates/l/splashsurf)](https://github.com/InteractiveComputerGraphics/splashsurf/blob/main/LICENSE)
[![Dependency status](https://deps.rs/repo/github/InteractiveComputerGraphics/splashsurf/status.svg)](https://deps.rs/repo/github/InteractiveComputerGraphics/splashsurf)
[![Build and test GitHub Actions workflow](https://github.com/InteractiveComputerGraphics/splashsurf/workflows/Build%20and%20test/badge.svg)](https://github.com/InteractiveComputerGraphics/splashsurf/actions/workflows/build.yml)

---
**NEW**: The project now implements the paper ["Weighted Laplacian Smoothing for Surface Reconstruction of Particle-based Fluids" (Löschner, Böttcher, Jeske, Bender; 2023)](https://animation.rwth-aachen.de/publication/0583/). 
It proposes a fast smoothing approach to avoid the typical bumpiness of the surfaces reconstructed from SPH simulations while preventing loss of volume for splashes and droplets that occurs with simple, non-speaclized smoothing methods.
The images below show a rendering of a typical surface reconstruction (on the right) with visible bumps due to the SPH particles compared to the same surface reconstruction with weighted smoothing applied (on the left):

<p align="center">
<img src="assets/example_unsmoothed.jpg" alt="Image of the original surface reconstruction without smoothing (bumpy & rough)" width="48%"> <img src="assets/example_smoothed.jpg" alt="Image of the surface reconstruction with weighted smoothing applied (nice & smooth)" width="48%">
</p>

You can see this rendering in motion in [this video](https://youtu.be/2bYvaUXlBQs). For more details see the [paper](https://animation.rwth-aachen.de/publication/0583/) and the [readme](https://github.com/InteractiveComputerGraphics/splashsurf) in the repository.

---


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
