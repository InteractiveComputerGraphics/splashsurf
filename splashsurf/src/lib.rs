//! Library target of the `splashsurf` CLI.
//!
//! To use the CLI you can install it via Cargo:
//! ```bash
//! cargo install splashsurf
//! ```
//! For documentation of the CLI see the [README](https://github.com/InteractiveComputerGraphics/splashsurf) in the project repository.
//!
//! This library target exposes some high-level functionality such as [running the CLI](cli) with a
//! list of command line arguments or specifically running the [reconstruction pipeline](reconstruct)
//! of the CLI (including a set of postprocessing steps) as a library function.
//! This functionality is mainly used by the `pySplashsurf` Pythong bindings and the CLI binary
//! target itself.
//!
//! If you only want to use a subset of this functionality (e.g. only the reconstruction itself,
//! without any postprocessing) in your crates, please refer to the [`splashsurf_lib`] crate instead.

pub mod cli;
mod convert;
mod io;
pub mod reconstruct;
#[macro_use]
mod allocator;
mod logging;

pub(crate) use register_counting_allocator;
