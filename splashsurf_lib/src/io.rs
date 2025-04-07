//! Convenience functions for importing particle data and meshes from various file formats

use crate::Real;
use anyhow::anyhow;
use nalgebra::Vector3;
use std::path::Path;

pub mod bgeo_format;
mod io_utils;
pub mod json_format;
pub mod obj_format;
pub mod ply_format;
pub mod vtk_format;
pub mod xyz_format;

/// Tries to load particles from the given file path, automatically detecting supported file extensions
pub fn particles_from_file<R: Real, P: AsRef<Path>>(
    input_file: P,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let input_file = input_file.as_ref();
    if let Some(extension) = input_file.extension() {
        let extension = extension
            .to_str()
            .ok_or(anyhow!("Invalid extension of input file"))?;

        match extension.to_lowercase().as_str() {
            "vtk" => vtk_format::particles_from_vtk(input_file),
            "vtu" => vtk_format::particles_from_vtk(input_file),
            "xyz" => xyz_format::particles_from_xyz(input_file),
            "ply" => ply_format::particles_from_ply(input_file),
            "bgeo" => bgeo_format::particles_from_bgeo(input_file),
            "json" => json_format::particles_from_json(input_file),
            _ => Err(anyhow!(
                "Unsupported file format extension \"{}\" for reading particles",
                extension
            )),
        }
    } else {
        Err(anyhow!(
            "Unable to detect file format of particle input file (file name has to end with supported extension)",
        ))
    }
}
