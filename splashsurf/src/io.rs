use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{anyhow, Context};
use coarse_prof::profile;
use log::info;
use ply_rs as ply;
use ply_rs::ply::Property;

use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::Real;

pub mod vtk_format;
pub mod xyz_format;

/// File format parameters for all io
#[derive(Clone, Debug, Default)]
pub struct FormatParameters {
    pub input: InputFormatParameters,
    pub output: OutputFormatParameters,
}

/// File format parameters for input files
#[derive(Clone, Debug)]
pub struct InputFormatParameters {}

impl Default for InputFormatParameters {
    fn default() -> Self {
        Self {}
    }
}

/// File format parameters for output files
#[derive(Clone, Debug)]
pub struct OutputFormatParameters {}

impl Default for OutputFormatParameters {
    fn default() -> Self {
        Self {}
    }
}

/// Loads particles positions form the given file path, automatically detects the file format
pub fn load_particle_positions<R: Real, P: AsRef<Path>>(
    input_file: P,
    _format_params: &InputFormatParameters,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let input_file = input_file.as_ref();
    info!("Loading dataset from \"{}\"...", input_file.display());

    let particle_positions = if let Some(extension) = input_file.extension() {
        profile!("loading particle positions");

        let extension = extension
            .to_str()
            .ok_or(anyhow!("Invalid extension of particle file"))?;

        match extension.to_lowercase().as_str() {
            "vtk" => vtk_format::particles_from_vtk(&input_file)?,
            "xyz" => xyz_format::particles_from_xyz(&input_file)?,
            "ply" => particles_from_ply(&input_file)?,
            _ => {
                return Err(anyhow!(
                    "Unsupported file format extension \"{}\" of particle file",
                    extension
                ));
            }
        }
    } else {
        return Err(anyhow!(
            "Unable to detect file format of particle file (file name has to end with supported extension)",
        ));
    };

    info!(
        "Loaded dataset with {} particle positions.",
        particle_positions.len()
    );

    Ok(particle_positions)
}

pub fn particles_from_ply<R: Real, P: AsRef<Path>>(
    ply_file: P,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let mut ply_file = std::fs::File::open(ply_file).unwrap();
    let parser = ply::parser::Parser::<ply::ply::DefaultElement>::new();

    let ply = parser
        .read_ply(&mut ply_file)
        .context("Failed to read PLY file")?;
    let elements = ply
        .payload
        .get("vertex")
        .ok_or(anyhow!("PLY file is missing a 'vertex' element"))?;

    let particles = elements
        .into_iter()
        .map(|e| {
            let vertex = (
                e.get("x").unwrap(),
                e.get("y").unwrap(),
                e.get("z").unwrap(),
            );

            let v = match vertex {
                (Property::Float(x), Property::Float(y), Property::Float(z)) => Vector3::new(
                    R::from_f32(*x).unwrap(),
                    R::from_f32(*y).unwrap(),
                    R::from_f32(*z).unwrap(),
                ),
                _ => {
                    return Err(anyhow!(
                        "Vertex properties have wrong PLY data type (expected float)"
                    ))
                }
            };

            Ok(v)
        })
        .collect::<Result<Vec<_>, anyhow::Error>>()?;

    Ok(particles)
}

#[allow(dead_code)]
pub fn to_binary_f32<R: Real, P: AsRef<Path>>(file: P, values: &[R]) -> Result<(), anyhow::Error> {
    let file = file.as_ref();
    let file = File::create(file).context("Unable to create binary file")?;
    let mut writer = BufWriter::new(file);

    for v in values {
        let v_f32 = v.to_f32().unwrap();
        writer.write(&v_f32.to_ne_bytes())?;
    }

    Ok(())
}
