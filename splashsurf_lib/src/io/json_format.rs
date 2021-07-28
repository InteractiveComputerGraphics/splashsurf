//! Helper functions for the JSON file format

use crate::utils::IteratorExt;
use crate::Real;
use anyhow::{anyhow, Context};
use nalgebra::Vector3;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

type ParticleVecF64 = Vec<[f64; 3]>;

/// Convenience function for loading particles from a JSON file
///
/// The particle coordinates are expected to be stored in an array of arrays. For Example:
/// ```json
/// [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
/// ```
pub fn particles_from_json<R: Real, P: AsRef<Path>>(
    json_file: P,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let path = json_file.as_ref();
    let file = File::open(path).context("Cannot open file for JSON parsing")?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `ParticleVecF32`.
    let json = serde_json::from_reader(reader)
        .context("Reading of file to JSON structure failed. Not a valid JSON file.")?;
    let particles = serde_json::from_value::<ParticleVecF64>(json)
        .context("Parsing of JSON structure as particle positions failed. Expected JSON file containing particle positions like e.g. '[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]'.")?;

    let len = particles.len();
    let particles = particles
        .into_iter()
        .map(|raw_particle| {
            Some(Vector3::new(
                raw_particle[0].try_convert()?,
                raw_particle[1].try_convert()?,
                raw_particle[2].try_convert()?,
            ))})
                .map(|vec| {
                    vec.ok_or_else(|| {
                        anyhow!("Failed to convert coordinate from input to output float type, value out of range?")
                    })
                })
        .try_collect_with_capacity(len)?;

    Ok(particles)
}
