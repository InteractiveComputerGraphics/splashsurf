//! Helper functions for the JSON file format

use crate::io::io_utils::IteratorExt;
use crate::{Real, RealConvert};
use anyhow::{Context, anyhow};
use nalgebra::Vector3;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};
use std::path::Path;

type ParticleF64 = [f64; 3];
type ParticleVecF64 = Vec<ParticleF64>;

/// Convenience function for loading particles from a JSON file
///
/// The particle coordinates are expected to be stored in an array of arrays, for example:
/// ```json
/// [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
/// ```
/// Note that the values are casted to `f64` during deserialization.
pub fn particles_from_json<R: Real, P: AsRef<Path>>(
    json_file: P,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let path = json_file.as_ref();
    let file = File::open(path).context("Cannot open file for JSON parsing")?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `ParticleVecF64`.
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
                        anyhow!("Failed to convert coordinate from f64 to output float type, value out of range?")
                    })
                })
        .try_collect_with_capacity(len)?;

    Ok(particles)
}

/// Tries to write a set of particles to a JSON file at the given path
///
/// The particle coordinates are expected will be stored as an array of arrays, for example:
/// ```json
/// [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
/// ```
pub fn particles_to_json<R: Real, P: AsRef<Path>>(
    particles: &[Vector3<R>],
    json_file: P,
) -> Result<(), anyhow::Error> {
    let path = json_file.as_ref();
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .context("Cannot open file for writing JSON")?;
    let writer = BufWriter::new(file);

    let particles = particles
        .iter()
        .map(|particle| {
            Some([
                particle[0].to_f64()?,
                particle[1].to_f64()?,
                particle[2].to_f64()?,
            ])
        })
        .map(|vec| {
            vec.ok_or_else(|| {
                anyhow!(
                    "Failed to convert coordinate from input float type to f64, value out of range?"
                )
            })
        })
        .try_collect_with_capacity(particles.len())?;

    serde_json::to_writer(writer, &particles)
        .context("Failed to deserialize particles to JSON file")?;

    Ok(())
}
