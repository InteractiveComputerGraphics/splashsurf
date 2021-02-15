use anyhow::Context;
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::Real;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

type ParticleVecF32 = Vec<[f32; 3]>;

pub fn particles_from_json<R: Real, P: AsRef<Path>>(
    json_file: P,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let path = json_file.as_ref();
    let file = File::open(path).context("Cannot open file for JSON parsing")?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `User`.
    let json =
        serde_json::from_reader(reader).context("Reading of file to JSON structure failed")?;
    let particles = serde_json::from_value::<ParticleVecF32>(json)
        .context("Parsing of JSON structure as particle positions failed")?;

    let particles = particles
        .into_iter()
        .map(|raw_particle| {
            Vector3::new(
                R::from_f32(raw_particle[0]).unwrap(),
                R::from_f32(raw_particle[1]).unwrap(),
                R::from_f32(raw_particle[2]).unwrap(),
            )
        })
        .collect();

    Ok(particles)
}
