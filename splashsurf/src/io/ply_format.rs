use std::path::Path;

use anyhow::{anyhow, Context};
use ply_rs as ply;
use ply_rs::ply::Property;

use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::Real;

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
