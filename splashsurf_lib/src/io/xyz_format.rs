//! Helper functions for the binary `.xyz` float coordinate format

use crate::{Real, RealConvert};
use anyhow::Context;
use nalgebra::Vector3;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

pub fn particles_from_xyz<R: Real, P: AsRef<Path>>(
    xyz_file: P,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let file = File::open(xyz_file).context("Unable to open XYZ file for reading")?;
    let mut reader = BufReader::new(file);

    let mut buffer = [0u8; 3 * 4];

    let get_four_bytes = |buffer: &[u8], offset: usize| -> [u8; 4] {
        [
            buffer[offset],
            buffer[offset + 1],
            buffer[offset + 2],
            buffer[offset + 3],
        ]
    };

    let mut particles = Vec::new();

    while reader.read_exact(&mut buffer).is_ok() {
        let x = f32::from_ne_bytes(get_four_bytes(&buffer, 0));
        let y = f32::from_ne_bytes(get_four_bytes(&buffer, 4));
        let z = f32::from_ne_bytes(get_four_bytes(&buffer, 8));
        particles.push(Vector3::new(x, y, z).try_convert().unwrap());
    }

    Ok(particles)
}
