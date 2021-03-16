use anyhow::Context;
use splashsurf_lib::mesh::{CellConnectivity, Mesh3d};
use splashsurf_lib::Real;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

// TODO: Support for mesh data, e.g. normals?

pub fn write_mesh<R: Real, M: Mesh3d<R>, P: AsRef<Path>>(
    mesh: &M,
    filename: P,
) -> Result<(), anyhow::Error> {
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(filename)
        .context("Failed to open file handle for writing OBJ file")?;
    let mut writer = BufWriter::with_capacity(100000, file);

    for v in mesh.vertices() {
        write!(&mut writer, "v {} {} {}\n", v.x, v.y, v.z)?;
    }

    for f in mesh.cells() {
        write!(writer, "f")?;
        f.try_for_each_vertex(|v| write!(writer, " {}", v + 1))?;
        write!(writer, "\n")?;
    }

    Ok(())
}
