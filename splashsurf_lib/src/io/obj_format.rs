//! Helper functions for the OBJ file format

use crate::mesh::{AttributeData, CellConnectivity, Mesh3d, MeshWithData};
use crate::Real;
use anyhow::Context;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

// TODO: Support for mesh data, e.g. normals?

pub fn mesh_to_obj<R: Real, M: Mesh3d<R>, P: AsRef<Path>>(
    mesh: &MeshWithData<R, M>,
    filename: P,
) -> Result<(), anyhow::Error> {
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(filename)
        .context("Failed to open file handle for writing OBJ file")?;
    let mut writer = BufWriter::with_capacity(100000, file);

    let mesh_vertices = &mesh.mesh;

    for v in mesh_vertices.vertices() {
        write!(&mut writer, "v {} {} {}\n", v.x, v.y, v.z)?;
    }

    let normals = mesh
        .point_attributes
        .iter()
        .find(|attrib| attrib.name == "normals");

    if let Some(normals) = normals {
        match &normals.data {
            AttributeData::Vector3Real(normals) => {
                for n in normals {
                    write!(&mut writer, "vn {} {} {}\n", n.x, n.y, n.z)?;
                }
            }
            _ => {}
        }
    }

    if normals.is_some() {
        for f in mesh_vertices.cells() {
            write!(writer, "f")?;
            f.try_for_each_vertex(|v| write!(writer, " {}//{}", v + 1, v + 1))?;
            write!(writer, "\n")?;
        }
    } else {
        for f in mesh_vertices.cells() {
            write!(writer, "f")?;
            f.try_for_each_vertex(|v| write!(writer, " {}", v + 1))?;
            write!(writer, "\n")?;
        }
    }

    Ok(())
}
