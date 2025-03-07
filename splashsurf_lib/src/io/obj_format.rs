//! Helper functions for the OBJ file format

use crate::mesh::{
    AttributeData, CellConnectivity, Mesh3d, MeshAttribute, MeshWithData, TriMesh3d,
};
use crate::{Real, RealConvert, io::io_utils, profile};
use anyhow::Context;
use nalgebra::Vector3;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

// TODO: Support for other mesh data (interpolated fields)?

/// Writes the given mesh to an OBJ file, supports outputting normals
pub fn mesh_to_obj<R: Real, M: Mesh3d<R>, P: AsRef<Path>>(
    mesh: &MeshWithData<R, M>,
    filename: P,
) -> Result<(), anyhow::Error> {
    profile!("mesh_to_obj");
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
        writeln!(&mut writer, "v {} {} {}", v.x, v.y, v.z)?;
    }

    let normals = mesh
        .point_attributes
        .iter()
        .find(|attrib| attrib.name == "normals");

    if let Some(normals) = normals {
        if let AttributeData::Vector3Real(normals) = &normals.data {
            for n in normals {
                writeln!(&mut writer, "vn {} {} {}", n.x, n.y, n.z)?;
            }
        }
    }

    if normals.is_some() {
        for f in mesh_vertices.cells() {
            write!(writer, "f")?;
            f.vertices()
                .iter()
                .copied()
                .try_for_each(|v| write!(writer, " {}//{}", v + 1, v + 1))?;
            writeln!(writer)?;
        }
    } else {
        for f in mesh_vertices.cells() {
            write!(writer, "f")?;
            f.vertices()
                .iter()
                .copied()
                .try_for_each(|v| write!(writer, " {}", v + 1))?;
            writeln!(writer)?;
        }
    }

    Ok(())
}

pub fn surface_mesh_from_obj<R: Real, P: AsRef<Path>>(
    obj_path: P,
) -> Result<MeshWithData<R, TriMesh3d<R>>, anyhow::Error> {
    let file = fs::File::open(obj_path).context("Failed to open file for reading")?;
    let mut reader = BufReader::with_capacity(1000000, file);

    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let mut normals = Vec::new();

    let buffer_to_vec3 = |buffer: &[&str]| -> Result<Vector3<R>, anyhow::Error> {
        Ok(Vector3::new(
            f64::from_str(buffer[0])?,
            f64::from_str(buffer[1])?,
            f64::from_str(buffer[2])?,
        )
        .try_convert()
        .unwrap())
    };

    let mut outer_buffer: Vec<&'static str> = Vec::new();
    let mut buffer_string = String::new();

    loop {
        let mut buffer = io_utils::recycle(outer_buffer);

        let read = reader.read_line(&mut buffer_string)?;
        if read == 0 {
            break;
        }

        let line = buffer_string.trim();

        if let Some(vert_string) = line.strip_prefix("v ") {
            buffer.extend(vert_string.split(' '));
            assert_eq!(buffer.len(), 3, "expected three coordinates per vertex");
            vertices.push(buffer_to_vec3(&buffer)?);
        } else if let Some(face_string) = line.strip_prefix("f ") {
            // TODO: Support mixed tri/quad meshes?
            buffer.extend(
                face_string
                    .split(' ')
                    // Support "v1/vt1", "v1/vt1/vn1" and "v1//vn1" formats (ignore everything after '/')
                    .map(|f| f.split_once('/').map(|(f, _)| f).unwrap_or(f)),
            );
            assert_eq!(
                buffer.len(),
                3,
                "expected three indices per faces (only triangles supported at the moment)"
            );
            let tri = [
                usize::from_str(buffer[0])? - 1,
                usize::from_str(buffer[1])? - 1,
                usize::from_str(buffer[2])? - 1,
            ];
            triangles.push(tri);
        } else if let Some(normal_string) = line.strip_prefix("vn ") {
            buffer.extend(normal_string.split(' '));
            assert_eq!(
                buffer.len(),
                3,
                "expected three normal components per vertex"
            );
            normals.push(buffer_to_vec3(&buffer)?);
        }

        outer_buffer = io_utils::recycle(buffer);
        buffer_string.clear();
    }

    let mut mesh = MeshWithData::new(TriMesh3d {
        vertices,
        triangles,
    });

    if !normals.is_empty() {
        assert_eq!(
            mesh.vertices().len(),
            normals.len(),
            "length of vertex and vertex normal array doesn't match"
        );
        mesh.point_attributes.push(MeshAttribute::new(
            "normals",
            AttributeData::Vector3Real(normals),
        ));
    }

    Ok(mesh)
}

#[cfg(test)]
pub mod test {
    use super::*;

    #[test]
    fn test_obj_read_icosphere() -> Result<(), anyhow::Error> {
        let mesh = surface_mesh_from_obj::<f32, _>("../data/icosphere.obj")?;

        assert_eq!(mesh.vertices().len(), 42);
        assert_eq!(mesh.cells().len(), 80);

        Ok(())
    }

    #[test]
    fn test_obj_read_icosphere_with_normals() -> Result<(), anyhow::Error> {
        let mesh = surface_mesh_from_obj::<f32, _>("../data/icosphere.obj")?;

        assert_eq!(mesh.vertices().len(), 42);
        assert_eq!(mesh.cells().len(), 80);
        let normals = mesh.point_attributes.iter().find(|a| a.name == "normals");
        if let Some(MeshAttribute { data, .. }) = normals {
            if let AttributeData::Vector3Real(normals) = data {
                assert_eq!(normals.len(), 42)
            }
        }

        Ok(())
    }
}
