//! Helper functions for the PLY file format

use crate::io::io_utils::IteratorExt;
use crate::mesh::{
    AttributeData, CellConnectivity, Mesh3d, MeshAttribute, MeshWithData, TriMesh3d,
};
use crate::{Real, RealConvert, profile};
use anyhow::{Context, anyhow};
use nalgebra::Vector3;
use num_traits::ToPrimitive;
use ply_rs::parser::Parser as PlyParser;
use ply_rs::ply::{DefaultElement, Ply, Property};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Tries to load the file at the given path as a PLY file and read particle positions from it
pub fn particles_from_ply<R: Real, P: AsRef<Path>>(
    ply_path: P,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let ply = PlyParser::new()
        .read_ply(&mut fs::File::open(ply_path).context("Failed to open file for reading")?)
        .context("Failed to parse PLY file")?;
    parse_particles_from_ply(&ply)
}

/// Tries to load the file at the given path as a PLY file and read a surface mesh from it
pub fn surface_mesh_from_ply<R: Real, P: AsRef<Path>>(
    ply_path: P,
) -> Result<MeshWithData<R, TriMesh3d<R>>, anyhow::Error> {
    let ply = PlyParser::new()
        .read_ply(&mut fs::File::open(ply_path).context("Failed to open file for reading")?)
        .context("Failed to parse PLY file")?;
    parse_mesh_from_ply(&ply)
}

/// Tries to extract particle positions from the given PLY structure
fn parse_particles_from_ply<R: Real>(
    ply_file: &Ply<DefaultElement>,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let elements = ply_file
        .payload
        .get("vertex")
        .ok_or(anyhow!("PLY file is missing a 'vertex' element"))?;

    let particles = elements
        .iter()
        .map(|e| {
            let vertex = (
                e.get("x").unwrap(),
                e.get("y").unwrap(),
                e.get("z").unwrap(),
            );

            let v = match vertex {
                (Property::Float(x), Property::Float(y), Property::Float(z)) => {
                    Vector3::new(*x, *y, *z).try_convert().unwrap()
                }
                _ => {
                    return Err(anyhow!(
                        "Vertex properties have wrong PLY data type (expected float)"
                    ));
                }
            };

            Ok(v)
        })
        .collect::<Result<Vec<_>, anyhow::Error>>()?;

    Ok(particles)
}

/// Tries to extract a surface mesh from the given PLY structure
///
/// The PLY file is expected to use the following structure which is used by Blender for export:
/// ```text
/// element vertex 24
/// property float x
/// property float y
/// property float z
/// property float nx
/// property float ny
/// property float nz
/// property float s
/// property float t
/// element face *
/// property list uchar uint vertex_indices
/// ```
fn parse_mesh_from_ply<R: Real>(
    ply_file: &Ply<DefaultElement>,
) -> Result<MeshWithData<R, TriMesh3d<R>>, anyhow::Error> {
    // TODO: Support other attributes
    // TODO: Support other data types

    let vertices_normals = ply_file
        .payload
        .get("vertex")
        .ok_or(anyhow!("PLY file is missing a 'vertex' element"))?;

    let mut vertices = Vec::with_capacity(vertices_normals.len());

    let load_vec3_property = |vertex: (&Property, &Property, &Property)| match vertex {
        (Property::Float(x), Property::Float(y), Property::Float(z)) => {
            Ok(Vector3::new(*x, *y, *z).try_convert().unwrap())
        }
        _ => Err(anyhow!(
            "Vertex properties have wrong PLY data type (expected float)"
        )),
    };

    for e in vertices_normals {
        let vertex = (
            e.get("x").unwrap(),
            e.get("y").unwrap(),
            e.get("z").unwrap(),
        );

        let vertex = load_vec3_property(vertex)?;

        vertices.push(vertex);
    }

    let faces = ply_file
        .payload
        .get("face")
        .ok_or(anyhow!("PLY file is missing a 'face' element"))?;

    let triangles = faces
        .iter()
        .map(|e| {
            let indices = e
                .get("vertex_indices")
                .ok_or_else(|| anyhow!("A face is missing a 'vertex_indices' element"))?;
            match indices {
                Property::ListUInt(indices) if indices.len() == 3 => Ok([
                    indices[0] as usize,
                    indices[1] as usize,
                    indices[2] as usize,
                ]),
                Property::ListUInt(indices) => Err(anyhow!(
                    "Invalid number of vertex indices per face: {} (expected 3)",
                    indices.len()
                )),
                _ => Err(anyhow!(
                    "Index properties have wrong PLY data type (expected uint)"
                )),
            }
        })
        .try_collect_with_capacity(faces.len())?;

    let mut mesh = MeshWithData::new(TriMesh3d {
        vertices,
        triangles,
    });

    {
        let vertex_properties = &ply_file.header.elements.get("vertex").unwrap().properties;
        let contains_normals = vertex_properties.contains_key("nx")
            && vertex_properties.contains_key("ny")
            && vertex_properties.contains_key("nz");

        if contains_normals {
            let mut normals = Vec::with_capacity(vertices_normals.len());

            for e in vertices_normals {
                let vertex = (
                    e.get("nx").unwrap(),
                    e.get("ny").unwrap(),
                    e.get("nz").unwrap(),
                );

                let normal = load_vec3_property(vertex)?;

                normals.push(normal);
            }

            let normals =
                MeshAttribute::new("normals".to_string(), AttributeData::Vector3Real(normals));
            mesh.point_attributes.push(normals);
        }
    }

    Ok(mesh)
}

/// Tries to write a mesh with attributes into a little endian PLY file
#[rustfmt::skip]
pub fn mesh_to_ply<R: Real, M: Mesh3d<R>, P: AsRef<Path>>(
    mesh: &MeshWithData<R, M>,
    filename: P,
) -> Result<(), anyhow::Error> {
    // TODO: Support attributes
    profile!("mesh_to_ply");
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(filename)
        .context("Failed to open file handle for writing PLY file")?;
    let mut writer = BufWriter::with_capacity(1000000, file);

    writeln!(&mut writer, "ply")?;
    writeln!(&mut writer, "format binary_little_endian 1.0")?;
    writeln!(&mut writer, "element vertex {}", mesh.vertices().len())?;
    writeln!(&mut writer, "property float x")?;
    writeln!(&mut writer, "property float y")?;
    writeln!(&mut writer, "property float z")?;
    for p_attr in &mesh.point_attributes {
        if p_attr.name == "normals" {
            writeln!(&mut writer, "property float nx")?;
            writeln!(&mut writer, "property float ny")?;
            writeln!(&mut writer, "property float nz")?;
        } else {
            match p_attr.data {
                AttributeData::ScalarU64(_) => writeln!(&mut writer, "property uint {}", p_attr.name)?,
                AttributeData::ScalarReal(_) => writeln!(&mut writer, "property float {}", p_attr.name)?,
                AttributeData::Vector3Real(_) => {
                    writeln!(&mut writer, "property float {}_x", p_attr.name)?;
                    writeln!(&mut writer, "property float {}_y", p_attr.name)?;
                    writeln!(&mut writer, "property float {}_z", p_attr.name)?;
                },
            }
        }
    }
    writeln!(&mut writer, "element face {}", mesh.cells().len())?;
    writeln!(&mut writer, "property list uchar uint vertex_indices")?;
    writeln!(&mut writer, "end_header")?;

    for (i, v) in mesh.vertices().iter().enumerate() {
        writer.write_all(&v.x.to_f32().expect("failed to convert coordinate to f32").to_le_bytes())?;
        writer.write_all(&v.y.to_f32().expect("failed to convert coordinate to f32").to_le_bytes())?;
        writer.write_all(&v.z.to_f32().expect("failed to convert coordinate to f32").to_le_bytes())?;

        for p_attr in &mesh.point_attributes {
            match &p_attr.data {
                AttributeData::ScalarU64(data) => {
                    let val = data[i].to_u32().expect("failed to convert attribute to u32");
                    writer.write_all(&val.to_le_bytes())?;
                },
                AttributeData::ScalarReal(data) => {
                    let val = data[i].to_f32().expect("failed to convert attribute to f32");
                    writer.write_all(&val.to_le_bytes())?;
                },
                AttributeData::Vector3Real(data) => {
                    let val = &data[i];
                    writer.write_all(&val.x.to_f32().expect("failed to convert attribute to f32").to_le_bytes())?;
                    writer.write_all(&val.y.to_f32().expect("failed to convert attribute to f32").to_le_bytes())?;
                    writer.write_all(&val.z.to_f32().expect("failed to convert attribute to f32").to_le_bytes())?;
                },
            }
        }
    }

    for c in mesh.cells() {
        let num_verts = c.num_vertices().to_u8().expect("failed to convert cell vertex count to u8");
        writer.write_all(&num_verts.to_le_bytes())?;
        c.vertices().iter().copied().try_for_each(|v| {
            let idx = v.to_u32().expect("failed to convert vertex index to u32");
            writer.write_all(&idx.to_le_bytes())
        })?;
    }

    Ok(())
}

#[cfg(test)]
pub mod test {
    use super::*;

    #[test]
    fn test_ply_read_cube() -> Result<(), anyhow::Error> {
        let input_file = Path::new("../data/cube.ply");

        let mesh: MeshWithData<f32, _> = surface_mesh_from_ply(input_file).with_context(|| {
            format!(
                "Failed to load surface mesh from file \"{}\"",
                input_file.display()
            )
        })?;

        assert_eq!(mesh.mesh.vertices.len(), 24);
        assert_eq!(mesh.mesh.triangles.len(), 12);

        Ok(())
    }

    #[test]
    fn test_ply_read_cube_with_normals() -> Result<(), anyhow::Error> {
        let input_file = Path::new("../data/cube_normals.ply");

        let mesh: MeshWithData<f32, _> = surface_mesh_from_ply(input_file).with_context(|| {
            format!(
                "Failed to load surface mesh from file \"{}\"",
                input_file.display()
            )
        })?;

        assert_eq!(mesh.mesh.vertices.len(), 24);
        assert_eq!(mesh.mesh.triangles.len(), 12);
        let normals = mesh.point_attributes.iter().find(|a| a.name == "normals");
        if let Some(MeshAttribute { data, .. }) = normals {
            if let AttributeData::Vector3Real(normals) = data {
                assert_eq!(normals.len(), 24)
            }
        }

        Ok(())
    }
}
