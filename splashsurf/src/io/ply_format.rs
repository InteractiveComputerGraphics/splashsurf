use std::path::Path;

use anyhow::{anyhow, Context};
use ply_rs as ply;
use ply_rs::ply::Property;

use splashsurf_lib::mesh::AttributeData;
use splashsurf_lib::mesh::MeshAttribute;
use splashsurf_lib::mesh::MeshWithData;
use splashsurf_lib::mesh::TriMesh3d;
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::Real;

/// Tries to read particle positions from the PLY file at the given path
pub fn particles_from_ply<R: Real, P: AsRef<Path>>(
    ply_path: P,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let mut ply_file = std::fs::File::open(ply_path).unwrap();
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

/// Tries to read a surface mesh from the PLY file at the given path
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
/// element face 12
/// property list uchar uint vertex_indices
/// ```
pub fn surface_mesh_from_ply<R: Real, P: AsRef<Path>>(
    ply_path: P,
) -> Result<MeshWithData<R, TriMesh3d<R>>, anyhow::Error> {
    let mut ply_file = std::fs::File::open(ply_path).unwrap();
    let parser = ply::parser::Parser::<ply::ply::DefaultElement>::new();

    let ply = parser
        .read_ply(&mut ply_file)
        .context("Failed to parse PLY file")?;
    let vertices_normals = ply
        .payload
        .get("vertex")
        .ok_or(anyhow!("PLY file is missing a 'vertex' element"))?;

    let mut vertices = Vec::with_capacity(vertices_normals.len());
    let mut normals = Vec::with_capacity(vertices_normals.len());

    for e in vertices_normals {
        let vertex = (
            e.get("x").unwrap(),
            e.get("y").unwrap(),
            e.get("z").unwrap(),
            e.get("nx").unwrap(),
            e.get("ny").unwrap(),
            e.get("nz").unwrap(),
        );

        let (vertex, normal) = match vertex {
            (
                Property::Float(x),
                Property::Float(y),
                Property::Float(z),
                Property::Float(nx),
                Property::Float(ny),
                Property::Float(nz),
            ) => Ok((
                Vector3::new(
                    R::from_f32(*x).unwrap(),
                    R::from_f32(*y).unwrap(),
                    R::from_f32(*z).unwrap(),
                ),
                Vector3::new(
                    R::from_f32(*nx).unwrap(),
                    R::from_f32(*ny).unwrap(),
                    R::from_f32(*nz).unwrap(),
                ),
            )),
            _ => Err(anyhow!(
                "Vertex properties have wrong PLY data type (expected float)"
            )),
        }?;

        vertices.push(vertex);
        normals.push(normal);
    }

    let faces = ply
        .payload
        .get("face")
        .ok_or(anyhow!("PLY file is missing a 'face' element"))?;

    let triangles = faces
        .into_iter()
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
        .collect::<Result<Vec<_>, _>>()?;

    let normals = MeshAttribute::new("normals", AttributeData::Vector3Real(normals));
    Ok(MeshWithData::new(TriMesh3d {
        vertices,
        triangles,
    })
    .with_point_data(normals))
}

#[cfg(test)]
pub mod test {
    use super::*;

    #[test]
    fn test_read_ply_cube() -> Result<(), anyhow::Error> {
        let input_file = Path::new("../data/cube.ply");

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
