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

/// Tries to read a surface mesh from the VTK file at the given path
pub fn surface_mesh_from_ply<R: Real, P: AsRef<Path>>(
    ply_file: P,
) -> Result<MeshWithData<R, TriMesh3d<R>>, anyhow::Error> {
    let mut ply_file = std::fs::File::open(ply_file).unwrap();
    let parser = ply::parser::Parser::<ply::ply::DefaultElement>::new();

    let ply = parser
        .read_ply(&mut ply_file)
        .context("Failed to read PLY file")?;
    let vertices_normals = ply
        .payload
        .get("vertex")
        .ok_or(anyhow!("PLY file is missing a 'vertex' element"))?;

    let vertices_normals: Vec<(Vector3<_>, Vector3<_>)> = vertices_normals
        .into_iter()
        .map(|e| {
            let vertex = (
                e.get("x").unwrap(),
                e.get("y").unwrap(),
                e.get("z").unwrap(),
                e.get("nx").unwrap(),
                e.get("ny").unwrap(),
                e.get("nz").unwrap(),
            );

            let v = match vertex {
                (
                    Property::Float(x),
                    Property::Float(y),
                    Property::Float(z),
                    Property::Float(nx),
                    Property::Float(ny),
                    Property::Float(nz),
                ) => (
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
                ),
                _ => {
                    return Err(anyhow!(
                        "Vertex properties have wrong PLY data type (expected float)"
                    ))
                }
            };

            Ok(v)
        })
        .map(|vn| vn.unwrap())
        .collect();

    let vertices: Vec<Vector3<_>> = vertices_normals.iter().map(|vn| vn.0.clone()).collect();
    let normals: Vec<Vector3<_>> = vertices_normals.iter().map(|vn| vn.1.clone()).collect();

    let faces = ply
        .payload
        .get("face")
        .ok_or(anyhow!("PLY file is missing a 'face' element"))?;

    let triangles = faces
        .into_iter()
        .map(|e| {
            // This is as per what blender creates for a
            let indices = e.get("vertex_indices");
            if let Some(indices) = indices {
                if let Property::ListUInt(indices) = indices {
                    if indices.len() == 3 {
                        return Ok([
                            indices[0] as usize,
                            indices[1] as usize,
                            indices[2] as usize,
                        ]);
                    } else {
                        return Err(anyhow!(
                            "Invalid number of vertex indices per cell: {}",
                            indices.len()
                        ));
                    }
                } else {
                    return Err(anyhow!(
                        "Index properties have wrong PLY data type (expected uint)"
                    ));
                }
            } else {
                return Err(anyhow!(
                    "Vertex properties have wrong PLY data type (expected uint)"
                ));
            }
        })
        .map(|e| e.unwrap())
        .collect();

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
    fn test_convert_cube() -> Result<(), anyhow::Error> {
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
