use splashsurf_lib::vtkio::model::{
    Attributes, CellType, Cells, UnstructuredGridPiece, VertexNumbers,
};
use std::fs::create_dir_all;
use std::path::Path;

use anyhow::{anyhow, Context};

use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::vtkio;
use splashsurf_lib::Real;

use vtkio::model::{ByteOrder, DataSet, Version, Vtk};
use vtkio::IOBuffer;

/// Tries to read a set of particles from the VTK file at the given path
pub fn particles_from_vtk<R: Real, P: AsRef<Path>>(
    vtk_file: P,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let particle_dataset = read_vtk(vtk_file)?;
    particles_from_dataset(particle_dataset)
}

/// Tries to write a set of particles to the VTK file at the given path
pub fn particles_to_vtk<R: Real, P: AsRef<Path>>(
    particles: &[Vector3<R>],
    vtk_file: P,
) -> Result<(), anyhow::Error> {
    write_vtk(
        UnstructuredGridPiece::from(Particles(particles)),
        vtk_file,
        "particles",
    )
}

/// Tries to write `data` that is convertible to a VTK `DataSet` into a big endian VTK file
pub fn write_vtk<P: AsRef<Path>>(
    data: impl Into<DataSet>,
    filename: P,
    title: &str,
) -> Result<(), anyhow::Error> {
    let vtk_file = Vtk {
        version: Version::new((4, 1)),
        title: title.to_string(),
        file_path: None,
        byte_order: ByteOrder::BigEndian,
        data: data.into(),
    };

    let filename = filename.as_ref();
    if let Some(dir) = filename.parent() {
        create_dir_all(dir).context("Failed to create parent directory of output file")?;
    }
    vtk_file
        .export_be(filename)
        .context("Error while writing VTK output to file")
}

/// Tries to read the given file into a VTK `DataSet`
pub fn read_vtk<P: AsRef<Path>>(filename: P) -> Result<DataSet, vtkio::Error> {
    let filename = filename.as_ref();
    Vtk::import_legacy_be(filename).map(|vtk| vtk.data)
}

/// Tries to convert a vector of consecutive coordinate triplets into a vector of `Vector3`, also converts between floating point types
pub fn particles_from_coords<RealOut: Real, RealIn: Real>(
    coords: &Vec<RealIn>,
) -> Result<Vec<Vector3<RealOut>>, anyhow::Error> {
    if coords.len() % 3 != 0 {
        anyhow!("The number of values in the particle data point buffer is not divisible by 3");
    }

    let num_points = coords.len() / 3;
    let mut positions = Vec::with_capacity(num_points);
    for i in 0..num_points {
        positions.push(Vector3::new(
            RealOut::from_f64(coords[3 * i + 0].to_f64().unwrap()).unwrap(),
            RealOut::from_f64(coords[3 * i + 1].to_f64().unwrap()).unwrap(),
            RealOut::from_f64(coords[3 * i + 2].to_f64().unwrap()).unwrap(),
        ))
    }

    Ok(positions)
}

/// Tries to convert a VTK `DataSet` into a vector of particle positions
pub fn particles_from_dataset<R: Real>(dataset: DataSet) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    if let DataSet::UnstructuredGrid { pieces, .. } = dataset {
        if let Some(piece) = pieces.into_iter().next() {
            let points = piece
                .into_loaded_piece_data(None)
                .context("Failed to load unstructured grid piece")?
                .points;

            match points {
                IOBuffer::F64(coords) => particles_from_coords(&coords),
                IOBuffer::F32(coords) => particles_from_coords(&coords),
                _ => Err(anyhow!(
                    "Point coordinate IOBuffer does not contain f32 or f64 values"
                )),
            }
        } else {
            Err(anyhow!(
                "Loaded dataset does not contain an unstructured grid piece"
            ))
        }
    } else {
        Err(anyhow!(
            "Loaded dataset does not contain an unstructured grid"
        ))
    }
}

/// Wrapper for a slice of particle positions for converting it into a VTK `UnstructuredGridPiece`
struct Particles<'a, R: Real>(&'a [Vector3<R>]);

impl<'a, R> From<Particles<'a, R>> for UnstructuredGridPiece
where
    R: Real,
{
    fn from(particles: Particles<'a, R>) -> Self {
        let particles = particles.0;

        let points = {
            let mut points: Vec<R> = Vec::with_capacity(particles.len() * 3);
            for p in particles.iter() {
                points.extend(p.as_slice());
            }
            points
        };

        let vertices = {
            let mut vertices = Vec::with_capacity(particles.len() * (1 + 1));
            for i in 0..particles.len() {
                vertices.push(1);
                vertices.push(i as u32);
            }
            vertices
        };

        let cell_types = vec![CellType::Vertex; particles.len()];

        UnstructuredGridPiece {
            points: points.into(),
            cells: Cells {
                cell_verts: VertexNumbers::Legacy {
                    num_cells: cell_types.len() as u32,
                    vertices,
                },
                types: cell_types,
            },
            data: Attributes::new(),
        }
    }
}
