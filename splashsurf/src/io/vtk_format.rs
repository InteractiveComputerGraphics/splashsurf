use std::fs::create_dir_all;
use std::path::Path;

use anyhow::{anyhow, Context};
use vtkio::model::{ByteOrder, DataSet, Version, Vtk};
use vtkio::{export_be, import_be, IOBuffer};

use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::Real;

pub fn particles_from_vtk<R: Real, P: AsRef<Path>>(
    vtk_file: P,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let sph_dataset = read_vtk(vtk_file)?;
    particles_from_dataset(sph_dataset)
}

pub fn write_vtk<P: AsRef<Path>>(
    data: impl Into<DataSet>,
    filename: P,
    title: &str,
) -> Result<(), anyhow::Error> {
    let vtk_file = Vtk {
        version: Version::new((4, 1)),
        title: title.to_string(),
        byte_order: ByteOrder::BigEndian,
        data: data.into(),
    };

    let filename = filename.as_ref();
    if let Some(dir) = filename.parent() {
        create_dir_all(dir).context("Failed to create parent directory of output file")?;
    }
    export_be(vtk_file, filename).context("Error while writing VTK output to file")
}

pub fn read_vtk<P: AsRef<Path>>(filename: P) -> Result<DataSet, vtkio::Error> {
    let filename = filename.as_ref();
    import_be(filename).map(|vtk| vtk.data)
}

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

pub fn particles_from_dataset<R: Real>(dataset: DataSet) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    if let DataSet::UnstructuredGrid { pieces, .. } = dataset {
        if let Some(piece) = pieces.into_iter().next() {
            let points = piece
                .load_piece_data()
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
