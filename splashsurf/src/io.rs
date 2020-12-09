use std::fs::{create_dir_all, File};
use std::io::Read;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{anyhow, Context};
use ply_rs as ply;
use ply_rs::ply::Property;
use splashsurf_lib::nalgebra::Vector3;
use vtkio::model::{DataSet, Version, Vtk};
use vtkio::{export_be, import_be};

use splashsurf_lib::Real;

/// File format parameters for all io
#[derive(Clone, Debug, Default)]
pub struct FormatParameters {
    pub input: InputFormatParameters,
    pub output: OutputFormatParameters,
}

/// File format parameters for input files
#[derive(Clone, Debug)]
pub struct InputFormatParameters {}

impl Default for InputFormatParameters {
    fn default() -> Self {
        Self {}
    }
}

/// File format parameters for output files
#[derive(Clone, Debug)]
pub struct OutputFormatParameters {}

impl Default for OutputFormatParameters {
    fn default() -> Self {
        Self {}
    }
}

pub fn write_vtk<P: AsRef<Path>>(
    data: impl Into<DataSet>,
    filename: P,
    title: &str,
) -> Result<(), anyhow::Error> {
    let vtk_file = Vtk {
        version: Version::new((4, 1)),
        title: title.to_string(),
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

pub fn particles_from_dataset<R: Real>(
    dataset: &DataSet,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    if let DataSet::UnstructuredGrid { points, .. } = dataset {
        if let Some(coords) = points.clone_into_vec::<f64>() {
            particles_from_coords(&coords)
        } else {
            if let Some(coords) = points.clone_into_vec::<f32>() {
                particles_from_coords(&coords)
            } else {
                Err(anyhow!(
                    "Point coordinate IOBuffer does not contain f32 or f64 values"
                ))
            }
        }
    } else {
        Err(anyhow!(
            "Loaded dataset does not contain an unstructured grid"
        ))
    }
}

pub fn particles_from_xyz<R: Real, P: AsRef<Path>>(
    xyz_file: P,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let file = File::open(xyz_file).context("Unable to open xyz file for reading")?;
    let mut reader = BufReader::new(file);

    let mut buffer = [0u8; 3 * 4];

    let get_four_bytes = |buffer: &[u8], offset: usize| -> [u8; 4] {
        [
            buffer[offset + 0],
            buffer[offset + 1],
            buffer[offset + 2],
            buffer[offset + 3],
        ]
    };

    let mut particles = Vec::new();

    while let Ok(_) = reader.read_exact(&mut buffer) {
        let x = f32::from_ne_bytes(get_four_bytes(&buffer, 0));
        let y = f32::from_ne_bytes(get_four_bytes(&buffer, 4));
        let z = f32::from_ne_bytes(get_four_bytes(&buffer, 8));
        particles.push(Vector3::new(
            R::from_f32(x).unwrap(),
            R::from_f32(y).unwrap(),
            R::from_f32(z).unwrap(),
        ));
    }

    Ok(particles)
}

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

#[allow(dead_code)]
pub fn to_binary_f32<R: Real, P: AsRef<Path>>(file: P, values: &[R]) -> Result<(), anyhow::Error> {
    let file = file.as_ref();
    let file = File::create(file).context("Unable to create binary file")?;
    let mut writer = BufWriter::new(file);

    for v in values {
        let v_f32 = v.to_f32().unwrap();
        writer.write(&v_f32.to_ne_bytes())?;
    }

    Ok(())
}
