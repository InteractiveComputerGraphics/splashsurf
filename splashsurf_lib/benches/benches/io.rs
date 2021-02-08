pub mod xyz {
    use std::fs::File;
    use std::io::{BufReader, Read};
    use std::path::Path;

    use anyhow::Context;

    use splashsurf_lib::nalgebra::Vector3;
    use splashsurf_lib::Real;

    pub fn particles_from_xyz<R: Real, P: AsRef<Path>>(
        xyz_file: P,
    ) -> Result<Vec<Vector3<R>>, anyhow::Error> {
        let file = File::open(xyz_file).context("Unable to open XYZ file for reading")?;
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
}

pub mod vtk {
    use splashsurf_lib::nalgebra::Vector3;
    use splashsurf_lib::Real;

    use std::fs::create_dir_all;
    use std::path::Path;

    use anyhow::{anyhow, Context};

    use vtkio::model::{ByteOrder, DataSet, Version, Vtk};
    use vtkio::IOBuffer;

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
            file_path: None,
            byte_order: ByteOrder::BigEndian,
            data: data.into(),
        };

        let filename = filename.as_ref();
        if let Some(dir) = filename.parent() {
            create_dir_all(dir).context("Failed to create parent directory of output file")?;
        }
        vtk_file.export_ascii(filename).context("Error while writing VTK output to file")
    }

    pub fn read_vtk<P: AsRef<Path>>(filename: P) -> Result<DataSet, vtkio::Error> {
        let filename = filename.as_ref();
        Vtk::import_legacy_be(filename).map(|vtk| vtk.data)
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
        dataset: DataSet,
    ) -> Result<Vec<Vector3<R>>, anyhow::Error> {
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
}
