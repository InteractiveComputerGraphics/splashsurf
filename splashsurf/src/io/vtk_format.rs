use anyhow::{anyhow, Context};
use splashsurf_lib::mesh::{MeshWithData, TriMesh3d};
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::vtkio;
use splashsurf_lib::vtkio::model::{
    Attributes, CellType, Cells, UnstructuredGridPiece, VertexNumbers,
};
use splashsurf_lib::{IteratorExt, Real};
use std::fs::create_dir_all;
use std::path::Path;
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

/// Tries to read a surface mesh from the VTK file at the given path
pub fn surface_mesh_from_vtk<R: Real, P: AsRef<Path>>(
    vtk_file: P,
) -> Result<MeshWithData<R, TriMesh3d<R>>, anyhow::Error> {
    let mesh_dataset = read_vtk(vtk_file)?;
    surface_mesh_from_dataset(mesh_dataset)
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
        return Err(anyhow!(
            "Particle point buffer length is not divisible by 3"
        ));
    }

    let num_points = coords.len() / 3;
    let positions = coords
        .chunks_exact(3)
        .map(|triplet| {
            Some(Vector3::new(
                triplet[0].try_convert()?,
                triplet[1].try_convert()?,
                triplet[2].try_convert()?,
            ))
        })
        .map(|vec| {
            vec.ok_or_else(|| {
                anyhow!("Failed to convert coordinate from input to output float type, value out of range?")
            })
        })
        .try_collect_with_capacity(num_points)?;

    Ok(positions)
}

/// Tries to convert a VTK `DataSet` into a vector of particle positions
pub fn particles_from_dataset<R: Real>(dataset: DataSet) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let unstructured_grid_pieces = match dataset {
        DataSet::UnstructuredGrid { pieces, .. } => Ok(pieces),
        _ => Err(anyhow!("Loaded dataset is not an unstructured grid")),
    }?;

    let first_piece = unstructured_grid_pieces
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("Loaded dataset does not contain any unstructured grid pieces"))?;

    let points = first_piece
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
}

/// Tries to convert a VTK `DataSet` into a surface mesh
pub fn surface_mesh_from_dataset<R: Real>(
    dataset: DataSet,
) -> Result<MeshWithData<R, TriMesh3d<R>>, anyhow::Error> {
    let unstructured_grid_pieces = match dataset {
        DataSet::UnstructuredGrid { pieces, .. } => Ok(pieces),
        _ => Err(anyhow!("Loaded dataset is not an unstructured grid")),
    }?;

    let first_piece = unstructured_grid_pieces
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("Loaded dataset does not contain any unstructured grid pieces"))?
        .into_loaded_piece_data(None)
        .context("Failed to load unstructured grid piece")?;

    let vertices = match first_piece.points {
        IOBuffer::F64(coords) => particles_from_coords(&coords),
        IOBuffer::F32(coords) => particles_from_coords(&coords),
        _ => Err(anyhow!(
            "Point coordinate IOBuffer does not contain f32 or f64 values"
        )),
    }?;

    let triangles = {
        let (num_cells, cell_verts) = first_piece.cells.cell_verts.into_legacy();

        if cell_verts.len() % 4 != 0 {
            return Err(anyhow!("Length of cell vertex array is invalid. Expected 4 values per cell (3 for each triangle vertex index + 1 for vertex count). There are {} values for {} cells.", cell_verts.len(), num_cells));
        }

        let cells = cell_verts
            .chunks_exact(4)
            .enumerate()
            .map(|(cell_idx, cell)| {
                let is_triangle = cell[0] == 0;
                is_triangle
                    .then(|| [cell[1] as usize, cell[2] as usize, cell[3] as usize])
                    .ok_or_else(|| anyhow!("Expected only triangle cells. Invalid number of vertex indices ({}) of cell {}", cell[0], cell_idx))
            })
            .try_collect_with_capacity(num_cells as usize)?;
        cells
    };

    Ok(MeshWithData::new(TriMesh3d {
        vertices,
        triangles,
    }))
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

        // Each particle has a cell of type `Vertex`
        let cell_types = vec![CellType::Vertex; particles.len()];

        let vertices = {
            let mut vertices = Vec::with_capacity(particles.len() * (1 + 1));
            for i in 0..particles.len() {
                // Number of vertices of the cell
                vertices.push(1);
                // Vertex index
                vertices.push(i as u32);
            }
            vertices
        };

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
