use crate::io::vtk_format::VtkFile;
use anyhow::{Context, anyhow};
use log::{info, warn};
use splashsurf_lib::Real;
use splashsurf_lib::mesh::{
    IntoVtkUnstructuredGridPiece, Mesh3d, MeshAttribute, MeshWithData, TriMesh3d,
};
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::{io, profile};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub(crate) use splashsurf_lib::io::*;

/// File format parameters for all io
#[derive(Clone, Debug, Default)]
pub struct FormatParameters {
    pub input: InputFormatParameters,
    pub output: OutputFormatParameters,
}

/// File format parameters for input files
#[derive(Clone, Debug, Default)]
pub struct InputFormatParameters {}

/// File format parameters for output files
#[derive(Clone, Debug)]
pub struct OutputFormatParameters {
    /// Enable compression for formats that support it
    enable_compression: bool,
}

impl Default for OutputFormatParameters {
    fn default() -> Self {
        Self {
            enable_compression: true,
        }
    }
}

/// Loads particles positions from the given file path, automatically detects the file format
pub fn read_particle_positions<R: Real, P: AsRef<Path>>(
    input_file: P,
    _format_params: &InputFormatParameters,
) -> Result<Vec<Vector3<R>>, anyhow::Error> {
    let input_file = input_file.as_ref();
    info!(
        "Reading particle dataset from \"{}\"...",
        input_file.display()
    );

    let particle_positions = {
        profile!("loading particle positions");
        io::particles_from_file(input_file)?
    };

    info!(
        "Successfully read dataset with {} particle positions.",
        particle_positions.len()
    );

    Ok(particle_positions)
}

/// Tries to read particle positions as well as attributes with the given names from the specified file
pub fn read_particle_positions_with_attributes<R: Real, P: AsRef<Path>>(
    input_file: P,
    attribute_names: &[String],
    format_params: &InputFormatParameters,
) -> Result<(Vec<Vector3<R>>, Vec<MeshAttribute<R>>), anyhow::Error> {
    if attribute_names.is_empty() {
        return read_particle_positions(input_file, format_params).map(|p| (p, Vec::new()));
    }

    let input_file = input_file.as_ref();
    info!(
        "Reading particle dataset and attributes from \"{}\"...",
        input_file.display()
    );

    profile!("loading particle positions and attributes");

    // Check file extension: only VTK and BGEO is supported for reading attributes at the moment
    let extension = input_file.extension().ok_or(anyhow!(
        "Unable to detect file format of particle input file (file name has to end with supported extension)",
    ))?.to_str().ok_or(anyhow!("Invalid extension of input file"))?.to_lowercase();

    let attributes_to_interpolate = attribute_names.iter().cloned().collect::<HashSet<_>>();

    let (particle_positions, attributes) = match extension.as_str() {
        "vtk" | "vtu" => {
            let vtk_pieces = VtkFile::load_file(input_file)
                .map(|f| f.into_pieces())
                .with_context(|| "Error while loading VTK file".to_string())?;

            if vtk_pieces.len() > 1 {
                warn!(
                    "VTK file contains more than one \"piece\". Only the first one will be loaded."
                );
            }

            let first_piece = vtk_pieces
                .into_iter()
                .next()
                .ok_or(anyhow!("VTK file does not contain a supported \"piece\"."))?;

            // Load particles
            let particle_positions = first_piece.load_as_particles()?;

            // Load attributes that should be interpolated
            let attributes = {
                // Check if all attributes to interpolate are present in the input file
                {
                    let available_attributes = first_piece
                        .point_attribute_names()
                        .into_iter()
                        .collect::<HashSet<_>>();

                    let missing_attributes = attributes_to_interpolate
                        .difference(&available_attributes)
                        .cloned()
                        .collect::<Vec<_>>();
                    if !missing_attributes.is_empty() {
                        return Err(anyhow!(
                            "Missing attribute(s) \"{}\" in input file",
                            missing_attributes.join("\", \""),
                        ));
                    }
                }

                first_piece.load_point_attributes::<R>(attribute_names)
            }?;

            (particle_positions, attributes)
        }
        "bgeo" => {
            let bgeo_file = bgeo_format::load_bgeo_file(input_file)
                .with_context(|| "Error while loading BGEO file".to_string())?;

            let particle_positions = bgeo_format::particles_from_bgeo_file::<R>(&bgeo_file)
                .with_context(|| {
                    "Error while loading particle positions from BGEO file".to_string()
                })?;

            // Load attributes that should be interpolated
            let attributes = {
                // Check if all attributes to interpolate are present in the input file
                {
                    let available_attributes = bgeo_file
                        .attribute_definitions
                        .iter()
                        .map(|a| a.name.clone())
                        .collect::<HashSet<_>>();

                    let missing_attributes = attributes_to_interpolate
                        .difference(&available_attributes)
                        .cloned()
                        .collect::<Vec<_>>();
                    if !missing_attributes.is_empty() {
                        return Err(anyhow!(
                            "Missing attribute(s) \"{}\" in input file",
                            missing_attributes.join("\", \""),
                        ));
                    }
                }

                bgeo_format::attributes_from_bgeo_file(
                    &bgeo_file,
                    &Vec::from_iter(attributes_to_interpolate),
                )?
            };

            (particle_positions, attributes)
        }
        _ => {
            return Err(anyhow!(
                "Unsupported file format extension \"{}\" for reading particles and attributes",
                extension
            ));
        }
    };

    info!(
        "Successfully loaded point {} attribute(s): \"{}\"",
        attributes.len(),
        attribute_names.join("\", \"")
    );

    Ok((particle_positions, attributes))
}

/// Writes particles positions to the given file path, automatically detects the file format
pub fn write_particle_positions<R: Real, P: AsRef<Path>>(
    particles: &[Vector3<R>],
    output_file: P,
    format_params: &OutputFormatParameters,
) -> Result<(), anyhow::Error> {
    let output_file = output_file.as_ref();
    info!(
        "Writing {} particles to \"{}\"...",
        particles.len(),
        output_file.display()
    );

    if let Some(extension) = output_file.extension() {
        profile!("writing particle positions");

        let extension = extension
            .to_str()
            .ok_or(anyhow!("Invalid extension of output file"))?;

        match extension.to_lowercase().as_str() {
            "vtk" => vtk_format::particles_to_vtk(particles, output_file),
            "bgeo" => bgeo_format::particles_to_bgeo(
                particles,
                output_file,
                format_params.enable_compression,
            ),
            "json" => json_format::particles_to_json(particles, output_file),
            _ => Err(anyhow!(
                "Unsupported file format extension \"{}\" for writing particles",
                extension
            )),
        }?;
    } else {
        return Err(anyhow!(
            "Unable to detect file format of particle output file (file name has to end with supported extension)",
        ));
    };

    info!("Successfully wrote particles to file.");
    Ok(())
}

/// Loads a surface mesh from the given file path, automatically detects the file format
pub fn read_surface_mesh<R: Real, P: AsRef<Path>>(
    input_file: P,
    _format_params: &InputFormatParameters,
) -> Result<MeshWithData<R, TriMesh3d<R>>, anyhow::Error> {
    let input_file = input_file.as_ref();
    info!("Reading mesh from \"{}\"...", input_file.display());

    let mesh = if let Some(extension) = input_file.extension() {
        profile!("loading surface mesh");

        let extension = extension
            .to_str()
            .ok_or(anyhow!("Invalid extension of input file"))?;

        match extension.to_lowercase().as_str() {
            "vtk" => vtk_format::surface_mesh_from_vtk(input_file),
            "ply" => ply_format::surface_mesh_from_ply(input_file),
            _ => Err(anyhow!(
                "Unsupported file format extension \"{}\" for reading surface meshes",
                extension
            )),
        }
    } else {
        Err(anyhow!(
            "Unable to detect file format of mesh input file (file name has to end with supported extension)",
        ))
    }?;

    info!(
        "Successfully read mesh with {} vertices and {} cells.",
        mesh.mesh.vertices().len(),
        mesh.mesh.cells().len()
    );

    Ok(mesh)
}

/// Writes a mesh and its attribute data to the given file path, automatically detects the file format
pub fn write_mesh<'a, R: Real, MeshT: Mesh3d<R>, P: AsRef<Path>>(
    mesh: &'a MeshWithData<R, MeshT>,
    output_file: P,
    _format_params: &OutputFormatParameters,
) -> Result<(), anyhow::Error>
where
    for<'b> &'b MeshWithData<R, MeshT>: IntoVtkUnstructuredGridPiece,
{
    let output_file = output_file.as_ref();
    info!(
        "Writing mesh with {} vertices and {} cells to \"{}\"...",
        mesh.mesh.vertices().len(),
        mesh.mesh.cells().len(),
        output_file.display()
    );

    if let Some(extension) = output_file.extension() {
        let extension = extension
            .to_str()
            .ok_or(anyhow!("Invalid extension of output file"))?;

        match extension.to_lowercase().as_str() {
            "vtk" => vtk_format::write_vtk(mesh, output_file, "mesh"),
            "ply" => ply_format::mesh_to_ply(mesh, output_file),
            "obj" => obj_format::mesh_to_obj(mesh, output_file),
            _ => Err(anyhow!(
                "Unsupported file format extension \"{}\"",
                extension,
            )),
        }?;
    } else {
        return Err(anyhow!(
            "Unable to detect file format of mesh output file (file name has to end with supported extension)",
        ));
    };

    info!("Successfully wrote mesh to file.");
    Ok(())
}

#[allow(dead_code)]
pub fn to_binary_f32<R: Real, P: AsRef<Path>>(file: P, values: &[R]) -> Result<(), anyhow::Error> {
    let file = file.as_ref();
    let file = File::create(file).context("Unable to create binary file")?;
    let mut writer = BufWriter::new(file);

    for v in values {
        let v_f32 = v.to_f32().unwrap();
        writer.write_all(&v_f32.to_ne_bytes())?;
    }

    Ok(())
}
