use crate::io;
use anyhow::Context;
use anyhow::anyhow;
use clap::value_parser;
use log::info;
use splashsurf_lib::mesh::MeshWithData;
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::{Aabb3d, nalgebra, profile};
use std::path::PathBuf;

// TODO: Support double input/output

/// Command line arguments for the `convert` subcommand
#[derive(Clone, Debug, clap::Parser)]
pub struct ConvertSubcommandArgs {
    /// Path to the input file with particles to read (supported formats: .vtk, .vtu, .bgeo, .ply, .xyz, .json)
    #[arg(
        long = "particles",
        value_parser = value_parser!(PathBuf),
        conflicts_with = "input_mesh"
    )]
    input_particles: Option<PathBuf>,
    /// Path to the input file with a surface to read (supported formats: .vtk, .ply)
    #[arg(
        long = "mesh",
        value_parser = value_parser!(PathBuf),
        conflicts_with = "input_particles"
    )]
    input_mesh: Option<PathBuf>,
    /// Path to the output file (supported formats for particles: .vtk, .bgeo, .json, for meshes: .obj, .vtk)
    #[arg(short = 'o', value_parser = value_parser!(PathBuf))]
    output_file: PathBuf,
    /// Whether to overwrite existing files without asking
    #[arg(long)]
    overwrite: bool,
    /// Lower corner of the domain of particles to keep (requires domain-max to be specified)
    #[arg(
        long,
        number_of_values = 1,
        number_of_values = 3,
        value_names = ["X_MIN", "Y_MIN", "Z_MIN"],
        allow_negative_numbers = true,
        requires = "domain_max"
    )]
    domain_min: Option<Vec<f64>>,
    /// Lower corner of the domain of particles to keep (requires domain-min to be specified)
    #[arg(
        long,
        number_of_values = 3,
        value_names = ["X_MIN", "Y_MIN", "Z_MIN"],
        allow_negative_numbers = true,
        requires = "domain_min"
    )]
    domain_max: Option<Vec<f64>>,
}

/// Executes the `convert` subcommand
pub fn convert_subcommand(cmd_args: &ConvertSubcommandArgs) -> Result<(), anyhow::Error> {
    // Check if file already exists
    overwrite_check(cmd_args)?;

    match (&cmd_args.input_particles, &cmd_args.input_mesh) {
        (Some(_), _) => convert_particles(cmd_args)?,
        (_, Some(_)) => convert_mesh(cmd_args)?,
        (_, _) => {
            return Err(anyhow!(
                "Aborting: No input file specified, either a particle or mesh input file has to be specified."
            ));
        }
    }

    Ok(())
}

fn convert_particles(cmd_args: &ConvertSubcommandArgs) -> Result<(), anyhow::Error> {
    profile!("particle file conversion cli");

    let io_params = io::FormatParameters::default();
    let input_file = cmd_args.input_particles.as_ref().unwrap();
    let output_file = &cmd_args.output_file;

    // Read particles
    let particle_positions: Vec<Vector3<f32>> =
        io::read_particle_positions(input_file.as_path(), &io_params.input).with_context(|| {
            format!(
                "Failed to load particle positions from file \"{}\"",
                input_file.as_path().display()
            )
        })?;

    // Filter particles by user specified domain
    let particle_positions = if let (Some(min), Some(max)) =
        (cmd_args.domain_min.clone(), cmd_args.domain_max.clone())
    {
        let min = nalgebra::convert(Vector3::from_iterator(min));
        let max = nalgebra::convert(Vector3::from_iterator(max));
        let aabb = Aabb3d::new(min, max);
        info!("Filtering out particles outside of {:?}", aabb);

        particle_positions
            .into_iter()
            .filter(|p| aabb.contains_point(p))
            .collect()
    } else {
        particle_positions
    };

    // Write particles
    io::write_particle_positions(
        particle_positions.as_slice(),
        output_file.as_path(),
        &io_params.output,
    )?;

    Ok(())
}

fn convert_mesh(cmd_args: &ConvertSubcommandArgs) -> Result<(), anyhow::Error> {
    profile!("mesh file conversion cli");

    let io_params = io::FormatParameters::default();
    let input_file = cmd_args.input_mesh.as_ref().unwrap();
    let output_file = &cmd_args.output_file;

    // Try to load surface mesh
    let mesh: MeshWithData<f32, _> = io::read_surface_mesh(input_file.as_path(), &io_params.input)
        .with_context(|| {
            format!(
                "Failed to load surface mesh from file \"{}\"",
                input_file.as_path().display()
            )
        })?;

    // Write mesh
    io::write_mesh(&mesh, output_file.as_path(), &io_params.output)?;

    Ok(())
}

/// Returns an error if the file already exists but overwrite is disabled
fn overwrite_check(cmd_args: &ConvertSubcommandArgs) -> Result<(), anyhow::Error> {
    if !cmd_args.overwrite && cmd_args.output_file.exists() {
        return Err(anyhow!(
            "Aborting: Output file \"{}\" already exists. Use overwrite flag to ignore this.",
            cmd_args.output_file.display()
        ));
    }

    Ok(())
}
