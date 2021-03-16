use crate::io;
use anyhow::anyhow;
use anyhow::Context;
use log::info;
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::{nalgebra, profile, AxisAlignedBoundingBox3d};
use std::path::PathBuf;
use structopt::StructOpt;

// TODO: Support double input/output

/// Command line arguments for the `convert` subcommand
#[derive(Clone, Debug, StructOpt)]
pub struct ConvertSubcommandArgs {
    /// Path to the input file with particles to read
    #[structopt(short = "-i", parse(from_os_str))]
    input_file: PathBuf,
    /// Path to the output file
    #[structopt(short = "-o", parse(from_os_str))]
    output_file: PathBuf,
    /// Whether to overwrite existing files without asking
    #[structopt(long)]
    overwrite: bool,
    /// Whether the input files contain particles (instead of surface meshes)
    #[structopt(long, conflicts_with = "mesh")]
    particles: bool,
    /// Whether the input files contains a surface mesh (instead of particles)
    #[structopt(long, conflicts_with = "particles")]
    mesh: bool,
    /// Lower corner of the domain of particles to keep, format: domain-min=x_min;y_min;z_min (requires domain-max to be specified)
    #[structopt(
        long,
        number_of_values = 3,
        value_delimiter = ";",
        requires = "domain-max"
    )]
    domain_min: Option<Vec<f64>>,
    /// Lower corner of the domain of particles to keep, format:domain-max=x_max;y_max;z_max (requires domain-min to be specified)
    #[structopt(
        long,
        number_of_values = 3,
        value_delimiter = ";",
        requires = "domain-min"
    )]
    domain_max: Option<Vec<f64>>,
}

/// Executes the `convert` subcommand
pub fn convert_subcommand(cmd_args: &ConvertSubcommandArgs) -> Result<(), anyhow::Error> {
    // Check if file already exists
    if !cmd_args.overwrite {
        if cmd_args.output_file.exists() {
            return Err(anyhow!(
                "Aborting: Output file \"{}\" already exists. Use overwrite flag to ignore this.",
                cmd_args.output_file.display()
            ));
        }
    }

    if cmd_args.particles || (!cmd_args.particles && !cmd_args.particles) {
        convert_particles(cmd_args)?;
    } else {
        convert_mesh(cmd_args)?;
    }

    Ok(())
}

fn convert_particles(cmd_args: &ConvertSubcommandArgs) -> Result<(), anyhow::Error> {
    profile!("particle file conversion cli");

    let io_params = io::FormatParameters::default();
    let input_file = &cmd_args.input_file;
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
        let aabb = AxisAlignedBoundingBox3d::new(min, max);
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
    let input_file = &cmd_args.input_file;
    let output_file = &cmd_args.output_file;

    unimplemented!("Mesh file conversion is currently not implemented");
}
