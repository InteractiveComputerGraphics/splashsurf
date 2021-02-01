use crate::io;
use anyhow::anyhow;
use anyhow::Context;
use splashsurf_lib::nalgebra::Vector3;
use splashsurf_lib::profile;
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
}

/// Executes the `convert` subcommand
pub fn convert_subcommand(cmd_args: &ConvertSubcommandArgs) -> Result<(), anyhow::Error> {
    profile!("particle file conversion cli");

    let io_params = io::FormatParameters::default();
    let input_file = &cmd_args.input_file;
    let output_file = &cmd_args.output_file;

    // Check if file already exists
    if !cmd_args.overwrite {
        if output_file.exists() {
            return Err(anyhow!(
                "Output file \"{}\" already exists. Use overwrite flag to ignore this.",
                output_file.display()
            ));
        }
    }

    // Read particles
    let particle_positions: Vec<Vector3<f32>> =
        io::read_particle_positions(input_file.as_path(), &io_params.input).with_context(|| {
            format!(
                "Failed to load particle positions from file \"{}\"",
                input_file.as_path().display()
            )
        })?;

    // Write particles
    io::write_particle_positions(
        particle_positions.as_slice(),
        output_file.as_path(),
        &io_params.output,
    )?;

    Ok(())
}
