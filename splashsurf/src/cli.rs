//! The `splashsurf` surface reconstruction CLI.
//!
//! For documentation of the CLI see the [README](https://github.com/InteractiveComputerGraphics/splashsurf) in the project repository.
//! The reconstruction procedure and other internals of the CLI are provided by the [`splashsurf_lib`] crate.

use crate::allocator::GetPeakAllocatedMemory;
use crate::{convert, logging, reconstruct};
use anyhow::Context;
use clap::Parser;
use log::info;

// Register allocator to track memory usage, might decrease performance if enabled
crate::register_counting_allocator!(GLOBAL_ALLOCATOR, enable = false);

// TODO: Use different logging approach when processing multiple files in parallel
// TODO: Does coarse_prof work with multiple threads?
// TODO: Check if all paths supplied using the cmd args are valid
// TODO: Clean up the parameter structs and conversions

static HELP_TEMPLATE: &str = "{before-help}{name} (v{version}) - {author-with-newline}{about-with-newline}\n{usage-heading} {usage}\n\n{all-args}{after-help}";

#[derive(Clone, Debug, clap::Parser)]
#[command(
    name = "splashsurf",
    author = "Fabian Löschner <loeschner@cs.rwth-aachen.de>",
    about = "Surface reconstruction for particle data from SPH simulations (https://github.com/InteractiveComputerGraphics/splashsurf)",
    version,
    propagate_version = true,
    help_template = HELP_TEMPLATE,
)]
struct CommandlineArgs {
    /// Enable quiet mode (no output except for severe panic messages), overrides verbosity level
    #[arg(long, short = 'q', global = true)]
    quiet: bool,
    /// Print more verbose output, use multiple "v"s for even more verbose output (-v, -vv)
    #[arg(short, action = clap::ArgAction::Count, global = true)]
    verbosity: u8,
    /// Subcommands
    #[command(subcommand)]
    subcommand: Subcommand,
}

#[derive(Clone, Debug, clap::Parser)]
enum Subcommand {
    /// Reconstruct a surface from particle data
    #[command(help_template = HELP_TEMPLATE)]
    Reconstruct(reconstruct::ReconstructSubcommandArgs),
    /// Convert particle or mesh files between different file formats
    #[command(help_template = HELP_TEMPLATE)]
    Convert(convert::ConvertSubcommandArgs),
}

/// A simple on/off switch for command line arguments.
///
/// For example an argument defined as:
/// ```rust ignore
/// /// Enable the use of double precision for all computations
/// #[arg(
///     long,
///     default_value = "off",
///     value_name = "off|on",
///     ignore_case = true,
///     require_equals = true
/// )]
/// pub double_precision: Switch,
/// ```
/// can be used in the CLI as `--double-precision=on` or `--double-precision=off`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Switch {
    Off,
    On,
}

impl Switch {
    pub(crate) fn into_bool(self) -> bool {
        match self {
            Switch::Off => false,
            Switch::On => true,
        }
    }
}

/// Runs the splashsurf CLI with the provided command line arguments.
///
/// This function behaves like the binary `splashsurf` command line tool including output to stdout
/// and stderr. It will also exit the process depending on the command line arguments, so it should
/// not be used in typical library contexts.
/// Note that the first argument is always ignored - this is typically the binary name when called using
/// `std::env::args()` from the terminal:
/// ```
/// splashsurf::cli::run_splashsurf(["splashsurf", "--version"]);
/// ```
/// If no placeholder for the binary name is provided it will return an error (and print a help message):
/// ```should_panic
/// splashsurf::cli::run_splashsurf(["--version"]);
/// ```
pub fn run_splashsurf<I, T>(args: I) -> Result<(), anyhow::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    run_splashsurf_impl(args).inspect_err(logging::log_error)
}

fn run_splashsurf_impl<I, T>(args: I) -> Result<(), anyhow::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cmd_args = CommandlineArgs::parse_from(args);

    let verbosity = VerbosityLevel::from(cmd_args.verbosity);
    let is_quiet = cmd_args.quiet;

    logging::initialize_logging(verbosity, is_quiet).context("Failed to initialize logging")?;
    logging::log_program_info();

    // Delegate to subcommands
    let result = match &cmd_args.subcommand {
        Subcommand::Reconstruct(cmd_args) => reconstruct::reconstruct_subcommand(cmd_args),
        Subcommand::Convert(cmd_args) => convert::convert_subcommand(cmd_args),
    };

    // Write coarse_prof stats using log::info
    info!("Timings:");
    splashsurf_lib::profiling::write_to_string()
        .unwrap()
        .split("\n")
        .filter(|l| !l.is_empty())
        .for_each(|l| info!("{}", l));

    // Print memory stats if available
    if let Some(peak_allocation_bytes) = GLOBAL_ALLOCATOR.get_peak_allocated_memory() {
        info!(
            "Peak memory usage: {} bytes ({:.2}MB)",
            peak_allocation_bytes,
            peak_allocation_bytes as f64 * 1e-6
        );
    }

    info!(
        "Finished at {}.",
        chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, false)
    );

    result
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum VerbosityLevel {
    None,
    Verbose,
    VeryVerbose,
    VeryVeryVerbose,
}

impl From<u8> for VerbosityLevel {
    fn from(value: u8) -> Self {
        match value {
            0 => VerbosityLevel::None,
            1 => VerbosityLevel::Verbose,
            2 => VerbosityLevel::VeryVerbose,
            3 => VerbosityLevel::VeryVeryVerbose,
            _ => VerbosityLevel::VeryVeryVerbose,
        }
    }
}

impl VerbosityLevel {
    /// Maps this verbosity level to a log filter
    pub fn into_filter(self) -> Option<log::LevelFilter> {
        match self {
            VerbosityLevel::None => None,
            VerbosityLevel::Verbose => Some(log::LevelFilter::Info),
            VerbosityLevel::VeryVerbose => Some(log::LevelFilter::Debug),
            VerbosityLevel::VeryVeryVerbose => Some(log::LevelFilter::Trace),
        }
    }
}

#[cfg(test)]
mod cli_args_tests {
    use super::*;

    #[test]
    fn verify_main_cli() {
        use clap::CommandFactory;
        CommandlineArgs::command().debug_assert()
    }

    #[test]
    fn verify_reconstruct_cli() {
        use clap::CommandFactory;
        crate::reconstruct::ReconstructSubcommandArgs::command().debug_assert()
    }

    #[test]
    fn verify_convert_cli() {
        use clap::CommandFactory;
        crate::convert::ConvertSubcommandArgs::command().debug_assert()
    }

    #[test]
    fn test_main_cli() {
        use clap::Parser;

        // Display help
        assert_eq!(
            CommandlineArgs::try_parse_from(["splashsurf", "--help",])
                .expect_err("this command is supposed to fail")
                .kind(),
            clap::error::ErrorKind::DisplayHelp
        );

        // Display help, reconstruct
        assert_eq!(
            CommandlineArgs::try_parse_from(["splashsurf", "reconstruct", "--help",])
                .expect_err("this command is supposed to fail")
                .kind(),
            clap::error::ErrorKind::DisplayHelp
        );

        // Display help, convert
        assert_eq!(
            CommandlineArgs::try_parse_from(["splashsurf", "convert", "--help",])
                .expect_err("this command is supposed to fail")
                .kind(),
            clap::error::ErrorKind::DisplayHelp
        );

        // Minimum arguments: input file
        if let Subcommand::Reconstruct(rec_args) = CommandlineArgs::try_parse_from([
            "splashsurf",
            "reconstruct",
            "test.vtk",
            "--particle-radius=0.05",
            "--smoothing-length=3.0",
            "--cube-size=0.75",
        ])
        .expect("this command is supposed to work")
        .subcommand
        {
            assert_eq!(
                rec_args.input_file_or_sequence,
                std::path::PathBuf::from("test.vtk")
            );
        };

        // Test on/off switch
        if let Subcommand::Reconstruct(rec_args) = CommandlineArgs::try_parse_from([
            "splashsurf",
            "reconstruct",
            "test.vtk",
            "--particle-radius=0.05",
            "--smoothing-length=3.0",
            "--cube-size=0.75",
            "--normals=on",
        ])
        .expect("this command is supposed to work")
        .subcommand
        {
            assert_eq!(rec_args.normals, Switch::On);
        };

        if let Subcommand::Reconstruct(rec_args) = CommandlineArgs::try_parse_from([
            "splashsurf",
            "reconstruct",
            "test.vtk",
            "--particle-radius=0.05",
            "--smoothing-length=3.0",
            "--cube-size=0.75",
            "--normals=off",
        ])
        .expect("this command is supposed to work")
        .subcommand
        {
            assert_eq!(rec_args.normals, Switch::Off);
        };

        // Test domain min/max: correct values
        if let Subcommand::Reconstruct(rec_args) = CommandlineArgs::try_parse_from([
            "splashsurf",
            "reconstruct",
            "test.vtk",
            "--particle-radius=0.05",
            "--smoothing-length=3.0",
            "--cube-size=0.75",
            "--particle-aabb-min",
            "-1.0",
            "1.0",
            "-1.0",
            "--particle-aabb-max",
            "-2.0",
            "2.0",
            "-2.0",
        ])
        .expect("this command is supposed to work")
        .subcommand
        {
            assert_eq!(rec_args.particle_aabb_min, Some(vec![-1.0, 1.0, -1.0]));
            assert_eq!(rec_args.particle_aabb_max, Some(vec![-2.0, 2.0, -2.0]));
        };

        // Test domain min/max: too many values
        assert_eq!(
            CommandlineArgs::try_parse_from([
                "splashsurf",
                "reconstruct",
                "test.vtk",
                "--particle-radius=0.05",
                "--smoothing-length=3.0",
                "--cube-size=0.75",
                "--particle-aabb-min",
                "-1.0",
                "1.0",
                "-1.0",
                "2.0",
                "--particle-aabb-max",
                "-2.0",
                "2.0",
                "-2.0",
            ])
            .expect_err("this command is supposed to fail")
            .kind(),
            clap::error::ErrorKind::UnknownArgument
        );
    }
}
