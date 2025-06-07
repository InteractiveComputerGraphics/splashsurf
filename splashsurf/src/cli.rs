//! The `splashsurf` surface reconstruction CLI.
//!
//! For documentation of the CLI see the [README](https://github.com/InteractiveComputerGraphics/splashsurf) in the project repository.
//! The reconstruction procedure and other internals of the CLI are provided by the [`splashsurf_lib`] crate.

use crate::allocator::GetPeakAllocatedMemory;
use crate::{convert, logging, reconstruction};
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
    Reconstruct(reconstruction::ReconstructSubcommandArgs),
    /// Convert particle or mesh files between different file formats
    #[command(help_template = HELP_TEMPLATE)]
    Convert(convert::ConvertSubcommandArgs),
}

pub fn run_splashsurf(args: &[std::ffi::OsString]) -> Result<(), anyhow::Error> {
    run_splashsurf_impl(args).inspect_err(logging::log_error)
}

fn run_splashsurf_impl(args: &[std::ffi::OsString]) -> Result<(), anyhow::Error> {
    let cmd_args = CommandlineArgs::parse_from(args);

    let verbosity = VerbosityLevel::from(cmd_args.verbosity);
    let is_quiet = cmd_args.quiet;

    logging::initialize_logging(verbosity, is_quiet).context("Failed to initialize logging")?;
    logging::log_program_info();

    // Delegate to subcommands
    let result = match &cmd_args.subcommand {
        Subcommand::Reconstruct(cmd_args) => reconstruction::reconstruct_subcommand(cmd_args),
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
