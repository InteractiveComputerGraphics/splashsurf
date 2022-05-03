mod convert;
mod io;
mod reconstruction;
#[macro_use]
mod allocator;

use crate::allocator::GetPeakAllocatedMemory;
use anyhow::{anyhow, Context};
use log::{error, info};
use std::env;
use structopt::StructOpt;

// Register allocator to track memory usage, might decrease performance if enabled
register_counting_allocator!(GLOBAL_ALLOCATOR, enable = false);

// TODO: Use different logging approach when processing multiple files in parallel
// TODO: Add start and end index for input file sequences
// TODO: Does coarse_prof work with multiple threads?
// TODO: Check if all paths supplied using the cmd args are valid
// TODO: Clean up the parameter structs and conversions

#[derive(Clone, Debug, StructOpt)]
#[structopt(
    name = "splashsurf",
    author = "Fabian Löschner <loeschner@cs.rwth-aachen.de>",
    about = "Surface reconstruction for particle data from SPH simulations (https://github.com/w1th0utnam3/splashsurf)"
)]
struct CommandlineArgs {
    /// Enable quiet mode (no output except for severe panic messages), overrides verbosity level
    #[structopt(long, short = "-q")]
    quiet: bool,
    /// Print more verbose output, use multiple "v"s for even more verbose output (-v, -vv)
    #[structopt(short, parse(from_occurrences))]
    verbosity: u64,
    /// Subcommands
    #[structopt(subcommand)]
    subcommand: Subcommand,
}

#[derive(Clone, Debug, StructOpt)]
enum Subcommand {
    /// Reconstruct a surface from particle data
    Reconstruct(reconstruction::ReconstructSubcommandArgs),
    /// Convert particle or mesh files between different file formats
    Convert(convert::ConvertSubcommandArgs),
}

fn main() -> Result<(), anyhow::Error> {
    /*
    // Panic hook for easier debugging
    panic::set_hook(Box::new(|panic_info| {
        println!("Panic occurred: {}", panic_info);
        println!("Add breakpoint here for debugging.");
    }));
    */

    std::process::exit(match run_splashsurf() {
        Ok(_) => 0,
        Err(err) => {
            log_error(&err);
            1
        }
    });
}

fn run_splashsurf() -> Result<(), anyhow::Error> {
    let cmd_args = CommandlineArgs::from_args();

    let verbosity = VerbosityLevel::from(cmd_args.verbosity);
    let is_quiet = cmd_args.quiet;

    initialize_logging(verbosity, is_quiet).context("Failed to initialize logging")?;
    log_program_info();

    // Delegate to subcommands
    match &cmd_args.subcommand {
        Subcommand::Reconstruct(cmd_args) => reconstruction::reconstruct_subcommand(cmd_args)?,
        Subcommand::Convert(cmd_args) => convert::convert_subcommand(cmd_args)?,
    }

    // Write coarse_prof stats using log::info
    info!("Timings:");
    splashsurf_lib::profiling::write_to_string()
        .unwrap()
        .split("\n")
        .filter(|l| l.len() > 0)
        .for_each(|l| info!("{}", l));

    // Print memory stats if available
    if let Some(peak_allocation_bytes) = GLOBAL_ALLOCATOR.get_peak_allocated_memory() {
        info!(
            "Peak memory usage: {} bytes ({:.2}MB)",
            peak_allocation_bytes,
            peak_allocation_bytes as f64 * 1e-6
        );
    }

    Ok(())
}

/// Prints an anyhow error and its full error chain using the log::error macro
pub fn log_error(err: &anyhow::Error) {
    error!("Error occurred: {}", err);
    err.chain()
        .skip(1)
        .for_each(|cause| error!("  caused by: {}", cause));
}

/// Initializes logging with fern
fn initialize_logging(verbosity: VerbosityLevel, quiet_mode: bool) -> Result<(), anyhow::Error> {
    let mut unknown_log_filter_level = None;
    let log_filter_level = if quiet_mode {
        // First option: disable logging in quiet mode
        log::LevelFilter::Off
    } else {
        // Second option: use verbosity level
        verbosity.into_filter().unwrap_or_else(|| {
            // Third option: use log level from env
            if let Some(log_level) = std::env::var_os("RUST_LOG") {
                let log_level = log_level.to_string_lossy().to_ascii_lowercase();
                match log_level.as_str() {
                    "off" => log::LevelFilter::Off,
                    "error" => log::LevelFilter::Error,
                    "warn" => log::LevelFilter::Warn,
                    "info" => log::LevelFilter::Info,
                    "debug" => log::LevelFilter::Debug,
                    "trace" => log::LevelFilter::Trace,
                    _ => {
                        unknown_log_filter_level = Some(log_level);
                        log::LevelFilter::Info
                    }
                }
            } else {
                // Fourth option: use default level
                log::LevelFilter::Info
            }
        })
    };

    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{}][{}][{}] {}",
                time::OffsetDateTime::now_local()
                    .unwrap_or_else(|_| time::OffsetDateTime::now_utc())
                    .format(&time::format_description::well_known::Rfc3339)
                    .unwrap_or_else(|_| String::new()),
                record.target(),
                record.level(),
                message
            ))
        })
        .level(log_filter_level)
        .chain(std::io::stdout())
        .apply()
        .map_err(|e| anyhow!("Unable to apply logger configuration ({:?})", e))?;

    if let Some(filter_level) = unknown_log_filter_level {
        error!(
            "Unknown log filter level '{}' defined in 'RUST_LOG' env variable, using INFO instead.",
            filter_level
        );
    }

    Ok(())
}

/// Prints program name, version etc. and command line arguments to log
fn log_program_info() {
    info!(
        "{} v{} ({})",
        env!("CARGO_BIN_NAME"),
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_NAME")
    );

    let cmd_line: String = {
        let mut cmd_line = String::new();
        for arg in env::args() {
            cmd_line.push_str(&arg);
            cmd_line.push(' ');
        }
        cmd_line.pop();
        cmd_line
    };
    info!("Called with command line: {}", cmd_line);
}

#[derive(Copy, Clone, Debug)]
enum VerbosityLevel {
    None,
    Verbose,
    VeryVerbose,
}

impl From<u64> for VerbosityLevel {
    fn from(value: u64) -> Self {
        match value {
            0 => VerbosityLevel::None,
            1 => VerbosityLevel::Verbose,
            2 => VerbosityLevel::VeryVerbose,
            _ => VerbosityLevel::VeryVerbose,
        }
    }
}

impl VerbosityLevel {
    /// Maps this verbosity level to a log filter
    fn into_filter(self) -> Option<log::LevelFilter> {
        match self {
            VerbosityLevel::None => None,
            VerbosityLevel::Verbose => Some(log::LevelFilter::Debug),
            VerbosityLevel::VeryVerbose => Some(log::LevelFilter::Trace),
        }
    }
}
