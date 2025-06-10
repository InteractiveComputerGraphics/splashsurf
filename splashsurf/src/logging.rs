use std::env;
use std::io::Write;

use anyhow::anyhow;
use fern::Output;
use indicatif::{ProgressBar, WeakProgressBar};
use log::{error, info};
use once_cell::sync::Lazy;
use parking_lot::RwLock;

use crate::cli::VerbosityLevel;

#[derive(Debug)]
pub struct ProgressHandler<T: Write + Send>(T);

/// Stores the reference to the current Progressbar, if there even is one
/// uses "parking_lot::RwLock", because it is faster than the standard libraries and to allow many readers (read-only access to the variable)
static CURRENT_PROGRESS_BAR: Lazy<RwLock<Option<WeakProgressBar>>> =
    Lazy::new(|| RwLock::new(None));

impl<T: Write + Send> ProgressHandler<T> {
    /// Helper function to execute everything before running "inner_function"
    fn handle<F: FnOnce(&mut Self) -> R, R>(&mut self, inner_function: F) -> R {
        let handle = get_progress_bar();

        match handle {
            Some(pb) => pb.suspend(|| inner_function(self)),
            None => inner_function(self),
        }
    }

    /// Create a new instance of "Self"
    pub fn new(pipe: T) -> Self {
        Self(pipe)
    }
}

impl<T: Write + Send + 'static> ProgressHandler<T> {
    pub fn into_output(self) -> Output {
        let boxed: Box<dyn Write + Send + 'static> = Box::new(self);
        boxed.into()
    }
}

impl<T: Write + Send> Write for ProgressHandler<T> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.handle(|this| this.0.write(buf))
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        self.handle(|this| this.0.flush())
    }
}

/// Sets the reference to the current ProgressBar for the writer
pub(crate) fn set_progress_bar(pb: Option<WeakProgressBar>) {
    *CURRENT_PROGRESS_BAR.write() = pb;
}

/// Returns the Progressbar, if it exists, otherwise "None"
pub(crate) fn get_progress_bar() -> Option<ProgressBar> {
    return CURRENT_PROGRESS_BAR.read().as_ref()?.upgrade();
}

/// Prints an anyhow error and its full error chain using the log::error macro
pub(crate) fn log_error(err: &anyhow::Error) {
    error!("Error occurred: {}", err);
    err.chain()
        .skip(1)
        .for_each(|cause| error!("  caused by: {}", cause));
}

/// Initializes logging with fern
pub(crate) fn initialize_logging(
    verbosity: VerbosityLevel,
    quiet_mode: bool,
) -> Result<(), anyhow::Error> {
    let mut unknown_log_filter_level = None;
    let log_filter_level = if quiet_mode {
        // First option: disable logging in quiet mode
        log::LevelFilter::Off
    } else {
        // Second option: use verbosity level
        verbosity.into_filter().unwrap_or_else(|| {
            // Third option: use log level from env
            if let Some(log_level) = env::var_os("RUST_LOG") {
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

    if matches!(verbosity, VerbosityLevel::None) {
        build_logger(log_filter_level, |out, message, record| {
            out.finish(format_args!(
                "[{}][{}] {}",
                chrono::Local::now().format("%T%.3f"),
                record.level(),
                message
            ))
        })?;
    } else {
        build_logger(log_filter_level, |out, message, record| {
            out.finish(format_args!(
                "[{}][{}][{}] {}",
                chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, false),
                record.target(),
                record.level(),
                message
            ))
        })?;
    }

    if let Some(filter_level) = unknown_log_filter_level {
        error!(
            "Unknown log filter level '{}' defined in 'RUST_LOG' env variable, using INFO instead.",
            filter_level
        );
    }

    Ok(())
}

fn build_logger<F>(log_filter_level: log::LevelFilter, formatter: F) -> Result<(), anyhow::Error>
where
    F: Fn(fern::FormatCallback, &std::fmt::Arguments, &log::Record) + Sync + Send + 'static,
{
    fern::Dispatch::new()
        .format(formatter)
        .level(log_filter_level)
        .chain(ProgressHandler::new(std::io::stdout()).into_output())
        .apply()
        .map_err(|e| anyhow!("Unable to apply logger configuration ({:?})", e))
}

/// Prints program name, version etc. and command line arguments to log
pub(crate) fn log_program_info() {
    info!(
        "{} v{} ({})",
        //env!("CARGO_BIN_NAME"),
        std::env::args()
            .next()
            .unwrap_or_else(|| "splashsurf".to_string()),
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
