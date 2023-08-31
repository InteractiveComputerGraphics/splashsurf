use crate::{io, logging};
use anyhow::{anyhow, Context};
use arguments::{
    ReconstructionRunnerArgs, ReconstructionRunnerPathCollection, ReconstructionRunnerPaths,
};
use clap::value_parser;
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rayon::prelude::*;
use splashsurf_lib::mesh::{AttributeData, Mesh3d, MeshAttribute, MeshWithData, PointCloud3d};
use splashsurf_lib::nalgebra::{Unit, Vector3};
use splashsurf_lib::profile;
use splashsurf_lib::sph_interpolation::SphInterpolator;
use splashsurf_lib::{density_map, Index, Real};
use std::convert::TryFrom;
use std::path::PathBuf;

use arguments::*;

// TODO: Detect smallest index type (i.e. check if ok to use i32 as index)

static ARGS_IO: &str = "Input/output";
static ARGS_BASIC: &str = "Numerical reconstruction parameters";
static ARGS_ADV: &str = "Advanced parameters";
static ARGS_OCTREE: &str = "Domain decomposition (octree or grid) parameters";
static ARGS_DEBUG: &str = "Debug options";
static ARGS_INTERP: &str = "Interpolation";
static ARGS_POSTPROC: &str = "Postprocessing";
static ARGS_OTHER: &str = "Remaining options";

/// Command line arguments for the `reconstruct` subcommand
#[derive(Clone, Debug, clap::Parser)]
#[clap(group = clap::ArgGroup::new("input").required(true))]
#[command(next_help_heading = ARGS_OTHER)]
pub struct ReconstructSubcommandArgs {
    /// Path to the input file where the particle positions are stored (supported formats: VTK 4.2, VTU, binary f32 XYZ, PLY, BGEO), use "{}" in the filename to indicate a placeholder for a sequence.
    #[arg(help_heading = ARGS_IO, group = "input", value_parser = value_parser!(PathBuf))]
    pub input_file_or_sequence: PathBuf,
    /// Filename for writing the reconstructed surface to disk (supported formats: VTK, PLY, OBJ, default: "{original_filename}_surface.vtk")
    #[arg(help_heading = ARGS_IO, short = 'o', long, value_parser = value_parser!(PathBuf))]
    pub output_file: Option<PathBuf>,
    /// Optional base directory for all output files (default: current working directory)
    #[arg(help_heading = ARGS_IO, long, value_parser = value_parser!(PathBuf))]
    pub output_dir: Option<PathBuf>,
    /// Index of the first input file to process when processing a sequence of files (default: lowest index of the sequence)
    #[arg(help_heading = ARGS_IO, short = 's', long)]
    pub start_index: Option<usize>,
    /// Index of the last input file to process when processing a sequence of files (default: highest index of the sequence)
    #[arg(help_heading = ARGS_IO, short = 'e', long)]
    pub end_index: Option<usize>,

    /// The particle radius of the input data
    #[arg(help_heading = ARGS_BASIC, short = 'r', long)]
    pub particle_radius: f64,
    /// The rest density of the fluid
    #[arg(help_heading = ARGS_BASIC, long, default_value = "1000.0")]
    pub rest_density: f64,
    /// The smoothing length radius used for the SPH kernel, the kernel compact support radius will be twice the smoothing length (in multiplies of the particle radius)
    #[arg(help_heading = ARGS_BASIC, short = 'l', long)]
    pub smoothing_length: f64,
    /// The cube edge length used for marching cubes in multiplies of the particle radius, corresponds to the cell size of the implicit background grid
    #[arg(help_heading = ARGS_BASIC, short = 'c', long)]
    pub cube_size: f64,
    /// The iso-surface threshold for the density, i.e. the normalized value of the reconstructed density level that indicates the fluid surface (in multiplies of the rest density)
    #[arg(help_heading = ARGS_BASIC, short = 't', long, default_value = "0.6")]
    pub surface_threshold: f64,

    /// Whether to enable the use of double precision for all computations
    #[arg(
        help_heading = ARGS_ADV,
        short = 'd',
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub double_precision: Switch,
    /// Lower corner of the domain where surface reconstruction should be performed (requires domain-max to be specified)
    #[arg(
        help_heading = ARGS_BASIC,
        long,
        number_of_values = 3,
        value_names = ["X_MIN", "Y_MIN", "Z_MIN"],
        allow_negative_numbers = true,
        requires = "particle_aabb_max",
    )]
    pub particle_aabb_min: Option<Vec<f64>>,
    /// Upper corner of the domain where surface reconstruction should be performed (requires domain-min to be specified)
    #[arg(
        help_heading = ARGS_BASIC,
        long,
        number_of_values = 3,
        value_names = ["X_MIN", "Y_MIN", "Z_MIN"],
        allow_negative_numbers = true,
        requires = "particle_aabb_min",
    )]
    pub particle_aabb_max: Option<Vec<f64>>,

    /// Flag to enable multi-threading to process multiple input files in parallel
    #[arg(
        help_heading = ARGS_ADV,
        long = "mt-files",
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub parallelize_over_files: Switch,
    /// Flag to enable multi-threading for a single input file by processing chunks of particles in parallel
    #[arg(
        help_heading = ARGS_ADV,
        long = "mt-particles",
        default_value = "on",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub parallelize_over_particles: Switch,
    /// Set the number of threads for the worker thread pool
    #[arg(help_heading = ARGS_ADV, long, short = 'n')]
    pub num_threads: Option<usize>,

    /// Whether to enable spatial decomposition using a regular grid-based approach
    #[arg(
        help_heading = ARGS_OCTREE,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub subdomain_grid: Switch,
    /// Each subdomain will be a cube consisting of this number of MC cube cells along each coordinate axis
    #[arg(help_heading = ARGS_OCTREE, long, default_value="64")]
    pub subdomain_cubes: u32,

    /// Whether to enable spatial decomposition using an octree (faster) instead of a global approach
    #[arg(
        help_heading = ARGS_OCTREE,
        long,
        default_value = "on",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub octree_decomposition: Switch,
    /// Whether to enable stitching of the disconnected local meshes resulting from the reconstruction when spatial decomposition is enabled (slower, but without stitching meshes will not be closed)
    #[arg(
        help_heading = ARGS_OCTREE,
        long,
        default_value = "on",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub octree_stitch_subdomains: Switch,
    /// The maximum number of particles for leaf nodes of the octree, default is to compute it based on the number of threads and particles
    #[arg(help_heading = ARGS_OCTREE, long)]
    pub octree_max_particles: Option<usize>,
    /// Safety factor applied to the kernel compact support radius when it's used as a margin to collect ghost particles in the leaf nodes when performing the spatial decomposition
    #[arg(help_heading = ARGS_OCTREE, long)]
    pub octree_ghost_margin_factor: Option<f64>,
    /// Whether to compute particle densities in a global step before domain decomposition (slower)
    #[arg(
        help_heading = ARGS_OCTREE,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub octree_global_density: Switch,
    /// Whether to compute particle densities per subdomain but synchronize densities for ghost-particles (faster, recommended).
    /// Note: if both this and global particle density computation is disabled the ghost particle margin has to be increased to at least 2.0
    /// to compute correct density values for ghost particles.
    #[arg(
        help_heading = ARGS_OCTREE,
        long,
        default_value = "on",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub octree_sync_local_density: Switch,

    /// Whether to compute surface normals at the mesh vertices and write them to the output file
    #[arg(
        help_heading = ARGS_INTERP,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub normals: Switch,
    /// Whether to compute the normals using SPH interpolation (smoother and more true to actual fluid surface, but slower) instead of just using area weighted triangle normals
    #[arg(
        help_heading = ARGS_INTERP,
        long,
        default_value = "on",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub sph_normals: Switch,
    /// List of point attribute field names from the input file that should be interpolated to the reconstructed surface. Currently this is only supported for VTK and VTU input files.
    #[arg(help_heading = ARGS_INTERP, long)]
    pub interpolate_attributes: Vec<String>,

    /// Lower corner of the bounding-box for the surface mesh, mesh outside gets cut away (requires mesh-max to be specified)
    #[arg(
        help_heading = ARGS_POSTPROC,
        long,
        number_of_values = 3,
        value_names = ["X_MIN", "Y_MIN", "Z_MIN"],
        allow_negative_numbers = true,
        requires = "mesh_aabb_max",
    )]
    pub mesh_aabb_min: Option<Vec<f64>>,
    /// Upper corner of the bounding-box for the surface mesh, mesh outside gets cut away (requires mesh-min to be specified)
    #[arg(
        help_heading = ARGS_POSTPROC,
        long,
        number_of_values = 3,
        value_names = ["X_MIN", "Y_MIN", "Z_MIN"],
        allow_negative_numbers = true,
        requires = "mesh_aabb_min",
    )]
    pub mesh_aabb_max: Option<Vec<f64>>,

    /// Optional filename for writing the point cloud representation of the intermediate density map to disk
    #[arg(help_heading = ARGS_DEBUG, long, value_parser = value_parser!(PathBuf))]
    pub output_dm_points: Option<PathBuf>,
    /// Optional filename for writing the grid representation of the intermediate density map to disk
    #[arg(help_heading = ARGS_DEBUG, long, value_parser = value_parser!(PathBuf))]
    pub output_dm_grid: Option<PathBuf>,
    /// Optional filename for writing the octree used to partition the particles to disk
    #[arg(help_heading = ARGS_DEBUG, long, value_parser = value_parser!(PathBuf))]
    pub output_octree: Option<PathBuf>,
    /// Whether to check the final mesh for topological problems such as holes (note that when stitching is disabled this will lead to a lot of reported problems)
    #[arg(
        help_heading = ARGS_DEBUG,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub check_mesh: Switch,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Switch {
    Off,
    On,
}

impl Switch {
    fn into_bool(self) -> bool {
        match self {
            Switch::Off => false,
            Switch::On => true,
        }
    }
}

/// Executes the `reconstruct` subcommand
pub fn reconstruct_subcommand(cmd_args: &ReconstructSubcommandArgs) -> Result<(), anyhow::Error> {
    profile!("reconstruct subcommand");

    let paths = ReconstructionRunnerPathCollection::try_from(cmd_args)
        .context("Failed parsing input file path(s) from command line")?
        .collect();
    let args = ReconstructionRunnerArgs::try_from(cmd_args)
        .context("Failed processing parameters from command line")?;

    let _pb = if paths.len() > 1 {
        let pb = ProgressBar::new(paths.len() as u64);
        pb.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40}] {pos}/{len} ({percent}%) - remaining: [{eta_precise}]")
            .unwrap()
            .progress_chars("=> "));
        logging::set_progress_bar(Some(pb.downgrade()));
        Some(pb)
    } else {
        None
    };

    let result = if cmd_args.parallelize_over_files.into_bool() {
        paths.par_iter().try_for_each(|path| {
            reconstruction_pipeline(path, &args)
                .with_context(|| {
                    format!(
                        "Error while processing input file \"{}\" from a file sequence",
                        path.input_file.display()
                    )
                })
                .map_err(|err| {
                    // Already log the error in case there are multiple errors
                    logging::log_error(&err);
                    err
                })
                .and_then(|_| {
                    logging::get_progress_bar().map(|pb| pb.inc(1));
                    Ok(())
                })
        })
    } else {
        paths.iter().try_for_each(|path| {
            reconstruction_pipeline(path, &args).and_then(|_| {
                logging::get_progress_bar().map(|pb| pb.inc(1));
                Ok(())
            })
        })
    };

    if paths.len() > 1 {
        logging::get_progress_bar().map(|pb| pb.finish());
        logging::set_progress_bar(None);
    }

    if result.is_ok() {
        info!("Successfully finished processing all inputs.");
    }

    result
}

/// Conversion and validation of command line arguments
mod arguments {
    use super::ReconstructSubcommandArgs;
    use crate::io;
    use anyhow::{anyhow, Context};
    use log::info;
    use regex::{escape, Regex};
    use splashsurf_lib::nalgebra::Vector3;
    use splashsurf_lib::{Aabb3d, ParticleDensityComputationStrategy};
    use std::convert::TryFrom;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::str::FromStr;
    use walkdir::WalkDir;

    pub struct ReconstructionRunnerPostprocessingArgs {
        pub check_mesh: bool,
        pub compute_normals: bool,
        pub sph_normals: bool,
        pub interpolate_attributes: Vec<String>,
        pub mesh_aabb: Option<Aabb3d<f64>>,
    }

    /// All arguments that can be supplied to the surface reconstruction tool converted to useful types
    pub struct ReconstructionRunnerArgs {
        pub params: splashsurf_lib::Parameters<f64>,
        pub use_double_precision: bool,
        pub io_params: io::FormatParameters,
        pub postprocessing: ReconstructionRunnerPostprocessingArgs,
    }

    fn try_aabb_from_min_max(
        min: &Vec<f64>,
        max: &Vec<f64>,
        error_str: &'static str,
    ) -> Result<Aabb3d<f64>, anyhow::Error> {
        // This should already be ensured by StructOpt parsing
        assert_eq!(min.len(), 3);
        assert_eq!(max.len(), 3);

        let aabb = Aabb3d::new(
            Vector3::from_iterator(min.clone()),
            Vector3::from_iterator(max.clone()),
        );

        if !aabb.is_consistent() {
            return Err(anyhow!("The user specified {error_str} min/max values are inconsistent! min: {:?} max: {:?}", aabb.min().as_slice(), aabb.max().as_slice()));
        }

        if aabb.is_degenerate() {
            return Err(anyhow!(
                "The user specified {error_str} is degenerate! min: {:?} max: {:?}",
                aabb.min().as_slice(),
                aabb.max().as_slice()
            ));
        }

        Ok(aabb)
    }

    // Convert raw command line arguments to more useful types
    impl TryFrom<&ReconstructSubcommandArgs> for ReconstructionRunnerArgs {
        type Error = anyhow::Error;

        fn try_from(args: &ReconstructSubcommandArgs) -> Result<Self, Self::Error> {
            // Convert particle domain args to aabb
            let particle_aabb = if let (Some(domain_min), Some(domain_max)) =
                (&args.particle_aabb_min, &args.particle_aabb_max)
            {
                Some(try_aabb_from_min_max(
                    domain_min,
                    domain_max,
                    "particle AABB",
                )?)
            } else {
                None
            };

            // Convert mesh domain args to aabb
            let mesh_aabb = if let (Some(mesh_min), Some(mesh_max)) =
                (&args.mesh_aabb_min, &args.mesh_aabb_max)
            {
                Some(try_aabb_from_min_max(mesh_min, mesh_max, "mesh AABB")?)
            } else {
                None
            };

            // Scale kernel radius and cube size by particle radius
            let compact_support_radius = args.particle_radius * 2.0 * args.smoothing_length;
            let cube_size = args.particle_radius * args.cube_size;

            let spatial_decomposition = if args.subdomain_grid.into_bool() {
                Some(splashsurf_lib::SpatialDecomposition::UniformGrid(
                    splashsurf_lib::GridDecompositionParameters {
                        subdomain_num_cubes_per_dim: args.subdomain_cubes,
                    },
                ))
            } else if args.octree_decomposition.into_bool() {
                let subdivision_criterion = if let Some(max_particles) = args.octree_max_particles {
                    splashsurf_lib::SubdivisionCriterion::MaxParticleCount(max_particles)
                } else {
                    splashsurf_lib::SubdivisionCriterion::MaxParticleCountAuto
                };
                let ghost_particle_safety_factor = args.octree_ghost_margin_factor;
                let enable_stitching = args.octree_stitch_subdomains.into_bool();

                let particle_density_computation = if args.octree_global_density.into_bool()
                    && args.octree_sync_local_density.into_bool()
                {
                    return Err(anyhow!("Cannot enable both global and merged local particle density computation at the same time. Switch off at least one."));
                } else {
                    if args.octree_global_density.into_bool() {
                        ParticleDensityComputationStrategy::Global
                    } else if args.octree_sync_local_density.into_bool() {
                        ParticleDensityComputationStrategy::SynchronizeSubdomains
                    } else {
                        ParticleDensityComputationStrategy::IndependentSubdomains
                    }
                };

                Some(splashsurf_lib::SpatialDecomposition::Octree(
                    splashsurf_lib::OctreeDecompositionParameters {
                        subdivision_criterion,
                        ghost_particle_safety_factor,
                        enable_stitching,
                        particle_density_computation,
                    },
                ))
            } else {
                None
            };

            // Assemble all parameters for the surface reconstruction
            let params = splashsurf_lib::Parameters {
                particle_radius: args.particle_radius,
                rest_density: args.rest_density,
                compact_support_radius,
                cube_size,
                iso_surface_threshold: args.surface_threshold,
                particle_aabb,
                enable_multi_threading: args.parallelize_over_particles.into_bool(),
                spatial_decomposition,
            };

            // Optionally initialize thread pool
            if let Some(num_threads) = args.num_threads {
                splashsurf_lib::initialize_thread_pool(num_threads)?;
            }

            let postprocessing = ReconstructionRunnerPostprocessingArgs {
                check_mesh: args.check_mesh.into_bool(),
                compute_normals: args.normals.into_bool(),
                sph_normals: args.sph_normals.into_bool(),
                interpolate_attributes: args.interpolate_attributes.clone(),
                mesh_aabb,
            };

            Ok(ReconstructionRunnerArgs {
                params,
                use_double_precision: args.double_precision.into_bool(),
                io_params: io::FormatParameters::default(),
                postprocessing,
            })
        }
    }

    #[derive(Clone, Debug)]
    pub struct ReconstructionRunnerPathCollection {
        is_sequence: bool,
        input_file: PathBuf,
        output_file: PathBuf,
        output_density_map_points_file: Option<PathBuf>,
        output_density_map_grid_file: Option<PathBuf>,
        output_octree_file: Option<PathBuf>,
        sequence_range: (Option<usize>, Option<usize>),
    }

    impl ReconstructionRunnerPathCollection {
        fn try_new<P: Into<PathBuf>>(
            is_sequence: bool,
            input_file: P,
            output_base_path: Option<P>,
            output_file: P,
            output_density_map_points_file: Option<P>,
            output_density_map_grid_file: Option<P>,
            output_octree_file: Option<P>,
            sequence_range: (Option<usize>, Option<usize>),
        ) -> Result<Self, anyhow::Error> {
            let input_file = input_file.into();
            let output_base_path = output_base_path.map(|p| p.into());
            let output_file = output_file.into();
            let output_density_map_points_file = output_density_map_points_file.map(|p| p.into());
            let output_density_map_grid_file = output_density_map_grid_file.map(|p| p.into());
            let output_octree_file = output_octree_file.map(|p| p.into());

            if let (Some(start), Some(end)) = sequence_range {
                if start > end {
                    return Err(anyhow!(
                        "Invalid input sequence range: \"{} to {}\"",
                        start,
                        end
                    ));
                }
            }

            if let Some(output_base_path) = output_base_path {
                let output_file = output_base_path.join(output_file);

                // Ensure that output directory exists/create it
                if let Some(output_dir) = output_file.parent() {
                    if !output_dir.exists() {
                        info!("The output directory \"{}\" of the output file \"{}\" does not exist. Trying to create it now...", output_dir.display(), output_file.display());
                        fs::create_dir_all(output_dir).with_context(|| {
                            format!(
                                "Unable to create output directory \"{}\"",
                                output_dir.display()
                            )
                        })?;
                    }
                }

                Ok(Self {
                    is_sequence,
                    input_file,
                    output_file,
                    output_density_map_points_file: output_density_map_points_file
                        .map(|f| output_base_path.join(f)),
                    output_density_map_grid_file: output_density_map_grid_file
                        .map(|f| output_base_path.join(f)),
                    output_octree_file: output_octree_file.map(|f| output_base_path.join(f)),
                    sequence_range,
                })
            } else {
                Ok(Self {
                    is_sequence,
                    input_file,
                    output_file,
                    output_density_map_points_file,
                    output_density_map_grid_file,
                    output_octree_file,
                    sequence_range,
                })
            }
        }

        /// Returns an input/output file path struct for each input file (basically one task per input file)
        pub(crate) fn collect(&self) -> Vec<ReconstructionRunnerPaths> {
            if self.is_sequence {
                let input_file = &self.input_file;
                let output_file = &self.output_file;

                let input_dir = input_file
                    .parent()
                    .expect("expected an input path ending in a filename");
                let output_dir = output_file
                    .parent()
                    .expect("expected an output path ending in a filename");

                let input_pattern = input_file.file_name().unwrap().to_string_lossy();
                let output_pattern = output_file.file_name().unwrap().to_string_lossy();

                let (input_prefix, input_suffix) = input_pattern
                    .split_once("{}")
                    .expect("sequence input filename has to include pattern");

                let input_re_str =
                    format!(r"{}(\d+){}", escape(input_prefix), escape(input_suffix));
                let input_re = Regex::new(&input_re_str).expect("expected a valid regex");

                let input_root = if input_dir == Path::new("") {
                    Path::new(".")
                } else {
                    input_dir
                };
                info!(
                    "Looking for input sequence files in root \"{}\"",
                    input_root.display()
                );

                let mut paths = Vec::new();

                for entry in WalkDir::new(input_root)
                    .max_depth(1)
                    .contents_first(true)
                    .sort_by(|a, b| {
                        let a = a.file_name().to_string_lossy();
                        let b = b.file_name().to_string_lossy();
                        lexical_sort::natural_cmp(&a, &b)
                    })
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_type().is_file())
                {
                    let entry_name = entry.file_name().to_string_lossy();
                    if input_re.is_match(&entry_name) {
                        let index = &input_re
                            .captures(&entry_name)
                            .expect("there should be a match")[1];
                        let index_usize =
                            usize::from_str(index).expect("index should be convertible to usize");

                        if let Some(start) = self.sequence_range.0 {
                            if index_usize < start {
                                continue;
                            }
                        }

                        if let Some(end) = self.sequence_range.1 {
                            if index_usize > end {
                                continue;
                            }
                        }

                        let input_filename_i = entry_name.as_ref();
                        let input_file_i = input_dir.join(input_filename_i);

                        let output_filename_i = output_pattern.replace("{}", index);
                        let output_file_i = output_dir.join(output_filename_i);

                        paths.push(ReconstructionRunnerPaths::new(
                            input_file_i,
                            output_file_i,
                            // Don't write density maps etc. when processing a sequence of files
                            None,
                            None,
                            None,
                        ));
                    }
                }

                info!(
                    "Found {} input files matching the pattern \"{}\" between in range {} to {}",
                    paths.len(),
                    input_re_str,
                    self.sequence_range
                        .0
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| "*".to_string()),
                    self.sequence_range
                        .1
                        .map(|i| i.to_string())
                        .unwrap_or_else(|| "*".to_string()),
                );
                paths
            } else {
                vec![
                    ReconstructionRunnerPaths::new(
                        self.input_file.clone(),
                        self.output_file.clone(),
                        self.output_density_map_points_file.clone(),
                        self.output_density_map_grid_file.clone(),
                        self.output_octree_file.clone(),
                    );
                    1
                ]
            }
        }
    }

    // Convert input file command line arguments to internal representation
    impl TryFrom<&ReconstructSubcommandArgs> for ReconstructionRunnerPathCollection {
        type Error = anyhow::Error;

        fn try_from(args: &ReconstructSubcommandArgs) -> Result<Self, Self::Error> {
            let output_suffix = "surface";

            // Make sure that the input path ends with a filename (and not with a path separator)
            let input_filename = match args.input_file_or_sequence.file_name() {
                Some(input_filename) => input_filename.to_string_lossy(),
                None => {
                    return Err(anyhow!(
                        "The input file path \"{}\" does not end with a filename",
                        args.input_file_or_sequence.display()
                    ))
                }
            };

            // Make sure that the parent directory of the input path exists
            if let Some(input_dir) = args.input_file_or_sequence.parent() {
                if !input_dir.is_dir() && input_dir != Path::new("") {
                    return Err(anyhow!(
                        "The parent directory \"{}\" of the input file path \"{}\" does not exist",
                        input_dir.display(),
                        args.input_file_or_sequence.display()
                    ));
                }
            }

            let is_sequence: bool;
            let output_filename: PathBuf;

            // Detect sequence pattern or existing input file
            if input_filename.contains("{}") {
                is_sequence = true;

                output_filename = if let Some(output_file) = &args.output_file {
                    let output_pattern = output_file.to_string_lossy();
                    if output_pattern.contains("{}") {
                        output_pattern.to_string().into()
                    } else {
                        return Err(anyhow!(
                            "The output filename \"{}\" does not contain a place holder \"{{}}\"",
                            output_file.display()
                        ));
                    }
                } else {
                    let input_stem = args
                        .input_file_or_sequence
                        .file_stem()
                        .unwrap()
                        .to_string_lossy();
                    // Use VTK format as default fallback
                    format!(
                        "{}.vtk",
                        input_stem.replace("{}", &format!("{}_{{}}", output_suffix))
                    )
                    .into()
                };
            } else {
                is_sequence = false;

                // Make sure that the input file actually exists
                if args.input_file_or_sequence.is_file() {
                    // Use the user defined output file name if provided...
                    output_filename = if let Some(output_file) = &args.output_file {
                        output_file.clone()
                        // ...otherwise, generate one based on the input filename
                    } else {
                        let input_stem = args
                            .input_file_or_sequence
                            .file_stem()
                            .unwrap()
                            .to_string_lossy();
                        format!("{}_{}.vtk", input_stem, output_suffix).into()
                    };
                } else {
                    return Err(anyhow!(
                        "Input file does not exist: \"{}\"",
                        args.input_file_or_sequence.display()
                    ));
                }
            }

            Self::try_new(
                is_sequence,
                args.input_file_or_sequence.clone(),
                args.output_dir.clone(),
                output_filename,
                args.output_dm_points.clone(),
                args.output_dm_grid.clone(),
                args.output_octree.clone(),
                (args.start_index, args.end_index),
            )
        }
    }

    /// All file paths that are relevant for running a single surface reconstruction task
    #[derive(Clone, Debug)]
    pub(crate) struct ReconstructionRunnerPaths {
        pub input_file: PathBuf,
        pub output_file: PathBuf,
        pub output_density_map_points_file: Option<PathBuf>,
        pub output_density_map_grid_file: Option<PathBuf>,
        pub output_octree_file: Option<PathBuf>,
    }

    impl ReconstructionRunnerPaths {
        fn new(
            input_file: PathBuf,
            output_file: PathBuf,
            output_density_map_points_file: Option<PathBuf>,
            output_density_map_grid_file: Option<PathBuf>,
            output_octree_file: Option<PathBuf>,
        ) -> Self {
            ReconstructionRunnerPaths {
                input_file,
                output_file,
                output_density_map_points_file,
                output_density_map_grid_file,
                output_octree_file,
            }
        }
    }
}

/// Calls the reconstruction pipeline for single or double precision depending on the runtime parameters
pub(crate) fn reconstruction_pipeline(
    paths: &ReconstructionRunnerPaths,
    args: &ReconstructionRunnerArgs,
) -> Result<(), anyhow::Error> {
    if args.use_double_precision {
        info!("Using double precision (f64) for surface reconstruction.");
        reconstruction_pipeline_generic::<i64, f64>(
            paths,
            &args.params,
            &args.io_params,
            &args.postprocessing,
        )?;
    } else {
        info!("Using single precision (f32) for surface reconstruction.");
        reconstruction_pipeline_generic::<i64, f32>(
            paths,
            &args.params.try_convert().ok_or(anyhow!(
                "Unable to convert surface reconstruction parameters from f64 to f32."
            ))?,
            &args.io_params,
            &args.postprocessing,
        )?;
    }

    Ok(())
}

/// Wrapper for the reconstruction pipeline: loads input file, runs reconstructions, stores output files
pub(crate) fn reconstruction_pipeline_generic<I: Index, R: Real>(
    paths: &ReconstructionRunnerPaths,
    params: &splashsurf_lib::Parameters<R>,
    io_params: &io::FormatParameters,
    postprocessing: &ReconstructionRunnerPostprocessingArgs,
) -> Result<(), anyhow::Error> {
    profile!("surface reconstruction");

    // Load particle positions and attributes to interpolate
    let (particle_positions, attributes) = io::read_particle_positions_with_attributes(
        &paths.input_file,
        &postprocessing.interpolate_attributes,
        &io_params.input,
    )
    .with_context(|| {
        format!(
            "Failed to load particle positions from file \"{}\"",
            paths.input_file.display()
        )
    })?;

    // Perform the surface reconstruction
    let reconstruction =
        splashsurf_lib::reconstruct_surface::<I, R>(particle_positions.as_slice(), &params)?;

    let grid = reconstruction.grid();
    let mesh = reconstruction.mesh();

    let mesh = if let Some(aabb) = &postprocessing.mesh_aabb {
        profile!("clamp mesh to aabb");
        info!("Post-processing: Clamping mesh to AABB...");

        let mut mesh = mesh.clone();
        mesh.clamp_with_aabb(
            &aabb
                .try_convert()
                .ok_or_else(|| anyhow!("Failed to convert mesh AABB"))?,
        );
        mesh
    } else {
        mesh.clone()
    };

    // Add normals to mesh if requested
    let mesh = if postprocessing.compute_normals || !attributes.is_empty() {
        profile!("compute normals");

        info!(
            "Constructing global acceleration structure for SPH interpolation to {} vertices...",
            mesh.vertices.len()
        );

        let particle_rest_density = params.rest_density;
        let particle_rest_volume = R::from_f64((4.0 / 3.0) * std::f64::consts::PI).unwrap()
            * params.particle_radius.powi(3);
        let particle_rest_mass = particle_rest_volume * particle_rest_density;

        let particle_densities = reconstruction
            .particle_densities()
            .ok_or_else(|| anyhow::anyhow!("Particle densities were not returned by surface reconstruction but are required for SPH normal computation"))?
            .as_slice();
        assert_eq!(
            particle_positions.len(),
            particle_densities.len(),
            "There has to be one density value per particle"
        );

        let interpolator = SphInterpolator::new(
            &particle_positions,
            particle_densities,
            particle_rest_mass,
            params.compact_support_radius,
        );

        let mut mesh_with_data = MeshWithData::new(mesh);
        let mesh = &mesh_with_data.mesh;

        // Compute normals if requested
        if postprocessing.compute_normals {
            let normals = if postprocessing.sph_normals {
                info!("Using SPH interpolation to compute surface normals");

                let sph_normals = interpolator.interpolate_normals(mesh.vertices());
                bytemuck::allocation::cast_vec::<Unit<Vector3<R>>, Vector3<R>>(sph_normals)
            } else {
                info!("Using area weighted triangle normals for surface normals");
                profile!("mesh.par_vertex_normals");
                let tri_normals = mesh.par_vertex_normals();

                // Convert unit vectors to plain vectors
                bytemuck::allocation::cast_vec::<Unit<Vector3<R>>, Vector3<R>>(tri_normals)
            };

            mesh_with_data.point_attributes.push(MeshAttribute::new(
                "normals".to_string(),
                AttributeData::Vector3Real(normals),
            ));
        }

        // Interpolate attributes if requested
        if !attributes.is_empty() {
            for attribute in attributes.into_iter() {
                info!("Interpolating attribute \"{}\"...", attribute.name);

                match attribute.data {
                    AttributeData::ScalarReal(values) => {
                        let interpolated_values = interpolator.interpolate_scalar_quantity(
                            values.as_slice(),
                            mesh.vertices(),
                            true,
                        );
                        mesh_with_data.point_attributes.push(MeshAttribute::new(
                            attribute.name,
                            AttributeData::ScalarReal(interpolated_values),
                        ));
                    }
                    AttributeData::Vector3Real(values) => {
                        let interpolated_values = interpolator.interpolate_vector_quantity(
                            values.as_slice(),
                            mesh.vertices(),
                            true,
                        );
                        mesh_with_data.point_attributes.push(MeshAttribute::new(
                            attribute.name,
                            AttributeData::Vector3Real(interpolated_values),
                        ));
                    }
                    _ => unimplemented!("Interpolation of this attribute type not implemented"),
                }
            }
        }

        mesh_with_data
    } else {
        MeshWithData::new(mesh)
    };

    // Store the surface mesh
    {
        profile!("write surface mesh to file");
        info!(
            "Writing surface mesh to \"{}\"...",
            paths.output_file.display()
        );

        io::write_mesh(&mesh, paths.output_file.clone(), &io_params.output).with_context(|| {
            anyhow!(
                "Failed to write output mesh to file \"{}\"",
                paths.output_file.display()
            )
        })?;
        info!("Done.");
    }

    // Store octree leaf nodes as hex cells
    if let Some(output_octree_file) = &paths.output_octree_file {
        info!("Writing octree to \"{}\"...", output_octree_file.display());
        io::vtk_format::write_vtk(
            reconstruction
                .octree()
                .unwrap()
                .hexmesh(grid, true)
                .to_unstructured_grid(),
            output_octree_file,
            "mesh",
        )
        .with_context(|| {
            format!(
                "Failed to write octree to output file \"{}\"",
                output_octree_file.display()
            )
        })?;
        info!("Done.");
    }

    // Store point cloud density map
    if let Some(output_density_map_points_file) = &paths.output_density_map_points_file {
        info!("Constructing density map point cloud...");
        let density_map = reconstruction
            .density_map()
            .ok_or_else(|| anyhow::anyhow!("No density map was created during reconstruction"))?;

        let point_cloud: PointCloud3d<R> = {
            let mut points = Vec::with_capacity(density_map.len());
            density_map.for_each(|flat_point_index, _| {
                let point = grid.try_unflatten_point_index(flat_point_index).unwrap();
                points.push(grid.point_coordinates(&point));
            });

            PointCloud3d::new(points)
        };

        info!(
            "Saving density map point cloud to \"{}\"...",
            output_density_map_points_file.display()
        );

        io::vtk_format::write_vtk(
            &point_cloud,
            output_density_map_points_file,
            "density_map_points",
        )?;

        info!("Done.");
    }

    // Store hex-mesh density map
    if let Some(output_density_map_grid_file) = &paths.output_density_map_grid_file {
        info!("Constructing density map hex mesh...");
        let density_map = reconstruction
            .density_map()
            .ok_or_else(|| anyhow::anyhow!("No density map was created during reconstruction"))?;

        let density_mesh =
            density_map::sparse_density_map_to_hex_mesh(&density_map, &grid, R::zero());

        info!(
            "Saving density map hex mesh to \"{}\"...",
            output_density_map_grid_file.display()
        );

        io::vtk_format::write_vtk(
            density_mesh.to_unstructured_grid(),
            output_density_map_grid_file,
            "density_map",
        )?;

        info!("Done.");
    }

    if postprocessing.check_mesh {
        if let Err(err) = splashsurf_lib::marching_cubes::check_mesh_consistency(grid, &mesh.mesh) {
            return Err(anyhow!("{}", err));
        } else {
            info!("Checked mesh for problems (holes, etc.), no problems were found.");
        }
    }

    Ok(())
}
