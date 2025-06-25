//! Implementation of the `reconstruct` subcommand of the splashsurf CLI.

use crate::cli::Switch;
use crate::reconstruct::arguments::*;
use crate::{io, logging};
use anyhow::{Context, anyhow};
use clap::value_parser;
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info, warn};
use rayon::prelude::*;
use splashsurf_lib::mesh::{
    AttributeData, Mesh3d, MeshAttribute, MeshWithData, MixedTriQuadMesh3d, TriMesh3d,
};
use splashsurf_lib::nalgebra::{Unit, Vector3};
use splashsurf_lib::sph_interpolation::SphInterpolator;
use splashsurf_lib::{Aabb3d, Index, Real, SurfaceReconstruction, profile};
use std::borrow::Cow;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::path::PathBuf;

// TODO: Detect smallest index type (i.e. check if ok to use i32 as index)

static ARGS_IO: &str = "Input/output";
static ARGS_BASIC: &str = "Numerical reconstruction parameters";
static ARGS_ADV: &str = "Advanced parameters";
static ARGS_OCTREE: &str = "Domain decomposition parameters";
static ARGS_DEBUG: &str = "Debug options";
static ARGS_INTERP: &str = "Interpolation & normals";
static ARGS_DECIMATE: &str = "Mesh decimation and cleanup";
static ARGS_SMOOTHING: &str = "Mesh smoothing";
static ARGS_POSTPROC: &str = "General postprocessing";
static ARGS_OTHER: &str = "Remaining options";

/// Command line arguments for the `reconstruct` subcommand
#[derive(Clone, Debug, clap::Parser)]
#[clap(group = clap::ArgGroup::new("input").required(true))]
#[command(next_help_heading = ARGS_OTHER)]
pub(crate) struct ReconstructSubcommandArgs {
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
    /// The smoothing length radius used for the SPH kernel, the kernel compact support radius will be twice the smoothing length (in multiples of the particle radius)
    #[arg(help_heading = ARGS_BASIC, short = 'l', long)]
    pub smoothing_length: f64,
    /// The cube edge length used for marching cubes in multiples of the particle radius, corresponds to the cell size of the implicit background grid
    #[arg(help_heading = ARGS_BASIC, short = 'c', long)]
    pub cube_size: f64,
    /// The iso-surface threshold used for the marching cubes algorithm, this is the value of the implicit surface function (here the color field) at which the surface is reconstructed
    #[arg(help_heading = ARGS_BASIC, short = 't', long, default_value = "0.6")]
    pub surface_threshold: f64,

    /// Enable the use of double precision for all computations
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

    /// Enable multithreading to process multiple input files in parallel (NOTE: Currently, the subdomain-grid domain decomposition approach and some post-processing functions including interpolation do not have sequential versions and therefore do not work well with this option enabled)
    #[arg(
        help_heading = ARGS_ADV,
        long = "mt-files",
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub parallelize_over_files: Switch,
    /// Enable multithreading for a single input file by processing chunks of particles in parallel
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

    /// Enable spatial decomposition using a regular grid-based approach (for efficient multithreading)
    #[arg(
        help_heading = ARGS_OCTREE,
        long,
        default_value = "on",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub subdomain_grid: Switch,
    /// Each subdomain will be a cube consisting of this number of MC grid cells along each coordinate axis
    #[arg(help_heading = ARGS_OCTREE, long, default_value="64")]
    pub subdomain_cubes: u32,

    /// Enable computing surface normals at the mesh vertices and write them to the output file
    #[arg(
        help_heading = ARGS_INTERP,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub normals: Switch,
    /// Enable computing the normals using SPH interpolation instead of using the area weighted triangle normals
    #[arg(
        help_heading = ARGS_INTERP,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub sph_normals: Switch,
    /// Number of smoothing iterations to apply to normals if normal interpolation is enabled (disabled by default)
    #[arg(help_heading = ARGS_INTERP, long)]
    pub normals_smoothing_iters: Option<usize>,
    /// Enable writing raw normals without smoothing to the output mesh if normal smoothing is enabled
    #[arg(
        help_heading = ARGS_INTERP,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub output_raw_normals: Switch,
    /// Interpolate a point attribute field with the given name from the input file to the reconstructed surface. Currently, this is only supported for BGEO, VTK and VTU input files. Specify the argument multiple times for each attribute that should be interpolated.
    #[arg(help_heading = ARGS_INTERP, long = "interpolate_attribute", short = 'a', value_name = "ATTRIBUTE_NAME")]
    pub interpolate_attributes: Vec<String>,

    /// Enable MC specific mesh decimation/simplification which removes bad quality triangles typically generated by MC by snapping (enabled by default if smoothing is enabled)
    #[arg(
        help_heading = ARGS_DECIMATE,
        long,
        default_value = "off",
        default_value_ifs([
            ("mesh_smoothing_iters", clap::builder::ArgPredicate::Equals("0".into()), "off"),
            ("mesh_smoothing_iters", clap::builder::ArgPredicate::IsPresent, "on")
        ]),
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub mesh_cleanup: Switch,
    /// If MC mesh cleanup is enabled, vertex snapping can be limited to this distance relative to the MC edge length (should be in range of [0.0,0.5])
    #[arg(
        help_heading = ARGS_DECIMATE,
        long,
    )]
    pub mesh_cleanup_snap_dist: Option<f64>,
    /// Enable decimation of some typical bad marching cubes triangle configurations (resulting in "barnacles" after Laplacian smoothing)
    #[arg(
        help_heading = ARGS_DECIMATE,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub decimate_barnacles: Switch,
    /// Enable preserving vertices without connectivity during decimation instead of filtering them out (faster and helps with debugging)
    #[arg(
        help_heading = ARGS_DECIMATE,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub keep_verts: Switch,

    /// Number of smoothing iterations to run on the reconstructed mesh
    #[arg(help_heading = ARGS_SMOOTHING, long)]
    pub mesh_smoothing_iters: Option<usize>,
    /// Enable feature weights for mesh smoothing if mesh smoothing enabled. Preserves isolated particles even under strong smoothing.
    #[arg(
        help_heading = ARGS_SMOOTHING,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub mesh_smoothing_weights: Switch,
    /// Override a manual normalization value from weighted number of neighbors to mesh smoothing weights
    #[arg(help_heading = ARGS_SMOOTHING, long, default_value = "13.0")]
    pub mesh_smoothing_weights_normalization: f64,
    /// Enable writing the smoothing weights as a vertex attribute to the output mesh file
    #[arg(
        help_heading = ARGS_SMOOTHING,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub output_smoothing_weights: Switch,

    /// Enable conversion of triangles to quads if they meet quality criteria
    #[arg(
        help_heading = ARGS_POSTPROC,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub generate_quads: Switch,
    /// Maximum allowed ratio of quad edge lengths to its diagonals to merge two triangles to a quad (inverse is used for minimum)
    #[arg(help_heading = ARGS_POSTPROC, long, default_value = "1.75")]
    pub quad_max_edge_diag_ratio: f64,
    /// Maximum allowed angle (in degrees) between triangle normals to merge them to a quad
    #[arg(help_heading = ARGS_POSTPROC, long, default_value = "10")]
    pub quad_max_normal_angle: f64,
    /// Maximum allowed vertex interior angle (in degrees) inside a quad to merge two triangles to a quad
    #[arg(help_heading = ARGS_POSTPROC, long, default_value = "135")]
    pub quad_max_interior_angle: f64,

    /// Lower corner of the bounding-box for the surface mesh, triangles completely outside are removed (requires mesh-aabb-max to be specified)
    #[arg(
        help_heading = ARGS_POSTPROC,
        long,
        number_of_values = 3,
        value_names = ["X_MIN", "Y_MIN", "Z_MIN"],
        allow_negative_numbers = true,
        requires = "mesh_aabb_max",
    )]
    pub mesh_aabb_min: Option<Vec<f64>>,
    /// Upper corner of the bounding-box for the surface mesh, triangles completely outside are removed (requires mesh-aabb-min to be specified)
    #[arg(
        help_heading = ARGS_POSTPROC,
        long,
        number_of_values = 3,
        value_names = ["X_MIN", "Y_MIN", "Z_MIN"],
        allow_negative_numbers = true,
        requires = "mesh_aabb_min",
    )]
    pub mesh_aabb_max: Option<Vec<f64>>,
    /// Enable clamping of vertices outside the specified mesh AABB to the AABB (only has an effect if mesh-aabb-min/max are specified)
    #[arg(
        help_heading = ARGS_POSTPROC,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub mesh_aabb_clamp_verts: Switch,

    /// Enable writing the raw reconstructed mesh before applying any post-processing steps (like smoothing or decimation)
    #[arg(
        help_heading = ARGS_POSTPROC,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub output_raw_mesh: Switch,

    /// Enable checking the final mesh for holes and non-manifold edges and vertices
    #[arg(
        help_heading = ARGS_DEBUG,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub check_mesh: Switch,
    /// Enable checking the final mesh for holes
    #[arg(
        help_heading = ARGS_DEBUG,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub check_mesh_closed: Switch,
    /// Enable checking the final mesh for non-manifold edges and vertices
    #[arg(
        help_heading = ARGS_DEBUG,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub check_mesh_manifold: Switch,
    /// Enable checking the final mesh for inverted triangles (compares angle between vertex normals and adjacent face normals)
    #[arg(
        help_heading = ARGS_DEBUG,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub check_mesh_orientation: Switch,
    /// Enable additional debug output for the check-mesh operations (has no effect if no other check-mesh option is enabled)
    #[arg(
        help_heading = ARGS_DEBUG,
        long,
        default_value = "off",
        value_name = "off|on",
        ignore_case = true,
        require_equals = true
    )]
    pub check_mesh_debug: Switch,
}

/// Executes the `reconstruct` subcommand
pub(crate) fn reconstruct_subcommand(
    cmd_args: &ReconstructSubcommandArgs,
) -> Result<(), anyhow::Error> {
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
            reconstruction_pipeline_from_args(path, &args)
                .with_context(|| {
                    format!(
                        "Error while processing input file \"{}\" from a file sequence",
                        path.input_file.display()
                    )
                })
                .inspect_err(|err| {
                    // Already log the error in case there are multiple errors
                    logging::log_error(err);
                })
                .map(|_| {
                    if let Some(pb) = logging::get_progress_bar() {
                        pb.inc(1)
                    }
                })
        })
    } else {
        paths.iter().try_for_each(|path| {
            reconstruction_pipeline_from_args(path, &args).map(|_| {
                if let Some(pb) = logging::get_progress_bar() {
                    pb.inc(1)
                }
            })
        })
    };

    if paths.len() > 1 {
        if let Some(pb) = logging::get_progress_bar() {
            pb.finish()
        }
        logging::set_progress_bar(None);
    }

    if result.is_ok() {
        info!("Successfully finished processing all inputs.");
    }

    result
}

/// Struct returned as output of the [`reconstruction_pipeline`] function
pub struct ReconstructionResult<I: Index, R: Real> {
    /// Holds the reconstructed triangle mesh (if [`generate_quads`](ReconstructionPostprocessingParameters::generate_quads) was not enabled)
    pub tri_mesh: Option<MeshWithData<R, TriMesh3d<R>>>,
    /// Holds the reconstructed mixed triangle/quad mesh (only if [`generate_quads`](ReconstructionPostprocessingParameters::generate_quads) was enabled)
    pub tri_quad_mesh: Option<MeshWithData<R, MixedTriQuadMesh3d<R>>>,
    /// Holds the initial [`SurfaceReconstruction`] with no post-processing applied (only if [`output_raw_mesh`](ReconstructionPostprocessingParameters::output_raw_mesh) was enabled)
    pub raw_reconstruction: Option<SurfaceReconstruction<I, R>>,
}

/// Parameters for the post-processing steps in the reconstruction pipeline
#[derive(Clone, Debug)]
pub struct ReconstructionPostprocessingParameters {
    /// Enable checking the final mesh for holes
    pub check_mesh_closed: bool,
    /// Enable checking the final mesh for non-manifold edges and vertices
    pub check_mesh_manifold: bool,
    /// Enable checking the final mesh for inverted triangles (compares angle between vertex normals and adjacent face normals)
    pub check_mesh_orientation: bool,
    /// Enable additional debug output for the check-mesh operations (has no effect if no other check-mesh option is enabled)
    pub check_mesh_debug: bool,
    /// Enable MC specific mesh decimation/simplification to remove bad quality triangles typically generated by MC by snapping (enabled by default if smoothing is enabled)
    pub mesh_cleanup: bool,
    /// If MC mesh cleanup is enabled, vertex snapping can be limited to this distance relative to the MC edge length (should be in the interval `[0.0,0.5]`)
    pub mesh_cleanup_snap_dist: Option<f64>,
    /// Enable decimation of some typical bad marching cubes triangle configurations (resulting in "barnacles" after Laplacian smoothing)
    pub decimate_barnacles: bool,
    /// Enable preserving vertices without connectivity during decimation instead of filtering them out (faster and helps with debugging)
    pub keep_vertices: bool,
    /// Enable computing per-vertex surface normals
    pub compute_normals: bool,
    /// Enable computing the normals using SPH interpolation instead of using the area weighted triangle normals
    pub sph_normals: bool,
    /// Number of Laplacian smoothing iterations to apply to normals if normal interpolation is enabled
    pub normals_smoothing_iters: Option<usize>,
    /// Interpolate point attributes with the given names from the input attributes to the reconstructed surface
    pub interpolate_attributes: Vec<String>,
    /// Number of Laplacian smoothing iterations to apply t the reconstructed mesh
    pub mesh_smoothing_iters: Option<usize>,
    /// Enable feature weights for mesh smoothing if mesh smoothing enabled. Preserves isolated particles even under strong smoothing.
    pub mesh_smoothing_weights: bool,
    /// Override the normalization value from weighted number of neighbors to mesh smoothing weights
    pub mesh_smoothing_weights_normalization: f64,
    /// Enable conversion of triangles to quads if they meet quality criteria
    pub generate_quads: bool,
    /// Maximum allowed ratio of quad edge lengths to its diagonals to merge two triangles to a quad (inverse is used for minimum)
    pub quad_max_edge_diag_ratio: f64,
    /// Maximum allowed angle (in degrees) between triangle normals to merge them to a quad
    pub quad_max_normal_angle: f64,
    /// Maximum allowed vertex interior angle (in degrees) inside a quad to merge two triangles to a quad
    pub quad_max_interior_angle: f64,
    /// Enable storing the smoothing weights as a vertex attribute to the output mesh
    pub output_mesh_smoothing_weights: bool,
    /// Enable storing the raw normals without smoothing to the output mesh if normal smoothing is enabled
    pub output_raw_normals: bool,
    /// Enable additionally returning the initial [`SurfaceReconstruction`] before any post-processing was applied
    pub output_raw_mesh: bool,
    /// Bounding-box for the surface mesh, triangles completely outside are removed
    pub mesh_aabb: Option<Aabb3d<f64>>,
    /// Enable clamping of vertices outside the specified mesh AABB to the AABB (only has an effect if mesh-aabb is specified)
    pub mesh_aabb_clamp_vertices: bool,
}

/// Conversion and validation of command line arguments
pub(crate) mod arguments {
    use super::{ReconstructSubcommandArgs, ReconstructionPostprocessingParameters};
    use crate::io;
    use anyhow::{Context, anyhow};
    use log::info;
    use regex::{Regex, escape};
    use splashsurf_lib::Aabb3d;
    use splashsurf_lib::nalgebra::Vector3;
    use std::convert::TryFrom;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::str::FromStr;
    use walkdir::WalkDir;

    /// All arguments that can be supplied to the surface reconstruction tool converted to useful types
    pub struct ReconstructionRunnerArgs {
        /// Parameters passed directly to the surface reconstruction
        pub params: splashsurf_lib::Parameters<f64>,
        pub use_double_precision: bool,
        pub io_params: io::FormatParameters,
        pub postprocessing: ReconstructionPostprocessingParameters,
    }

    fn try_aabb_from_min_max(
        min: &[f64],
        max: &[f64],
        error_str: &'static str,
    ) -> Result<Aabb3d<f64>, anyhow::Error> {
        // This should already be ensured by StructOpt parsing
        assert_eq!(min.len(), 3);
        assert_eq!(max.len(), 3);

        let aabb = Aabb3d::new(
            Vector3::from_column_slice(min),
            Vector3::from_column_slice(max),
        );

        if !aabb.is_consistent() {
            return Err(anyhow!(
                "The user specified {error_str} min/max values are inconsistent! min: {:?} max: {:?}",
                aabb.min().as_slice(),
                aabb.max().as_slice()
            ));
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
                        ..Default::default()
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
                global_neighborhood_list: args.mesh_smoothing_weights.into_bool(),
            };

            // Optionally initialize thread pool
            if let Some(num_threads) = args.num_threads {
                splashsurf_lib::initialize_thread_pool(num_threads)?;
            }

            let postprocessing = ReconstructionPostprocessingParameters {
                check_mesh_closed: args.check_mesh.into_bool()
                    || args.check_mesh_closed.into_bool(),
                check_mesh_manifold: args.check_mesh.into_bool()
                    || args.check_mesh_manifold.into_bool(),
                check_mesh_orientation: args.check_mesh.into_bool()
                    || args.check_mesh_orientation.into_bool(),
                check_mesh_debug: args.check_mesh_debug.into_bool(),
                mesh_cleanup: args.mesh_cleanup.into_bool(),
                mesh_cleanup_snap_dist: args.mesh_cleanup_snap_dist,
                decimate_barnacles: args.decimate_barnacles.into_bool(),
                keep_vertices: args.keep_verts.into_bool(),
                compute_normals: args.normals.into_bool(),
                sph_normals: args.sph_normals.into_bool(),
                normals_smoothing_iters: args.normals_smoothing_iters,
                interpolate_attributes: args.interpolate_attributes.clone(),
                mesh_smoothing_iters: args.mesh_smoothing_iters,
                mesh_smoothing_weights: args.mesh_smoothing_weights.into_bool(),
                mesh_smoothing_weights_normalization: args.mesh_smoothing_weights_normalization,
                generate_quads: args.generate_quads.into_bool(),
                quad_max_edge_diag_ratio: args.quad_max_edge_diag_ratio,
                quad_max_normal_angle: args.quad_max_normal_angle,
                quad_max_interior_angle: args.quad_max_interior_angle,
                output_mesh_smoothing_weights: args.output_smoothing_weights.into_bool(),
                output_raw_normals: args.output_raw_normals.into_bool(),
                output_raw_mesh: args.output_raw_mesh.into_bool(),
                mesh_aabb,
                mesh_aabb_clamp_vertices: args.mesh_aabb_clamp_verts.into_bool(),
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
        sequence_range: (Option<usize>, Option<usize>),
    }

    impl ReconstructionRunnerPathCollection {
        fn try_new<P: Into<PathBuf>>(
            is_sequence: bool,
            input_file: P,
            output_base_path: Option<P>,
            output_file: P,
            sequence_range: (Option<usize>, Option<usize>),
        ) -> Result<Self, anyhow::Error> {
            let input_file = input_file.into();
            let output_base_path = output_base_path.map(|p| p.into());
            let output_file = output_file.into();

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
                        info!(
                            "The output directory \"{}\" of the output file \"{}\" does not exist. Trying to create it now...",
                            output_dir.display(),
                            output_file.display()
                        );
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
                    sequence_range,
                })
            } else {
                Ok(Self {
                    is_sequence,
                    input_file,
                    output_file,
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

                        paths.push(ReconstructionRunnerPaths::new(input_file_i, output_file_i));
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
                    ));
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
                (args.start_index, args.end_index),
            )
        }
    }

    /// All file paths that are relevant for running a single surface reconstruction task
    #[derive(Clone, Debug)]
    pub(crate) struct ReconstructionRunnerPaths {
        pub input_file: PathBuf,
        pub output_file: PathBuf,
    }

    impl ReconstructionRunnerPaths {
        fn new(input_file: PathBuf, output_file: PathBuf) -> Self {
            ReconstructionRunnerPaths {
                input_file,
                output_file,
            }
        }
    }
}

/// Calls the reconstruction pipeline for single or double precision depending on the runtime parameters
pub(crate) fn reconstruction_pipeline_from_args(
    paths: &ReconstructionRunnerPaths,
    args: &ReconstructionRunnerArgs,
) -> Result<(), anyhow::Error> {
    if args.use_double_precision {
        info!("Using double precision (f64) for surface reconstruction.");
        reconstruction_pipeline_from_path::<i64, f64>(
            paths,
            &args.params,
            &args.io_params,
            &args.postprocessing,
        )?;
    } else {
        info!("Using single precision (f32) for surface reconstruction.");
        reconstruction_pipeline_from_path::<i64, f32>(
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

/// Performs a surface reconstruction including optional post-processing steps
///
/// This function implements the surface reconstruction pipeline used by the `reconstruct` subcommand
/// of the `splashsurf` CLI.
/// Inputs are the particle positions, a (possibly empty) list of attributes defined on the particles,
/// [`Parameters`](splashsurf_lib::Parameters) for the surface reconstruction itself, and a set of parameters for optional
/// post-processing steps.
/// Please note that, unlike the CLI, the parameters for the surface reconstruction are not relative
/// to the particle radius but absolute values.
pub fn reconstruction_pipeline<I: Index, R: Real>(
    particle_positions: &[Vector3<R>],
    attributes: Vec<MeshAttribute<R>>,
    params: &splashsurf_lib::Parameters<R>,
    postprocessing: &ReconstructionPostprocessingParameters,
) -> Result<ReconstructionResult<I, R>, anyhow::Error> {
    // Perform the surface reconstruction
    let reconstruction = splashsurf_lib::reconstruct_surface::<I, R>(particle_positions, params)?;

    let reconstruction_output = if postprocessing.output_raw_mesh {
        Some(reconstruction.clone())
    } else {
        None
    };

    let grid = reconstruction.grid();
    let mut mesh_with_data = MeshWithData::new(Cow::Borrowed(reconstruction.mesh()));

    // Perform post-processing
    {
        profile!("postprocessing");
        let mut vertex_connectivity = None;

        if postprocessing.mesh_cleanup {
            info!("Post-processing: Performing mesh cleanup");
            let tris_before = mesh_with_data.mesh.triangles.len();
            let verts_before = mesh_with_data.mesh.vertices.len();
            vertex_connectivity = Some(splashsurf_lib::postprocessing::marching_cubes_cleanup(
                mesh_with_data.mesh.to_mut(),
                reconstruction.grid(),
                postprocessing
                    .mesh_cleanup_snap_dist
                    .map(|d| R::from_float(d)),
                5,
                postprocessing.keep_vertices,
            ));
            let tris_after = mesh_with_data.mesh.triangles.len();
            let verts_after = mesh_with_data.mesh.vertices.len();
            info!(
                "Post-processing: Cleanup reduced number of vertices to {:.2}% and number of triangles to {:.2}% of original mesh.",
                (verts_after as f64 / verts_before as f64) * 100.0,
                (tris_after as f64 / tris_before as f64) * 100.0
            )
        }

        // Decimate mesh if requested
        if postprocessing.decimate_barnacles {
            info!("Post-processing: Performing decimation");
            vertex_connectivity = Some(splashsurf_lib::postprocessing::decimation(
                mesh_with_data.mesh.to_mut(),
                postprocessing.keep_vertices,
            ));
        }

        // Initialize SPH interpolator if required later
        let interpolator_required = postprocessing.mesh_smoothing_weights
            || postprocessing.sph_normals
            || !attributes.is_empty();
        let interpolator = if interpolator_required {
            profile!("initialize interpolator");
            info!("Post-processing: Initializing interpolator...");

            info!(
                "Constructing global acceleration structure for SPH interpolation to {} vertices...",
                mesh_with_data.vertices().len()
            );

            let particle_rest_density = params.rest_density;
            let particle_rest_volume =
                R::from_float(4.0) * R::frac_pi_3() * params.particle_radius.powi(3);
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

            Some(SphInterpolator::new(
                &particle_positions,
                particle_densities,
                particle_rest_mass,
                params.compact_support_radius,
            ))
        } else {
            None
        };

        // Compute mesh vertex-vertex connectivity map if required later
        let vertex_connectivity_required = postprocessing.normals_smoothing_iters.is_some()
            || postprocessing.mesh_smoothing_iters.is_some();
        if vertex_connectivity.is_none() && vertex_connectivity_required {
            vertex_connectivity = Some(mesh_with_data.mesh.vertex_vertex_connectivity());
        }

        // Compute smoothing weights if requested
        let smoothing_weights = if postprocessing.mesh_smoothing_weights {
            profile!("compute smoothing weights");
            info!("Post-processing: Computing smoothing weights...");

            // TODO: Switch between parallel/single threaded
            // TODO: Re-use data from reconstruction?

            // Global neighborhood search
            let nl = reconstruction
                .particle_neighbors()
                .map(Cow::Borrowed)
                .unwrap_or_else(||
                    {
                        let search_radius = params.compact_support_radius;

                        let mut domain = Aabb3d::from_points(particle_positions);
                        domain.grow_uniformly(search_radius);

                        let mut nl = Vec::new();
                        splashsurf_lib::neighborhood_search::neighborhood_search_spatial_hashing_parallel::<I, R>(
                            &domain,
                            particle_positions,
                            search_radius,
                            &mut nl,
                        );
                        assert_eq!(nl.len(), particle_positions.len());
                        Cow::Owned(nl)
                    }
                );

            // Compute weighted neighbor count
            let squared_r = params.compact_support_radius * params.compact_support_radius;
            let weighted_ncounts = nl
                .par_iter()
                .enumerate()
                .map(|(i, nl)| {
                    nl.iter()
                        .copied()
                        .map(|j| {
                            let dist =
                                (particle_positions[i] - particle_positions[j]).norm_squared();

                            R::one() - (dist / squared_r).clamp(R::zero(), R::one())
                        })
                        .fold(R::zero(), R::add)
                })
                .collect::<Vec<_>>();

            let vertex_weighted_num_neighbors = {
                profile!("interpolate weighted neighbor counts");
                interpolator
                    .as_ref()
                    .expect("interpolator is required")
                    .interpolate_scalar_quantity(
                        weighted_ncounts.as_slice(),
                        mesh_with_data.vertices(),
                        true,
                    )
            };

            let smoothing_weights = {
                let offset = R::zero();
                let normalization =
                    R::from_f64(postprocessing.mesh_smoothing_weights_normalization).expect(
                        "smoothing weight normalization value cannot be represented as Real type",
                    ) - offset;

                // Normalize number of neighbors
                let smoothing_weights = vertex_weighted_num_neighbors
                    .par_iter()
                    .copied()
                    .map(|n| (n - offset).max(R::zero()))
                    .map(|n| (n / normalization).min(R::one()))
                    // Smooth-Step function
                    .map(|x| x.powi(5).times(6) - x.powi(4).times(15) + x.powi(3).times(10))
                    .collect::<Vec<_>>();

                if postprocessing.output_mesh_smoothing_weights {
                    // Raw distance-weighted number of neighbors value per vertex (can be used to determine normalization value)
                    mesh_with_data.point_attributes.push(MeshAttribute::new(
                        "wnn".to_string(),
                        AttributeData::ScalarReal(vertex_weighted_num_neighbors),
                    ));
                    // Final smoothing weights per vertex
                    mesh_with_data.point_attributes.push(MeshAttribute::new(
                        "sw".to_string(),
                        AttributeData::ScalarReal(smoothing_weights.clone()),
                    ));
                }

                smoothing_weights
            };

            Some(smoothing_weights)
        } else {
            None
        };

        // Perform smoothing if requested
        if let Some(mesh_smoothing_iters) = postprocessing.mesh_smoothing_iters {
            profile!("mesh smoothing");
            info!("Post-processing: Smoothing mesh...");

            // TODO: Switch between parallel/single threaded

            let smoothing_weights = smoothing_weights
                .unwrap_or_else(|| vec![R::one(); mesh_with_data.vertices().len()]);

            splashsurf_lib::postprocessing::par_laplacian_smoothing_inplace(
                mesh_with_data.mesh.to_mut(),
                vertex_connectivity
                    .as_ref()
                    .expect("vertex connectivity is required"),
                mesh_smoothing_iters,
                R::one(),
                &smoothing_weights,
            );
        }

        // Add normals to mesh if requested
        if postprocessing.compute_normals {
            profile!("compute normals");
            info!("Post-processing: Computing surface normals...");

            // Compute normals
            let normals = if postprocessing.sph_normals {
                info!("Using SPH interpolation to compute surface normals");

                let sph_normals = interpolator
                    .as_ref()
                    .expect("interpolator is required")
                    .interpolate_normals(mesh_with_data.vertices());
                bytemuck::allocation::cast_vec::<Unit<Vector3<R>>, Vector3<R>>(sph_normals)
            } else {
                info!("Using area weighted triangle normals for surface normals");
                profile!("mesh.par_vertex_normals");
                let tri_normals = mesh_with_data.mesh.par_vertex_normals();

                // Convert unit vectors to plain vectors
                bytemuck::allocation::cast_vec::<Unit<Vector3<R>>, Vector3<R>>(tri_normals)
            };

            // Smooth normals
            if let Some(smoothing_iters) = postprocessing.normals_smoothing_iters {
                info!("Post-processing: Smoothing normals...");

                let mut smoothed_normals = normals.clone();
                splashsurf_lib::postprocessing::par_laplacian_smoothing_normals_inplace(
                    &mut smoothed_normals,
                    vertex_connectivity
                        .as_ref()
                        .expect("vertex connectivity is required"),
                    smoothing_iters,
                );

                mesh_with_data.point_attributes.push(MeshAttribute::new(
                    "normals".to_string(),
                    AttributeData::Vector3Real(smoothed_normals),
                ));
                if postprocessing.output_raw_normals {
                    mesh_with_data.point_attributes.push(MeshAttribute::new(
                        "raw_normals".to_string(),
                        AttributeData::Vector3Real(normals),
                    ));
                }
            } else {
                mesh_with_data.point_attributes.push(MeshAttribute::new(
                    "normals".to_string(),
                    AttributeData::Vector3Real(normals),
                ));
            }
        }

        // Interpolate attributes if requested
        if !attributes.is_empty() {
            profile!("interpolate attributes");
            info!("Post-processing: Interpolating attributes...");
            let interpolator = interpolator.as_ref().expect("interpolator is required");

            for attribute in attributes.into_iter() {
                info!("Interpolating attribute \"{}\"...", attribute.name);

                match attribute.data {
                    AttributeData::ScalarReal(values) => {
                        let interpolated_values = interpolator.interpolate_scalar_quantity(
                            values.as_slice(),
                            mesh_with_data.vertices(),
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
                            mesh_with_data.vertices(),
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
    }

    // Remove and clamp cells outside AABB
    let mesh_with_data = if let Some(mesh_aabb) = &postprocessing.mesh_aabb {
        profile!("clamp mesh to aabb");
        info!("Post-processing: Clamping mesh to AABB...");

        mesh_with_data.par_clamp_with_aabb(
            &mesh_aabb
                .try_convert()
                .ok_or_else(|| anyhow!("Failed to convert mesh AABB"))?,
            postprocessing.mesh_aabb_clamp_vertices,
            postprocessing.keep_vertices,
        )
    } else {
        mesh_with_data
    };

    // Convert triangles to quads
    let (mut tri_mesh, tri_quad_mesh) = if postprocessing.generate_quads {
        info!("Post-processing: Convert triangles to quads...");
        let non_squareness_limit = R::from_f64(postprocessing.quad_max_edge_diag_ratio).unwrap();
        let normal_angle_limit_rad =
            R::from_f64(postprocessing.quad_max_normal_angle.to_radians()).unwrap();
        let max_interior_angle =
            R::from_f64(postprocessing.quad_max_interior_angle.to_radians()).unwrap();

        let tri_quad_mesh = splashsurf_lib::postprocessing::convert_tris_to_quads(
            &mesh_with_data.mesh,
            non_squareness_limit,
            normal_angle_limit_rad,
            max_interior_angle,
        );

        let mut mesh_with_data = mesh_with_data;
        // Remove any cell attributes if present (they would have to be interpolated/merged)
        if !mesh_with_data.cell_attributes.is_empty() {
            warn!(
                "Post-processing: Cell attributes are not supported when converting triangle meshes to quad meshes. The following cell attribute(s) will be discarded: {}.",
                mesh_with_data
                    .cell_attributes
                    .iter()
                    .map(|a| a.name.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        mesh_with_data.cell_attributes.clear();

        (None, Some(mesh_with_data.with_mesh(tri_quad_mesh)))
    } else {
        (Some(mesh_with_data), None)
    };

    // TODO: Option to continue processing sequences even if checks fail. Maybe return special error type?

    if postprocessing.check_mesh_closed || postprocessing.check_mesh_manifold {
        if let Err(err) = match (&tri_mesh, &tri_quad_mesh) {
            (Some(mesh), None) => splashsurf_lib::marching_cubes::check_mesh_consistency(
                grid,
                &mesh.mesh,
                postprocessing.check_mesh_closed,
                postprocessing.check_mesh_manifold,
                postprocessing.check_mesh_debug,
            ),
            (None, Some(_mesh)) => {
                info!("Checking for mesh consistency not implemented for quad mesh at the moment.");
                return Ok(ReconstructionResult {
                    tri_mesh: None,
                    tri_quad_mesh: Some(_mesh.to_owned()),
                    raw_reconstruction: reconstruction_output,
                });
            }
            _ => unreachable!(),
        } {
            error!(
                "Checked mesh for problems (holes: {}, non-manifold edges/vertices: {}), problems were found!",
                postprocessing.check_mesh_closed, postprocessing.check_mesh_manifold
            );
            error!("{}", err);
            return Err(anyhow!("{}", err))
                .context(format!("Checked mesh for problems (holes: {}, non-manifold edges/vertices: {}), problems were found!", postprocessing.check_mesh_closed, postprocessing.check_mesh_manifold))
                .context("Problem found with mesh");
        } else {
            info!(
                "Checked mesh for problems (holes: {}, non-manifold edges/vertices: {}), no problems were found.",
                postprocessing.check_mesh_closed, postprocessing.check_mesh_manifold
            );
        }
    }

    if postprocessing.check_mesh_orientation {
        if let Err(err) = match (&tri_mesh, &tri_quad_mesh) {
            (Some(mesh), None) => {
                use splashsurf_lib::mesh::TriMesh3dExt;

                let tri_normals = mesh
                    .mesh
                    .triangles
                    .par_iter()
                    .map(|ijk| mesh.mesh.tri_normal_ijk::<R>(ijk))
                    .collect::<Vec<_>>();
                let vertex_face_map = mesh.vertex_cell_connectivity();
                let vertex_normals = mesh.mesh.par_vertex_normals();

                let mut flipped_faces = HashMap::new();
                for i in 0..vertex_normals.len() {
                    let n1 = vertex_normals[i];
                    for j in 0..vertex_face_map[i].len() {
                        let tri = vertex_face_map[i][j];
                        let n2 = tri_normals[tri];
                        let angle = n1.angle(&n2).to_f64().unwrap();
                        if angle > std::f64::consts::PI * 0.99 {
                            flipped_faces.insert(tri, (i, angle));
                        }
                    }
                }

                if !flipped_faces.is_empty() {
                    let mut error_strings = Vec::new();
                    error_strings.push(format!("Mesh is not consistently oriented. Found {} faces with normals flipped relative to adjacent vertices.", flipped_faces.len()));
                    if postprocessing.check_mesh_debug {
                        for (tri, (i, angle)) in flipped_faces.iter() {
                            error_strings.push(format!(
                                "\tAngle between normals of face {} and vertex {} is {:.2}°",
                                tri,
                                i,
                                angle.to_degrees()
                            ));
                        }
                    }
                    Err(anyhow!(error_strings.join("\n")))
                } else {
                    Ok(())
                }
            }
            (None, Some(_mesh)) => {
                info!(
                    "Checking for normal orientation not implemented for quad mesh at the moment."
                );
                Ok(())
            }
            _ => unreachable!(),
        } {
            error!("Checked mesh orientation (flipped normals), problems were found!");
            error!("{}", err);
            return Err(anyhow!("{}", err))
                .context("Checked mesh orientation (flipped normals), problems were found!")
                .context("Problem found with mesh");
        } else {
            info!("Checked mesh orientation (flipped normals), no problems were found.");
        }
    }

    match (&mut tri_mesh, &tri_quad_mesh) {
        (Some(mesh), None) => {
            let mut res: MeshWithData<R, TriMesh3d<R>> =
                MeshWithData::new(mesh.to_owned().mesh.into_owned());
            res.point_attributes = std::mem::take(&mut mesh.point_attributes);
            res.cell_attributes = std::mem::take(&mut mesh.cell_attributes);

            Ok(ReconstructionResult {
                tri_mesh: Some(res),
                tri_quad_mesh: None,
                raw_reconstruction: reconstruction_output,
            })
        }
        (None, Some(_mesh)) => Ok(ReconstructionResult {
            tri_mesh: None,
            tri_quad_mesh: Some(_mesh.to_owned()),
            raw_reconstruction: reconstruction_output,
        }),
        _ => unreachable!(),
    }
}

/// Wrapper for the reconstruction pipeline: loads input file, runs reconstructions, stores output files
pub(crate) fn reconstruction_pipeline_from_path<I: Index, R: Real>(
    paths: &ReconstructionRunnerPaths,
    params: &splashsurf_lib::Parameters<R>,
    io_params: &io::FormatParameters,
    postprocessing: &ReconstructionPostprocessingParameters,
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

    let ReconstructionResult {
        tri_mesh,
        tri_quad_mesh,
        raw_reconstruction: reconstruction,
    } = reconstruction_pipeline::<I, R>(&particle_positions, attributes, params, postprocessing)?;

    if postprocessing.output_raw_mesh {
        profile!("write surface mesh to file");

        let reconstruction = reconstruction.expect(
            "reconstruction_pipeline_from_data did not return a SurfaceReconstruction object",
        );
        let mesh = reconstruction.mesh();

        let output_path = paths
            .output_file
            .parent()
            // Add a trailing separator if the parent is non-empty
            .map(|p| p.join(""))
            .unwrap_or_default();
        let output_filename = format!(
            "raw_{}",
            paths.output_file.file_name().unwrap().to_string_lossy()
        );
        let raw_output_file = output_path.join(output_filename);

        info!(
            "Writing unprocessed surface mesh to \"{}\"...",
            raw_output_file.display()
        );

        io::write_mesh(
            &MeshWithData::new(mesh.to_owned()),
            raw_output_file,
            &io_params.output,
        )
        .with_context(|| {
            anyhow!(
                "Failed to write raw output mesh to file \"{}\"",
                paths.output_file.display()
            )
        })?;
    }

    // Store the surface mesh
    {
        profile!("write surface mesh to file");

        match (&tri_mesh, &tri_quad_mesh) {
            (Some(mesh), None) => {
                io::write_mesh(mesh, paths.output_file.clone(), &io_params.output)
            }
            (None, Some(mesh)) => {
                io::write_mesh(mesh, paths.output_file.clone(), &io_params.output)
            }

            _ => unreachable!(),
        }
        .with_context(|| {
            anyhow!(
                "Failed to write output mesh to file \"{}\"",
                paths.output_file.display()
            )
        })?;
    }

    Ok(())
}
