use crate::reconstruction::Switch;
use crate::Subcommand;
use std::path::PathBuf;

#[test]
fn verify_main_cli() {
    use clap::CommandFactory;
    crate::CommandlineArgs::command().debug_assert()
}

#[test]
fn verify_reconstruct_cli() {
    use clap::CommandFactory;
    crate::reconstruction::ReconstructSubcommandArgs::command().debug_assert()
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
        crate::CommandlineArgs::try_parse_from(["splashsurf", "--help",])
            .expect_err("this command is supposed to fail")
            .kind(),
        clap::error::ErrorKind::DisplayHelp
    );

    // Display help, reconstruct
    assert_eq!(
        crate::CommandlineArgs::try_parse_from(["splashsurf", "reconstruct", "--help",])
            .expect_err("this command is supposed to fail")
            .kind(),
        clap::error::ErrorKind::DisplayHelp
    );

    // Display help, convert
    assert_eq!(
        crate::CommandlineArgs::try_parse_from(["splashsurf", "convert", "--help",])
            .expect_err("this command is supposed to fail")
            .kind(),
        clap::error::ErrorKind::DisplayHelp
    );

    // Minimum arguments: input file
    if let Subcommand::Reconstruct(rec_args) = crate::CommandlineArgs::try_parse_from([
        "splashsurf",
        "reconstruct",
        "--input-file",
        "test.vtk",
        "--particle-radius=0.05",
        "--smoothing-length=3.0",
        "--cube-size=0.75",
    ])
    .expect("this command is supposed to work")
    .subcommand
    {
        assert_eq!(rec_args.input_file, Some(PathBuf::from("test.vtk")));
    };

    // Minimum arguments: input sequence
    if let Subcommand::Reconstruct(rec_args) = crate::CommandlineArgs::try_parse_from([
        "splashsurf",
        "reconstruct",
        "--input-sequence",
        "test.vtk",
        "--particle-radius=0.05",
        "--smoothing-length=3.0",
        "--cube-size=0.75",
    ])
    .expect("this command is supposed to work")
    .subcommand
    {
        assert_eq!(rec_args.input_sequence, Some(PathBuf::from("test.vtk")));
    };

    // Input file & sequence conflict
    assert_eq!(
        crate::CommandlineArgs::try_parse_from([
            "splashsurf",
            "reconstruct",
            "--input-file",
            "test.vtk",
            "--input-sequence",
            "test.vtk",
            "--particle-radius=0.05",
            "--smoothing-length=3.0",
            "--cube-size=0.75"
        ])
        .expect_err("this command is supposed to fail")
        .kind(),
        clap::error::ErrorKind::ArgumentConflict
    );

    // Test on/off switch
    if let Subcommand::Reconstruct(rec_args) = crate::CommandlineArgs::try_parse_from([
        "splashsurf",
        "reconstruct",
        "--input-file",
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

    if let Subcommand::Reconstruct(rec_args) = crate::CommandlineArgs::try_parse_from([
        "splashsurf",
        "reconstruct",
        "--input-file",
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

    assert_eq!(
        crate::CommandlineArgs::try_parse_from([
            "splashsurf",
            "reconstruct",
            "--input-file",
            "test.vtk",
            "--input-sequence",
            "test.vtk",
            "--particle-radius=0.05",
            "--smoothing-length=3.0",
            "--cube-size=0.75",
            "--normals"
        ])
        .expect_err("this command is supposed to fail")
        .kind(),
        clap::error::ErrorKind::NoEquals
    );

    // Test domain min/max: correct values
    if let Subcommand::Reconstruct(rec_args) = crate::CommandlineArgs::try_parse_from([
        "splashsurf",
        "reconstruct",
        "--input-file",
        "test.vtk",
        "--particle-radius=0.05",
        "--smoothing-length=3.0",
        "--cube-size=0.75",
        "--domain-min",
        "-1.0",
        "1.0",
        "-1.0",
        "--domain-max",
        "-2.0",
        "2.0",
        "-2.0",
    ])
    .expect("this command is supposed to work")
    .subcommand
    {
        assert_eq!(rec_args.domain_min, Some(vec![-1.0, 1.0, -1.0]));
        assert_eq!(rec_args.domain_max, Some(vec![-2.0, 2.0, -2.0]));
    };

    // Test domain min/max: too many values
    assert_eq!(
        crate::CommandlineArgs::try_parse_from([
            "splashsurf",
            "reconstruct",
            "--input-file",
            "test.vtk",
            "--input-sequence",
            "test.vtk",
            "--particle-radius=0.05",
            "--smoothing-length=3.0",
            "--cube-size=0.75",
            "--domain-min",
            "-1.0",
            "1.0",
            "-1.0",
            "2.0",
            "--domain-max",
            "-2.0",
            "2.0",
            "-2.0",
        ])
            .expect_err("this command is supposed to fail")
            .kind(),
        clap::error::ErrorKind::UnknownArgument
    );
}
