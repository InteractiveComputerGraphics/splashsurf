// Run `cargo run --bin stub_gen` to generate a stub file for the extension
use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    // `stub_info` is a function defined by `define_stub_info_gatherer!` macro.
    let stub = pysplashsurf::stub_info()?;
    stub.generate()?;
    std::fs::rename("pysplashsurf.pyi", "pysplashsurf/pysplashsurf.pyi")?;
    Ok(())
}
