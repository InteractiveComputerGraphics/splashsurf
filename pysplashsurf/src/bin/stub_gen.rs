// Run `cargo run --bin stub_gen` to generate a stub file for the extension

use pyo3_stub_gen::Result;
use std::fs;

fn add_future_imports() -> Result<()> {
    // Read the original file
    let content = fs::read_to_string("pysplashsurf.pyi")?;
    let lines: Vec<&str> = content.lines().collect();

    // Find where to insert the import (after comments)
    let mut insert_index = 0;
    for (i, line) in lines.iter().enumerate() {
        if !line.trim().starts_with('#') && !line.trim().is_empty() {
            insert_index = i;
            break;
        }
    }

    // Create new content with the import added
    let mut new_lines = Vec::new();
    new_lines.extend_from_slice(&lines[..insert_index]);
    new_lines.push("from __future__ import annotations");
    if insert_index < lines.len() && !lines[insert_index].is_empty() {
        new_lines.push("");
    }
    new_lines.extend_from_slice(&lines[insert_index..]);

    // Write the modified content back to the file
    fs::write("pysplashsurf.pyi", new_lines.join("\n"))?;

    Ok(())
}

fn main() -> Result<()> {
    // `stub_info` is a function defined by `define_stub_info_gatherer!` macro.
    let stub = pysplashsurf::stub_info()?;
    stub.generate()?;

    add_future_imports()?;

    fs::rename("pysplashsurf.pyi", "pysplashsurf/pysplashsurf.pyi")?;
    Ok(())
}
