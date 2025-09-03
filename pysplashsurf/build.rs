fn main() {
    // Required to run cargo check or stub gen outside maturin build
    if std::env::var_os("CARGO_CFG_TARGET_OS=macos").is_some() {
        pyo3_build_config::get()
            .lib_dir
            .clone()
            .map(|lib_dir| println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir));
    }
}
