fn main() {
    if std::env::var_os("CARGO_CFG_TARGET_OS=macos").is_some() {
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,{}",
            pyo3_build_config::get()
                .lib_dir
                .clone()
                .expect("Python lib dir not found")
        );
    }
}
