fn main() -> Result<(), anyhow::Error> {
    /*
    // Panic hook for easier debugging
    panic::set_hook(Box::new(|panic_info| {
        println!("Panic occurred: {}", panic_info);
        println!("Add breakpoint here for debugging.");
    }));
    */

    std::process::exit(
        match splashsurf::cli::run_splashsurf(std::env::args_os().collect::<Vec<_>>().as_slice()) {
            Ok(_) => 0,
            Err(_) => 1,
        },
    );
}
