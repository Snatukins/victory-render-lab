fn main() {
    println!("cargo:rerun-if-changed=src/writeback.c");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        cc::Build::new()
            .file("src/writeback.c")
            .opt_level(2)
            .compile("victory_gpu_writeback");
    }
}
