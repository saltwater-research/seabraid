// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Gives the cdylib a versioned SONAME (Linux) or install name and compatibility version (macOS).
//!
//! WHY a SONAME at all: without one the linker records the consumer's DT_NEEDED as the path the library was
//! linked from (a relative build path), so the consumer only starts from that working directory.
//! WHY `libseabraid.so.MAJOR.MINOR` and not `libseabraid.so.MAJOR`: while the major version is 0 any minor bump
//! may break the ABI, so a consumer built against 0.5 must refuse to load 0.6 instead of misreading memory.
//! The loader enforces that for free once the name carries the minor. After 1.0 this becomes `.so.MAJOR`.
//! WHY derived from CARGO_PKG_VERSION_*: the soname cannot drift from the release number, and CMake derives the
//! same string from the same VERSION file; the SONAME CTest asserts they agree.
//! Cargo emits only `libseabraid.so`; CMake (build tree) and the install rules supply the versioned file name
//! and the unversioned dev symlink. Windows keeps `seabraid.dll`: the DLL name carries no ABI version, so
//! consumers there must check seabraid_abi_epoch() and bundle the DLL they built against.
//! The crate stays dependency-free: this is plain `println!` directives, nothing else.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let version =
        |part: &str| std::env::var(format!("CARGO_PKG_VERSION_{part}")).unwrap_or_default();
    let (major, minor, patch) = (version("MAJOR"), version("MINOR"), version("PATCH"));
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libseabraid.so.{major}.{minor}");
        }
        Ok("macos") => {
            // The file keeps its unversioned name, so the install name does too; the compatibility version is
            // what dyld compares, and it moves with the minor for the same 0.x reason as the SONAME.
            // LIMIT, stated so nobody reads this as the Linux policy: dyld refuses a library whose compatibility
            // version is LOWER than the one a client recorded, never a higher one. A client linked against 0.5
            // therefore refuses a 0.4 library but still loads a 0.6 one, where Linux refuses it (the SONAME
            // names the minor). A name-based match would need a versioned file name and install name; that
            // layout cannot be built or tested without a Mac, so macOS clients rely on the start-up
            // seabraid_abi_epoch() check, which docs/ABI.md requires of every consumer.
            println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/libseabraid.dylib");
            println!("cargo:rustc-cdylib-link-arg=-Wl,-compatibility_version,{major}.{minor}.0");
            println!("cargo:rustc-cdylib-link-arg=-Wl,-current_version,{major}.{minor}.{patch}");
        }
        _ => {}
    }
}
