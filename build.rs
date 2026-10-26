//! Embed source identity without requiring Git or adding runtime file access.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

fn source_files(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read native source directory") {
        let path = entry.expect("read native source entry").path();
        if path.is_dir() {
            source_files(&path, files);
        } else if path.is_file() {
            files.push(path);
        }
    }
}

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let mut files: Vec<_> = ["Cargo.toml", "Cargo.lock", "build.rs"]
        .iter()
        .map(|name| root.join(name))
        .collect();
    source_files(&root.join("src"), &mut files);
    let mut entries: Vec<_> = files
        .into_iter()
        .map(|path| {
            let name = path
                .strip_prefix(&root)
                .unwrap()
                .to_str()
                .unwrap()
                .replace('\\', "/");
            (name, path)
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    // Keep this framing in sync with tools/check_native_build.py. Lengths make
    // renamed files and ambiguous concatenations produce distinct identities.
    let mut digest = Sha256::new();
    digest.update(b"fastscipy-native-source-v1\0");
    println!("cargo:rerun-if-changed=src"); // Also detect added/deleted files.
    for (name, path) in entries {
        println!("cargo:rerun-if-changed={name}");
        let content = fs::read(path).expect("read native build input");
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update((content.len() as u64).to_le_bytes());
        digest.update(content);
    }
    println!(
        "cargo:rustc-env=FASTSCIPY_SOURCE_SHA256={:x}",
        digest.finalize()
    );
    for (key, variable) in [
        ("TARGET", "TARGET"),
        ("PROFILE", "PROFILE"),
        ("OPT_LEVEL", "OPT_LEVEL"),
    ] {
        println!(
            "cargo:rustc-env=FASTSCIPY_BUILD_{key}={}",
            env::var(variable).unwrap()
        );
    }
    let rustc = Command::new(env::var_os("RUSTC").unwrap())
        .arg("--version")
        .output()
        .expect("read Rust compiler version");
    assert!(rustc.status.success(), "rustc --version failed");
    println!(
        "cargo:rustc-env=FASTSCIPY_BUILD_RUSTC={}",
        String::from_utf8(rustc.stdout).unwrap().trim()
    );
}
