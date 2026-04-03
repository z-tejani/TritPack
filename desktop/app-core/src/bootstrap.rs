use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::AppPaths;

#[derive(Debug, Clone)]
pub struct ResourcePaths {
    pub root: PathBuf,
    pub llama_manifest: PathBuf,
    pub llama_dir: PathBuf,
    pub python_dir: PathBuf,
    pub python_requirements: PathBuf,
    pub python_runtime_manifest: PathBuf,
    pub python_worker_script: PathBuf,
    pub python_wheelhouse: PathBuf,
    pub info_plist_template: PathBuf,
}

impl ResourcePaths {
    pub fn discover(repo_root: &Path) -> Result<Self> {
        let bundled_root = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(Path::to_path_buf))
            .and_then(|macos_dir| {
                macos_dir
                    .parent()
                    .map(|contents| contents.join("Resources"))
            })
            .filter(|resources| resources.exists());
        let root = bundled_root.unwrap_or_else(|| repo_root.join("resources"));
        Ok(Self {
            llama_manifest: root.join("llama.cpp/manifest.toml"),
            llama_dir: root.join("llama.cpp"),
            python_dir: root.join("python"),
            python_requirements: root.join("python/requirements.lock"),
            python_runtime_manifest: root.join("python/runtime-manifest.toml"),
            python_worker_script: root.join("python/worker/desktop_worker.py"),
            python_wheelhouse: root.join("python/wheelhouse"),
            info_plist_template: repo_root.join("packaging/macos/Info.plist"),
            root,
        })
    }

    pub fn bundled_llama_cli(&self) -> PathBuf {
        self.llama_dir.join("llama-cli")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonRuntimeManifest {
    pub version: String,
    pub source: String,
    pub archive_name: String,
    pub archive_url: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapMarker {
    pub python_version: String,
    pub lock_sha256: String,
    pub installed_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct WorkerEnvironmentStatus {
    pub ok: bool,
    pub message: String,
    pub interpreter_path: PathBuf,
    pub venv_path: PathBuf,
    pub marker_path: PathBuf,
    pub lock_sha256: String,
    pub python_version: String,
}

pub fn load_python_runtime_manifest(resources: &ResourcePaths) -> Result<PythonRuntimeManifest> {
    let raw = fs::read_to_string(&resources.python_runtime_manifest).with_context(|| {
        format!(
            "failed to read runtime manifest {}",
            resources.python_runtime_manifest.display()
        )
    })?;
    Ok(toml::from_str(&raw)?)
}

pub fn discover_worker_script(resources: &ResourcePaths, repo_root: &Path) -> PathBuf {
    if resources.python_worker_script.exists() {
        resources.python_worker_script.clone()
    } else {
        repo_root.join("python/tritpack/desktop_worker.py")
    }
}

pub fn ensure_managed_python_env(
    paths: &AppPaths,
    resources: &ResourcePaths,
    repo_root: &Path,
) -> Result<WorkerEnvironmentStatus> {
    paths.ensure()?;
    fs::create_dir_all(&paths.python_root)?;
    sync_managed_cpython(paths, resources)?;

    let requirements = fs::read(&resources.python_requirements).with_context(|| {
        format!(
            "failed to read worker requirements {}",
            resources.python_requirements.display()
        )
    })?;
    let mut hasher = Sha256::new();
    hasher.update(&requirements);
    let lock_sha256 = format!("{:x}", hasher.finalize());

    let runtime_manifest = load_python_runtime_manifest(resources)?;
    let bootstrap_python = resolve_bootstrap_python(paths, resources);
    let venv_python = paths.managed_venv.join("bin/python3");
    let marker_path = paths.python_marker.clone();
    let worker_script = discover_worker_script(resources, repo_root);
    let offline_wheelhouse = has_offline_wheelhouse(resources);

    let is_current = match fs::read_to_string(&marker_path) {
        Ok(raw) => {
            let marker: BootstrapMarker = serde_json::from_str(&raw)?;
            marker.python_version == runtime_manifest.version
                && marker.lock_sha256 == lock_sha256
                && venv_python.exists()
                && worker_script.exists()
        }
        Err(_) => false,
    };

    if !is_current {
        if let Some(parent) = marker_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir_all(&paths.managed_cpython)?;
        run_command(
            &bootstrap_python,
            &["-m", "venv", paths.managed_venv.to_string_lossy().as_ref()],
            "create managed virtualenv",
        )?;
        install_worker_requirements(&venv_python, resources, offline_wheelhouse)?;
        let marker = BootstrapMarker {
            python_version: runtime_manifest.version.clone(),
            lock_sha256: lock_sha256.clone(),
            installed_at: Utc::now(),
        };
        fs::write(&marker_path, serde_json::to_vec_pretty(&marker)?)?;
    }

    validate_python_modules(&venv_python)?;
    validate_worker_protocol(&venv_python, &worker_script)?;

    let python_version =
        python_version(&venv_python).unwrap_or_else(|_| runtime_manifest.version.clone());

    Ok(WorkerEnvironmentStatus {
        ok: venv_python.exists() && worker_script.exists(),
        message: format!(
            "Managed worker env ready | python={} | worker={} | lock={} | mode={}",
            venv_python.display(),
            worker_script.display(),
            &lock_sha256[..12],
            if offline_wheelhouse {
                "offline-wheelhouse"
            } else {
                "pypi-fallback"
            }
        ),
        interpreter_path: venv_python,
        venv_path: paths.managed_venv.clone(),
        marker_path,
        lock_sha256,
        python_version,
    })
}

pub fn bootstrap_failure_status(
    paths: &AppPaths,
    resources: &ResourcePaths,
    error: &anyhow::Error,
) -> WorkerEnvironmentStatus {
    WorkerEnvironmentStatus {
        ok: false,
        message: format!(
            "Managed worker bootstrap failed: {} | requirements={} | runtime_manifest={}",
            error,
            resources.python_requirements.display(),
            resources.python_runtime_manifest.display()
        ),
        interpreter_path: paths.managed_venv.join("bin/python3"),
        venv_path: paths.managed_venv.clone(),
        marker_path: paths.python_marker.clone(),
        lock_sha256: String::new(),
        python_version: String::new(),
    }
}

fn resolve_bootstrap_python(paths: &AppPaths, resources: &ResourcePaths) -> PathBuf {
    if paths.managed_cpython.join("bin/python3").exists() {
        paths.managed_cpython.join("bin/python3")
    } else if resources.python_dir.join("cpython/bin/python3").exists() {
        resources.python_dir.join("cpython/bin/python3")
    } else if let Some(value) = std::env::var_os("TRITPACK_BOOTSTRAP_PYTHON") {
        PathBuf::from(value)
    } else {
        PathBuf::from("python3")
    }
}

fn run_command(program: &Path, args: &[&str], context: &str) -> Result<()> {
    let status = Command::new(program)
        .args(args)
        .status()
        .with_context(|| format!("failed to {}", context))?;
    if !status.success() {
        return Err(anyhow!(
            "command failed to {} with status {}",
            context,
            status
        ));
    }
    Ok(())
}

fn python_version(program: &Path) -> Result<String> {
    let output = Command::new(program).arg("--version").output()?;
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !stdout.is_empty() {
            return Ok(stdout);
        }
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if !stderr.is_empty() {
            return Ok(stderr);
        }
    }
    Err(anyhow!(
        "failed to determine python version for {}",
        program.display()
    ))
}

fn sync_managed_cpython(paths: &AppPaths, resources: &ResourcePaths) -> Result<()> {
    let vendored_root = resources.python_dir.join("cpython");
    let managed_python = paths.managed_cpython.join("bin/python3");
    let vendored_python = vendored_root.join("bin/python3");
    if !vendored_python.exists() || managed_python.exists() {
        return Ok(());
    }

    if paths.managed_cpython.exists() {
        fs::remove_dir_all(&paths.managed_cpython)?;
    }
    copy_dir_all(&vendored_root, &paths.managed_cpython)?;
    Ok(())
}

fn install_worker_requirements(
    venv_python: &Path,
    resources: &ResourcePaths,
    offline_wheelhouse: bool,
) -> Result<()> {
    if offline_wheelhouse {
        run_command(
            venv_python,
            &[
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                resources.python_wheelhouse.to_string_lossy().as_ref(),
                "-r",
                resources.python_requirements.to_string_lossy().as_ref(),
            ],
            "install TritPack worker requirements from the bundled wheelhouse",
        )
    } else {
        run_command(
            venv_python,
            &[
                "-m",
                "pip",
                "install",
                "-r",
                resources.python_requirements.to_string_lossy().as_ref(),
            ],
            "install TritPack worker requirements from PyPI",
        )
    }
}

fn copy_dir_all(source: &Path, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_all(&source_path, &dest_path)?;
        } else {
            if let Some(parent) = dest_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&source_path, &dest_path)?;
        }
    }
    Ok(())
}

fn has_offline_wheelhouse(resources: &ResourcePaths) -> bool {
    resources.python_wheelhouse.exists()
        && fs::read_dir(&resources.python_wheelhouse)
            .map(|entries| {
                entries.flatten().any(|entry| {
                    entry
                        .path()
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| matches!(ext, "whl" | "zip"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
}

fn validate_python_modules(program: &Path) -> Result<()> {
    run_command(
        program,
        &["-c", "import numpy, gguf; print('ok')"],
        "validate managed worker Python modules",
    )
}

fn validate_worker_protocol(program: &Path, worker_script: &Path) -> Result<()> {
    let mut child = Command::new(program)
        .arg(worker_script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to launch worker {}", worker_script.display()))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(
            br#"{"request_id":"bootstrap-health","command":"shutdown","payload":{}}
"#,
        )?;
        stdin.flush()?;
    }

    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(anyhow!(
            "worker protocol check failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains("\"status\": \"ok\"") && !stdout.contains("\"status\":\"ok\"") {
        return Err(anyhow!(
            "worker protocol check did not return an ok response: {}",
            stdout.trim()
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn resource_discovery_prefers_repo_root_when_not_packaged() -> Result<()> {
        let root = tempdir()?;
        fs::create_dir_all(root.path().join("resources/python"))?;
        let resources = ResourcePaths::discover(root.path())?;
        assert_eq!(resources.root, root.path().join("resources"));
        Ok(())
    }

    #[test]
    fn parses_python_runtime_manifest() -> Result<()> {
        let root = tempdir()?;
        fs::create_dir_all(root.path().join("resources/python"))?;
        fs::write(
            root.path().join("resources/python/runtime-manifest.toml"),
            r#"
version = "3.11.15"
source = "astral-sh/python-build-standalone"
archive_name = "cpython.tar.gz"
archive_url = "https://example.com/cpython.tar.gz"
sha256 = "abc123"
"#,
        )?;
        let resources = ResourcePaths::discover(root.path())?;
        let manifest = load_python_runtime_manifest(&resources)?;
        assert_eq!(manifest.version, "3.11.15");
        assert_eq!(manifest.sha256, "abc123");
        Ok(())
    }

    #[test]
    fn detects_offline_wheelhouse() -> Result<()> {
        let root = tempdir()?;
        fs::create_dir_all(root.path().join("resources/python/wheelhouse"))?;
        let resources = ResourcePaths::discover(root.path())?;
        assert!(!has_offline_wheelhouse(&resources));

        fs::write(
            root.path()
                .join("resources/python/wheelhouse/numpy-1.0.0-py3-none-any.whl"),
            b"placeholder",
        )?;
        assert!(has_offline_wheelhouse(&resources));
        Ok(())
    }
}
