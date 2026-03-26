use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use tritpack_app_core::{
    bootstrap_failure_status, discover_worker_script, ensure_managed_python_env, AppPaths,
    DesktopApp, ResourcePaths, SqliteModelStore,
};
use tritpack_conversion::{PythonBridgeConfig, RustPythonConversionBackend};
use tritpack_runtime_llamacpp::LlamaCppBackend;

fn main() -> Result<()> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let paths = AppPaths::default_macos()?;
    let resources = ResourcePaths::discover(&repo_root)?;
    let worker_status = match ensure_managed_python_env(&paths, &resources, &repo_root) {
        Ok(status) => status,
        Err(error) => bootstrap_failure_status(&paths, &resources, &error),
    };
    let store = Arc::new(SqliteModelStore::open(&paths.state_db)?);
    let conversion = Arc::new(RustPythonConversionBackend::new(PythonBridgeConfig::new(
        worker_status.interpreter_path.clone(),
        discover_worker_script(&resources, &repo_root),
        &repo_root,
    )));
    let backend = Arc::new(LlamaCppBackend::new(
        Arc::clone(&conversion),
        Some(resources.bundled_llama_cli()),
    ));

    let app = Arc::new(DesktopApp::new(
        paths,
        resources,
        store,
        conversion,
        backend,
        worker_status,
    )?);
    tritpack_ui_slint::run(app)?;
    Ok(())
}
