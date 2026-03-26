use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::thread;

use anyhow::{anyhow, Context, Result};
use parking_lot::Mutex;
use tritpack_conversion::RustPythonConversionBackend;
use tritpack_runtime_api::{
    BackendCapabilities, BackendHealth, GenerateRequest, GenerationStream, InferenceBackend,
    InferenceEvent, LoadModelRequest, ModelHandle,
};
use uuid::Uuid;

pub struct LlamaCppBackend {
    conversion: Arc<RustPythonConversionBackend>,
    bundled_binary: Option<PathBuf>,
    active_processes: Arc<Mutex<HashMap<String, Arc<Mutex<Child>>>>>,
    loaded_models: Arc<Mutex<HashMap<String, ModelHandle>>>,
}

impl LlamaCppBackend {
    pub fn new(
        conversion: Arc<RustPythonConversionBackend>,
        bundled_binary: Option<PathBuf>,
    ) -> Self {
        Self {
            conversion,
            bundled_binary,
            active_processes: Arc::new(Mutex::new(HashMap::new())),
            loaded_models: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn candidate_binaries(&self, request_binary: Option<PathBuf>) -> Vec<PathBuf> {
        let mut candidates = Vec::new();
        if let Some(path) = request_binary {
            candidates.push(path);
        }
        if let Some(path) = self.bundled_binary.clone() {
            candidates.push(path);
        }
        if let Some(path) = std::env::var_os("TRITPACK_LLAMA_CPP") {
            candidates.push(PathBuf::from(path));
        }
        if let Some(paths) = std::env::var_os("PATH") {
            for dir in std::env::split_paths(&paths) {
                candidates.push(dir.join("llama-cli"));
                candidates.push(dir.join("llama-run"));
            }
        }
        candidates
    }

    fn resolve_binary(&self, request_binary: Option<PathBuf>) -> Result<PathBuf> {
        for candidate in self.candidate_binaries(request_binary) {
            if candidate.exists() {
                return Ok(candidate);
            }
        }
        Err(anyhow!(
            "No llama.cpp binary found. Set TRITPACK_LLAMA_CPP or bundle llama-cli into the app resources."
        ))
    }

    fn binary_diagnostics(&self, request_binary: Option<PathBuf>) -> String {
        let candidates = self.candidate_binaries(request_binary);
        if candidates.is_empty() {
            return "No candidate binary paths were generated".to_string();
        }

        candidates
            .into_iter()
            .map(|candidate| {
                if !candidate.exists() {
                    return format!("{} (missing)", candidate.display());
                }

                match Command::new(&candidate).arg("--version").output() {
                    Ok(output) if output.status.success() => {
                        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
                        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                        let version = if !stdout.is_empty() { stdout } else { stderr };
                        if version.is_empty() {
                            format!("{} (available)", candidate.display())
                        } else {
                            format!("{} ({version})", candidate.display())
                        }
                    }
                    Ok(output) => format!(
                        "{} (found but --version failed with status {})",
                        candidate.display(),
                        output.status
                    ),
                    Err(error) => format!(
                        "{} (found but could not execute: {error})",
                        candidate.display()
                    ),
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn ensure_cached_gguf(&self, request: &LoadModelRequest) -> Result<PathBuf> {
        fs::create_dir_all(&request.cache_dir)?;
        let cache_path = request.cache_dir.join(format!(
            "{}-{}.gguf",
            request.model_id,
            &request.original_hash[..12]
        ));
        let hash_marker = cache_path.with_extension("gguf.source-hash");
        let cache_is_fresh = cache_path.exists()
            && hash_marker.exists()
            && fs::read_to_string(&hash_marker).unwrap_or_default() == request.original_hash;

        if !cache_is_fresh {
            self.conversion
                .reconstruct_gguf(&request.tritpack_dir, &cache_path)
                .with_context(|| format!("failed to reconstruct {}", cache_path.display()))?;
            fs::write(&hash_marker, &request.original_hash)?;
        }

        Ok(cache_path)
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn name(&self) -> &'static str {
        "llama.cpp"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_streaming: true,
            supports_interrupt: true,
            supports_embeddings: false,
        }
    }

    fn health_check(&self) -> Result<BackendHealth> {
        match self.resolve_binary(None) {
            Ok(path) => Ok(BackendHealth {
                ok: true,
                message: format!(
                    "Using {} | {}",
                    path.display(),
                    self.binary_diagnostics(Some(path.clone()))
                ),
            }),
            Err(error) => Ok(BackendHealth {
                ok: false,
                message: format!("{} Tried: {}", error, self.binary_diagnostics(None)),
            }),
        }
    }

    fn prepare_model(&self, request: LoadModelRequest) -> Result<ModelHandle> {
        let prepared_path = self.ensure_cached_gguf(&request)?;
        let handle = ModelHandle {
            id: Uuid::new_v4().to_string(),
            model_id: request.model_id.clone(),
            backend_name: self.name().to_string(),
            prepared_path,
        };
        self.loaded_models
            .lock()
            .insert(handle.model_id.clone(), handle.clone());
        Ok(handle)
    }

    fn load_model(&self, request: LoadModelRequest) -> Result<ModelHandle> {
        self.prepare_model(request)
    }

    fn unload_model(&self, handle: &ModelHandle) -> Result<()> {
        self.loaded_models.lock().remove(&handle.model_id);
        Ok(())
    }

    fn generate_stream(&self, request: GenerateRequest) -> Result<GenerationStream> {
        let binary = self.resolve_binary(None)?;
        let (tx, rx) = channel();
        let mut command = Command::new(binary);
        command
            .arg("-m")
            .arg(&request.model.prepared_path)
            .arg("-p")
            .arg(&request.prompt)
            .arg("-n")
            .arg(request.runtime_profile.max_tokens.to_string())
            .arg("-c")
            .arg(request.runtime_profile.context_size.to_string())
            .arg("--temp")
            .arg(request.runtime_profile.temperature.to_string())
            .arg("--top-p")
            .arg(request.runtime_profile.top_p.to_string())
            .arg("-t")
            .arg(request.runtime_profile.threads.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if request.runtime_profile.gpu_layers >= 0 {
            command
                .arg("--n-gpu-layers")
                .arg(request.runtime_profile.gpu_layers.to_string());
        }

        let child = command
            .spawn()
            .context("failed to launch llama.cpp subprocess")?;
        let generation_id = request.generation_id.clone();
        let child = Arc::new(Mutex::new(child));
        self.active_processes
            .lock()
            .insert(generation_id.clone(), Arc::clone(&child));

        let active = self.active_processes.clone();
        thread::spawn(move || {
            let _ = tx.send(InferenceEvent::Started {
                generation_id: generation_id.clone(),
            });

            let mut stdout = match child.lock().stdout.take() {
                Some(stdout) => stdout,
                None => {
                    let _ = tx.send(InferenceEvent::Error(
                        "llama.cpp stdout was not captured".to_string(),
                    ));
                    active.lock().remove(&generation_id);
                    return;
                }
            };

            let mut buffer = [0_u8; 1024];
            let mut output = String::new();
            loop {
                match stdout.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(read) => {
                        let chunk = String::from_utf8_lossy(&buffer[..read]).to_string();
                        output.push_str(&chunk);
                        let _ = tx.send(InferenceEvent::Token(chunk));
                    }
                    Err(error) => {
                        let _ = tx.send(InferenceEvent::Error(error.to_string()));
                        break;
                    }
                }
            }

            let status = child.lock().wait();
            match status {
                Ok(code) if code.success() => {
                    let _ = tx.send(InferenceEvent::Completed { output });
                }
                Ok(code) => {
                    let _ = tx.send(InferenceEvent::Error(format!(
                        "llama.cpp exited with status {code}"
                    )));
                }
                Err(error) => {
                    let _ = tx.send(InferenceEvent::Error(error.to_string()));
                }
            }
            active.lock().remove(&generation_id);
        });

        Ok(GenerationStream { receiver: rx })
    }

    fn interrupt(&self, generation_id: &str) -> Result<()> {
        if let Some(child) = self.active_processes.lock().remove(generation_id) {
            child.lock().kill()?;
        }
        Ok(())
    }
}
