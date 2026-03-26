use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::thread;

use anyhow::{anyhow, bail, Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tritpack_runtime_api::{
    ConversionBackend, ConversionEstimate, ConversionEvent, ConversionProfile, ConversionStream,
    ConvertRequest, GgufInspection, VerifyReport, VerifyRequest,
};
use uuid::Uuid;

pub use tritpack_native as native;

#[derive(Debug, Clone)]
pub struct PythonBridgeConfig {
    pub python_bin: PathBuf,
    pub worker_script: PathBuf,
    pub project_root: PathBuf,
}

impl PythonBridgeConfig {
    pub fn from_repo_root(root: impl Into<PathBuf>) -> Self {
        let project_root = root.into();
        Self {
            python_bin: PathBuf::from("python3"),
            worker_script: project_root.join("python/tritpack/desktop_worker.py"),
            project_root,
        }
    }
}

impl PythonBridgeConfig {
    pub fn new(
        python_bin: impl Into<PathBuf>,
        worker_script: impl Into<PathBuf>,
        project_root: impl Into<PathBuf>,
    ) -> Self {
        Self {
            python_bin: python_bin.into(),
            worker_script: worker_script.into(),
            project_root: project_root.into(),
        }
    }
}

pub struct RustPythonConversionBackend {
    config: PythonBridgeConfig,
    worker: Arc<Mutex<Option<WorkerProcess>>>,
    active_jobs: Arc<Mutex<std::collections::HashMap<String, Arc<Mutex<WorkerProcess>>>>>,
}

impl RustPythonConversionBackend {
    pub fn new(config: PythonBridgeConfig) -> Self {
        Self {
            config,
            worker: Arc::new(Mutex::new(None)),
            active_jobs: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    fn worker(&self) -> Result<WorkerClient> {
        let mut guard = self.worker.lock();
        if guard.is_none() {
            *guard = Some(WorkerProcess::spawn(&self.config)?);
        }
        Ok(WorkerClient {
            inner: Arc::clone(&self.worker),
        })
    }
}

impl ConversionBackend for RustPythonConversionBackend {
    fn inspect_gguf(&self, path: &Path) -> Result<GgufInspection> {
        inspect_gguf_header(path)
    }

    fn estimate_conversion(
        &self,
        inspection: &GgufInspection,
        profile: &ConversionProfile,
    ) -> Result<ConversionEstimate> {
        let ratio = if profile.alpha <= 0.55 {
            4.8
        } else if profile.alpha <= 0.7 {
            5.6
        } else {
            6.0
        };
        Ok(ConversionEstimate {
            estimated_bytes: (inspection.size_bytes as f64 / ratio) as u64,
            compression_ratio: ratio,
        })
    }

    fn convert_to_tritpack(&self, request: ConvertRequest) -> Result<ConversionStream> {
        self.spawn_job_stream(
            request.job_id,
            "convert",
            json!({
                "source_path": request.source_path,
                "output_dir": request.output_dir,
                "alpha": request.profile.alpha,
                "block_size": request.profile.block_size,
                "sample_tensors": request.sample_tensors,
            }),
        )
    }

    fn reconstruct_gguf_stream(
        &self,
        job_id: String,
        tritpack_dir: PathBuf,
        output_path: PathBuf,
    ) -> Result<ConversionStream> {
        self.spawn_job_stream(
            job_id,
            "reconstruct",
            json!({
                "tritpack_dir": tritpack_dir,
                "output_path": output_path,
            }),
        )
    }

    fn verify_conversion(&self, request: VerifyRequest) -> Result<VerifyReport> {
        let worker = self.worker()?;
        let payload = worker.request(
            "verify",
            json!({
                "source_path": request.source_path,
                "tritpack_dir": request.tritpack_dir,
                "sample_tensors": request.sample_tensors,
            }),
        )?;
        Ok(serde_json::from_value(payload)?)
    }

    fn cancel_job(&self, job_id: &str) -> Result<()> {
        if let Some(worker) = self.active_jobs.lock().remove(job_id) {
            worker.lock().kill()?;
        }
        Ok(())
    }
}

impl RustPythonConversionBackend {
    pub fn reconstruct_gguf(&self, tritpack_dir: &Path, output_path: &Path) -> Result<()> {
        let worker = self.worker()?;
        worker.request(
            "reconstruct",
            json!({
                "tritpack_dir": tritpack_dir,
                "output_path": output_path,
            }),
        )?;
        Ok(())
    }

    fn spawn_job_stream(
        &self,
        job_id: String,
        command: &str,
        payload: Value,
    ) -> Result<ConversionStream> {
        let (tx, rx) = channel();
        let worker = Arc::new(Mutex::new(WorkerProcess::spawn(&self.config)?));
        self.active_jobs
            .lock()
            .insert(job_id.clone(), Arc::clone(&worker));
        let active_jobs = Arc::clone(&self.active_jobs);
        let error_tx = tx.clone();
        let command = command.to_string();

        thread::spawn(move || {
            let result = stream_dedicated_command(&worker, &command, &payload, &job_id, tx);
            active_jobs.lock().remove(&job_id);
            if let Err(error) = result {
                let _ = error_tx.send(ConversionEvent::Error {
                    job_id,
                    message: error.to_string(),
                });
            }
        });

        Ok(ConversionStream { receiver: rx })
    }
}

pub fn inspect_gguf_header(path: &Path) -> Result<GgufInspection> {
    let mut file = File::open(path)?;
    let size_bytes = file.metadata()?.len();
    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("{} is not a GGUF file", path.display());
    }

    let version = read_u32(&mut file)?;
    if version < 2 {
        bail!("unsupported GGUF version {version}");
    }
    let tensor_count = read_u64(&mut file)?;
    let metadata_count = read_u64(&mut file)?;
    let mut metadata = BTreeMap::new();

    for _ in 0..metadata_count {
        let key = read_gguf_string(&mut file)?;
        let value_type = read_u32(&mut file)?;
        let value = read_metadata_value(&mut file, value_type)
            .unwrap_or_else(|_| "<unsupported>".to_string());
        metadata.insert(key, value);
    }

    let mut tensor_names = Vec::new();
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut file)?;
        tensor_names.push(name);
        let n_dimensions = read_u32(&mut file)? as usize;
        for _ in 0..n_dimensions {
            let _ = read_u64(&mut file)?;
        }
        let _ = read_u32(&mut file)?;
        let _ = read_u64(&mut file)?;
    }

    Ok(GgufInspection {
        path: path.to_path_buf(),
        size_bytes,
        version,
        tensor_count,
        metadata_count,
        metadata,
        tensor_names,
    })
}

fn read_u32(file: &mut File) -> Result<u32> {
    let mut buf = [0_u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(file: &mut File) -> Result<u64> {
    let mut buf = [0_u8; 8];
    file.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i8(file: &mut File) -> Result<i8> {
    let mut buf = [0_u8; 1];
    file.read_exact(&mut buf)?;
    Ok(buf[0] as i8)
}

fn read_u8(file: &mut File) -> Result<u8> {
    let mut buf = [0_u8; 1];
    file.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i16(file: &mut File) -> Result<i16> {
    let mut buf = [0_u8; 2];
    file.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u16(file: &mut File) -> Result<u16> {
    let mut buf = [0_u8; 2];
    file.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i32(file: &mut File) -> Result<i32> {
    let mut buf = [0_u8; 4];
    file.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_i64(file: &mut File) -> Result<i64> {
    let mut buf = [0_u8; 8];
    file.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32(file: &mut File) -> Result<f32> {
    let mut buf = [0_u8; 4];
    file.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(file: &mut File) -> Result<f64> {
    let mut buf = [0_u8; 8];
    file.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_gguf_string(file: &mut File) -> Result<String> {
    let len = read_u64(file)? as usize;
    let mut buf = vec![0_u8; len];
    file.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn read_metadata_value(file: &mut File, value_type: u32) -> Result<String> {
    Ok(match value_type {
        0 => read_u8(file)?.to_string(),
        1 => read_i8(file)?.to_string(),
        2 => read_u16(file)?.to_string(),
        3 => read_i16(file)?.to_string(),
        4 => read_u32(file)?.to_string(),
        5 => read_i32(file)?.to_string(),
        6 => read_f32(file)?.to_string(),
        7 => {
            let value = read_u8(file)?;
            if value == 0 {
                "false".to_string()
            } else {
                "true".to_string()
            }
        }
        8 => read_gguf_string(file)?,
        9 => {
            let inner_type = read_u32(file)?;
            let len = read_u64(file)? as usize;
            let mut values = Vec::with_capacity(len.min(16));
            for index in 0..len {
                let item = read_metadata_value(file, inner_type)?;
                if index < 16 {
                    values.push(item);
                }
            }
            if len > 16 {
                values.push(format!("... (+{} more)", len - 16));
            }
            format!("[{}]", values.join(", "))
        }
        10 => read_u64(file)?.to_string(),
        11 => read_i64(file)?.to_string(),
        12 => read_f64(file)?.to_string(),
        other => return Err(anyhow!("unsupported GGUF metadata type {other}")),
    })
}

struct WorkerProcess {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl WorkerProcess {
    fn spawn(config: &PythonBridgeConfig) -> Result<Self> {
        let mut child = Command::new(&config.python_bin)
            .arg(&config.worker_script)
            .current_dir(&config.project_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to spawn {}", config.worker_script.display()))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("worker stdin unavailable"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("worker stdout unavailable"))?;
        Ok(Self {
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
        })
    }

    fn send(&mut self, command: &str, payload: Value) -> Result<String> {
        let request_id = Uuid::new_v4().to_string();
        let line = json!({
            "request_id": request_id,
            "command": command,
            "payload": payload,
        });
        writeln!(self.stdin, "{}", serde_json::to_string(&line)?)?;
        self.stdin.flush()?;
        Ok(request_id)
    }

    fn read_message(&mut self) -> Result<WorkerMessage> {
        let mut line = String::new();
        let read = self.stdout.read_line(&mut line)?;
        if read == 0 {
            bail!("python worker exited unexpectedly");
        }
        Ok(serde_json::from_str::<WorkerMessage>(line.trim())?)
    }

    fn kill(&mut self) -> Result<()> {
        self.child.kill()?;
        Ok(())
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        let _ = writeln!(
            self.stdin,
            "{}",
            json!({
                "request_id": Uuid::new_v4().to_string(),
                "command": "shutdown",
                "payload": {},
            })
        );
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn normalized_output_path(payload: &Value) -> Result<PathBuf> {
    payload
        .get("output_dir")
        .or_else(|| payload.get("output_path"))
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("worker did not return output path"))
}

fn stream_dedicated_command(
    worker: &Arc<Mutex<WorkerProcess>>,
    command: &str,
    payload: &Value,
    job_id: &str,
    tx: std::sync::mpsc::Sender<ConversionEvent>,
) -> Result<()> {
    let request_id = worker.lock().send(command, payload.clone())?;
    loop {
        let message = worker.lock().read_message()?;
        match message {
            WorkerMessage::Progress {
                request_id: resp,
                progress,
                stage,
                message,
            } if resp == request_id => {
                tx.send(ConversionEvent::Progress {
                    job_id: job_id.to_string(),
                    progress,
                    stage,
                    message,
                })?;
            }
            WorkerMessage::Ok {
                request_id: resp,
                payload,
            } if resp == request_id => {
                let output_dir = normalized_output_path(&payload)?;
                let summary = payload
                    .get("summary")
                    .and_then(Value::as_str)
                    .unwrap_or("Job finished");
                tx.send(ConversionEvent::Complete {
                    job_id: job_id.to_string(),
                    output_dir,
                    report_summary: summary.to_string(),
                })?;
                return Ok(());
            }
            WorkerMessage::Error {
                request_id: resp,
                message,
            } if resp == request_id => {
                tx.send(ConversionEvent::Error {
                    job_id: job_id.to_string(),
                    message,
                })?;
                return Ok(());
            }
            _ => continue,
        }
    }
}

#[derive(Clone)]
struct WorkerClient {
    inner: Arc<Mutex<Option<WorkerProcess>>>,
}

impl WorkerClient {
    fn request(&self, command: &str, payload: Value) -> Result<Value> {
        let mut guard = self.inner.lock();
        let worker = guard
            .as_mut()
            .ok_or_else(|| anyhow!("python worker is not running"))?;
        let request_id = worker.send(command, payload)?;
        loop {
            match worker.read_message()? {
                WorkerMessage::Progress { .. } => continue,
                WorkerMessage::Ok {
                    request_id: resp,
                    payload,
                } if resp == request_id => return Ok(payload),
                WorkerMessage::Error {
                    request_id: resp,
                    message,
                } if resp == request_id => bail!(message),
                _ => continue,
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum WorkerMessage {
    Progress {
        request_id: String,
        progress: f32,
        stage: String,
        message: String,
    },
    Ok {
        request_id: String,
        payload: Value,
    },
    Error {
        request_id: String,
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn reads_gguf_header_counts() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("sample.gguf");
        fs::write(
            &path,
            [
                b"GGUF".as_slice(),
                &3_u32.to_le_bytes(),
                &0_u64.to_le_bytes(),
                &0_u64.to_le_bytes(),
            ]
            .concat(),
        )?;

        let inspection = inspect_gguf_header(&path)?;
        assert_eq!(inspection.version, 3);
        assert_eq!(inspection.tensor_count, 0);
        assert_eq!(inspection.metadata_count, 0);
        Ok(())
    }

    #[test]
    fn parses_common_metadata_fields() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("meta.gguf");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&3_u64.to_le_bytes());

        push_string(&mut bytes, "general.name");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(&mut bytes, "TinyLlama");

        push_string(&mut bytes, "context_length");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&4096_u32.to_le_bytes());

        push_string(&mut bytes, "tokenizer.ggml.add_bos_token");
        bytes.extend_from_slice(&7_u32.to_le_bytes());
        bytes.push(1);

        push_string(&mut bytes, "blk.0.attn_q.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&64_u64.to_le_bytes());
        bytes.extend_from_slice(&128_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());

        fs::write(&path, bytes)?;
        let inspection = inspect_gguf_header(&path)?;

        assert_eq!(
            inspection.metadata.get("general.name"),
            Some(&"TinyLlama".to_string())
        );
        assert_eq!(
            inspection.metadata.get("context_length"),
            Some(&"4096".to_string())
        );
        assert_eq!(
            inspection.metadata.get("tokenizer.ggml.add_bos_token"),
            Some(&"true".to_string())
        );
        assert_eq!(
            inspection.tensor_names,
            vec!["blk.0.attn_q.weight".to_string()]
        );
        Ok(())
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
}
