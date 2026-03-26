use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelSourceKind {
    Local,
    DirectUrl,
    HuggingFace,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelStatus {
    Imported,
    Inspecting,
    ReadyToConvert,
    Converting,
    Converted,
    Verified,
    Launchable,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArtifactKind {
    OriginalGguf,
    TritpackDir,
    ReconstructedCache,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerificationMode {
    Sampled,
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum JobKind {
    Import,
    Download,
    Convert,
    Verify,
    Reconstruct,
    Chat,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum JobState {
    Queued,
    Running,
    Cancelled,
    Failed,
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord {
    pub id: String,
    pub display_name: String,
    pub source: String,
    pub source_kind: ModelSourceKind,
    pub family: Option<String>,
    pub parameter_hint: Option<String>,
    pub license: Option<String>,
    pub status: ModelStatus,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub model_id: String,
    pub kind: ArtifactKind,
    pub path: PathBuf,
    pub size_bytes: i64,
    pub hash: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionProfile {
    pub alpha: f32,
    pub block_size: u32,
    pub verification_mode: VerificationMode,
}

impl Default for ConversionProfile {
    fn default() -> Self {
        Self {
            alpha: 0.7,
            block_size: 64,
            verification_mode: VerificationMode::Sampled,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeProfile {
    pub context_size: u32,
    pub gpu_layers: i32,
    pub threads: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,
}

impl Default for RuntimeProfile {
    fn default() -> Self {
        Self {
            context_size: 4096,
            gpu_layers: 99,
            threads: 8,
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: 256,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: String,
    pub model_id: String,
    pub messages: Vec<ChatMessage>,
    pub sampling_params: RuntimeProfile,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub kind: JobKind,
    pub state: JobState,
    pub model_id: Option<String>,
    pub progress: f32,
    pub stage: String,
    pub cancelable: bool,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufInspection {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub metadata: BTreeMap<String, String>,
    pub tensor_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionEstimate {
    pub estimated_bytes: u64,
    pub compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyRequest {
    pub source_path: PathBuf,
    pub tritpack_dir: PathBuf,
    pub sample_tensors: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyReport {
    pub ok: bool,
    pub tensors_verified: usize,
    pub metadata_complete: bool,
    pub mean_cosine_similarity: Option<f64>,
    pub mean_snr_db: Option<f64>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertRequest {
    pub job_id: String,
    pub source_path: PathBuf,
    pub output_dir: PathBuf,
    pub profile: ConversionProfile,
    pub sample_tensors: usize,
}

#[derive(Debug)]
pub struct ConversionStream {
    pub receiver: Receiver<ConversionEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConversionEvent {
    Progress {
        job_id: String,
        progress: f32,
        stage: String,
        message: String,
    },
    Complete {
        job_id: String,
        output_dir: PathBuf,
        report_summary: String,
    },
    Error {
        job_id: String,
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub supports_streaming: bool,
    pub supports_interrupt: bool,
    pub supports_embeddings: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendHealth {
    pub ok: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelRequest {
    pub model_id: String,
    pub tritpack_dir: PathBuf,
    pub original_hash: String,
    pub cache_dir: PathBuf,
    pub runtime_profile: RuntimeProfile,
    pub backend_binary: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHandle {
    pub id: String,
    pub model_id: String,
    pub backend_name: String,
    pub prepared_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub generation_id: String,
    pub model: ModelHandle,
    pub prompt: String,
    pub runtime_profile: RuntimeProfile,
    pub system_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceEvent {
    Started { generation_id: String },
    Token(String),
    Log(String),
    Completed { output: String },
    Error(String),
}

#[derive(Debug)]
pub struct GenerationStream {
    pub receiver: Receiver<InferenceEvent>,
}

pub trait ConversionBackend: Send + Sync {
    fn inspect_gguf(&self, path: &std::path::Path) -> Result<GgufInspection>;
    fn estimate_conversion(
        &self,
        inspection: &GgufInspection,
        profile: &ConversionProfile,
    ) -> Result<ConversionEstimate>;
    fn convert_to_tritpack(&self, request: ConvertRequest) -> Result<ConversionStream>;
    fn reconstruct_gguf_stream(
        &self,
        job_id: String,
        tritpack_dir: PathBuf,
        output_path: PathBuf,
    ) -> Result<ConversionStream>;
    fn verify_conversion(&self, request: VerifyRequest) -> Result<VerifyReport>;
    fn cancel_job(&self, job_id: &str) -> Result<()>;
}

pub trait InferenceBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> BackendCapabilities;
    fn health_check(&self) -> Result<BackendHealth>;
    fn prepare_model(&self, request: LoadModelRequest) -> Result<ModelHandle>;
    fn load_model(&self, request: LoadModelRequest) -> Result<ModelHandle>;
    fn unload_model(&self, handle: &ModelHandle) -> Result<()>;
    fn generate_stream(&self, request: GenerateRequest) -> Result<GenerationStream>;
    fn interrupt(&self, generation_id: &str) -> Result<()>;
}
