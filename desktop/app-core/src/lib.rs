mod bootstrap;

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_LENGTH, RANGE};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tritpack_runtime_api::{
    ArtifactKind, ArtifactRecord, ChatMessage, ChatRole, ChatSession, ConversionBackend,
    ConversionEvent, ConversionProfile, GenerateRequest, GgufInspection, InferenceBackend,
    InferenceEvent, JobKind, JobRecord, JobState, LoadModelRequest, ModelRecord, ModelSourceKind,
    ModelStatus, RuntimeProfile, VerificationMode, VerifyRequest,
};
use uuid::Uuid;

pub use bootstrap::{
    bootstrap_failure_status, discover_worker_script, ensure_managed_python_env,
    load_python_runtime_manifest, ResourcePaths, WorkerEnvironmentStatus,
};

pub trait ModelStore: Send + Sync {
    fn import_local(
        &self,
        source_path: &Path,
        managed_path: &Path,
        hash: &str,
    ) -> Result<ModelRecord>;
    fn import_download(&self, source: &str, managed_path: &Path, hash: &str)
        -> Result<ModelRecord>;
    fn list_models(&self) -> Result<Vec<ModelRecord>>;
    fn resolve_artifact(
        &self,
        model_id: &str,
        kind: ArtifactKind,
    ) -> Result<Option<ArtifactRecord>>;
    fn mark_state(&self, model_id: &str, status: ModelStatus) -> Result<()>;
    fn update_model_metadata(
        &self,
        model_id: &str,
        family: Option<&str>,
        parameter_hint: Option<&str>,
        license: Option<&str>,
    ) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct HuggingFaceFile {
    pub repo_id: String,
    pub filename: String,
    pub size_bytes: Option<u64>,
    pub download_url: String,
}

#[derive(Debug, Clone)]
pub struct AppPaths {
    pub root: PathBuf,
    pub models_original: PathBuf,
    pub models_tritpack: PathBuf,
    pub cache_reconstructed: PathBuf,
    pub cache_downloads: PathBuf,
    pub logs: PathBuf,
    pub state_db: PathBuf,
    pub settings_dir: PathBuf,
    pub python_root: PathBuf,
    pub managed_cpython: PathBuf,
    pub managed_venv: PathBuf,
    pub python_marker: PathBuf,
}

impl AppPaths {
    pub fn default_macos() -> Result<Self> {
        let base = dirs::data_dir()
            .ok_or_else(|| anyhow!("Could not resolve the platform data directory"))?
            .join("TritPack");
        Self::from_root(base)
    }

    pub fn from_root(root: PathBuf) -> Result<Self> {
        let paths = Self {
            models_original: root.join("models/original"),
            models_tritpack: root.join("models/tritpack"),
            cache_reconstructed: root.join("cache/reconstructed"),
            cache_downloads: root.join("cache/downloads"),
            logs: root.join("logs"),
            state_db: root.join("state/app.db"),
            settings_dir: root.join("settings"),
            python_root: root.join("python"),
            managed_cpython: root.join("python/cpython"),
            managed_venv: root.join("python/.venv"),
            python_marker: root.join("python/bootstrap-marker.json"),
            root,
        };
        paths.ensure()?;
        Ok(paths)
    }

    pub fn ensure(&self) -> Result<()> {
        for dir in [
            &self.root,
            &self.models_original,
            &self.models_tritpack,
            &self.cache_reconstructed,
            &self.cache_downloads,
            &self.logs,
            &self.settings_dir,
            &self.python_root,
            &self.managed_cpython,
            self.state_db.parent().expect("state dir"),
        ] {
            fs::create_dir_all(dir)
                .with_context(|| format!("failed to create directory {}", dir.display()))?;
        }
        Ok(())
    }
}

pub struct SqliteModelStore {
    conn: Mutex<Connection>,
}

impl SqliteModelStore {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                source TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                family TEXT,
                parameter_hint TEXT,
                license TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                model_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(kind, hash)
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'queued',
                model_id TEXT,
                progress REAL NOT NULL,
                stage TEXT NOT NULL,
                cancelable INTEGER NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                sampling_params TEXT NOT NULL,
                messages TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS downloads (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                destination TEXT NOT NULL,
                bytes_downloaded INTEGER NOT NULL,
                total_bytes INTEGER,
                etag TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "#,
        )?;
        let mut stmt = conn.prepare("PRAGMA table_info(jobs)")?;
        let columns = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if !columns.iter().any(|name| name == "state") {
            conn.execute(
                "ALTER TABLE jobs ADD COLUMN state TEXT NOT NULL DEFAULT 'queued'",
                [],
            )?;
        }
        Ok(())
    }

    fn find_model_by_artifact_hash(
        &self,
        kind: ArtifactKind,
        hash: &str,
    ) -> Result<Option<ModelRecord>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            r#"
            SELECT m.id, m.display_name, m.source, m.source_kind, m.family, m.parameter_hint,
                   m.license, m.status, m.created_at
            FROM models m
            JOIN artifacts a ON a.model_id = m.id
            WHERE a.kind = ?1 AND a.hash = ?2
            "#,
        )?;
        let record = stmt
            .query_row(params![artifact_kind_str(&kind), hash], map_model_row)
            .optional()?;
        Ok(record)
    }

    pub fn upsert_artifact(&self, artifact: &ArtifactRecord) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT INTO artifacts (model_id, kind, path, size_bytes, hash, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT(kind, hash) DO UPDATE SET
                model_id = excluded.model_id,
                path = excluded.path,
                size_bytes = excluded.size_bytes,
                created_at = excluded.created_at
            "#,
            params![
                artifact.model_id,
                artifact_kind_str(&artifact.kind),
                artifact.path.to_string_lossy().to_string(),
                artifact.size_bytes,
                artifact.hash,
                artifact.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn list_artifacts(&self, model_id: &str) -> Result<Vec<ArtifactRecord>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            r#"
            SELECT model_id, kind, path, size_bytes, hash, created_at
            FROM artifacts
            WHERE model_id = ?1
            ORDER BY created_at ASC
            "#,
        )?;
        let rows = stmt.query_map([model_id], map_artifact_row)?;
        let artifacts = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(artifacts)
    }

    pub fn create_job(
        &self,
        kind: JobKind,
        model_id: Option<&str>,
        stage: &str,
        cancelable: bool,
    ) -> Result<JobRecord> {
        let now = Utc::now();
        let job = JobRecord {
            id: Uuid::new_v4().to_string(),
            kind,
            state: JobState::Queued,
            model_id: model_id.map(ToOwned::to_owned),
            progress: 0.0,
            stage: stage.to_string(),
            cancelable,
            error: None,
            created_at: now,
            updated_at: now,
        };
        self.upsert_job(&job)?;
        Ok(job)
    }

    pub fn upsert_job(&self, job: &JobRecord) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT INTO jobs (id, kind, state, model_id, progress, stage, cancelable, error, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ON CONFLICT(id) DO UPDATE SET
                state = excluded.state,
                progress = excluded.progress,
                stage = excluded.stage,
                cancelable = excluded.cancelable,
                error = excluded.error,
                updated_at = excluded.updated_at
            "#,
            params![
                job.id,
                job_kind_str(&job.kind),
                job_state_str(&job.state),
                job.model_id,
                job.progress,
                job.stage,
                if job.cancelable { 1 } else { 0 },
                job.error,
                job.created_at.to_rfc3339(),
                job.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn list_jobs(&self) -> Result<Vec<JobRecord>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            r#"
            SELECT id, kind, state, model_id, progress, stage, cancelable, error, created_at, updated_at
            FROM jobs
            ORDER BY updated_at DESC
            "#,
        )?;
        let rows = stmt.query_map([], map_job_row)?;
        let jobs = rows.collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(jobs)
    }

    pub fn save_session(&self, session: &ChatSession) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT OR REPLACE INTO sessions (id, model_id, sampling_params, messages, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5)
            "#,
            params![
                session.id,
                session.model_id,
                serde_json::to_string(&session.sampling_params)?,
                serde_json::to_string(&session.messages)?,
                session.created_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn list_sessions(&self) -> Result<Vec<ChatSession>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            r#"
            SELECT id, model_id, sampling_params, messages, created_at
            FROM sessions
            ORDER BY created_at DESC
            "#,
        )?;
        let rows = stmt.query_map([], |row| {
            let sampling_params: String = row.get(2)?;
            let messages: String = row.get(3)?;
            let created_at: String = row.get(4)?;
            Ok(ChatSession {
                id: row.get(0)?,
                model_id: row.get(1)?,
                sampling_params: serde_json::from_str(&sampling_params).map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        2,
                        rusqlite::types::Type::Text,
                        Box::new(error),
                    )
                })?,
                messages: serde_json::from_str(&messages).map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        3,
                        rusqlite::types::Type::Text,
                        Box::new(error),
                    )
                })?,
                created_at: DateTime::parse_from_rfc3339(&created_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            4,
                            rusqlite::types::Type::Text,
                            Box::new(error),
                        )
                    })?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn get_setting(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock();
        let value = conn
            .query_row("SELECT value FROM settings WHERE key = ?1", [key], |row| {
                row.get(0)
            })
            .optional()?;
        Ok(value)
    }
}

impl ModelStore for SqliteModelStore {
    fn import_local(
        &self,
        source_path: &Path,
        managed_path: &Path,
        hash: &str,
    ) -> Result<ModelRecord> {
        if let Some(existing) =
            self.find_model_by_artifact_hash(ArtifactKind::OriginalGguf, hash)?
        {
            return Ok(existing);
        }

        let display_name = source_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model.gguf")
            .to_string();
        let model = ModelRecord {
            id: Uuid::new_v4().to_string(),
            display_name,
            source: source_path.display().to_string(),
            source_kind: ModelSourceKind::Local,
            family: None,
            parameter_hint: None,
            license: None,
            status: ModelStatus::Imported,
            created_at: Utc::now(),
        };

        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT INTO models (id, display_name, source, source_kind, family, parameter_hint, license, status, created_at)
            VALUES (?1, ?2, ?3, ?4, NULL, NULL, NULL, ?5, ?6)
            "#,
            params![
                model.id,
                model.display_name,
                model.source,
                source_kind_str(&model.source_kind),
                model_status_str(&model.status),
                model.created_at.to_rfc3339(),
            ],
        )?;
        drop(conn);

        let artifact = ArtifactRecord {
            model_id: model.id.clone(),
            kind: ArtifactKind::OriginalGguf,
            path: managed_path.to_path_buf(),
            size_bytes: fs::metadata(managed_path)?.len() as i64,
            hash: hash.to_string(),
            created_at: Utc::now(),
        };
        self.upsert_artifact(&artifact)?;
        Ok(model)
    }

    fn import_download(
        &self,
        source: &str,
        managed_path: &Path,
        hash: &str,
    ) -> Result<ModelRecord> {
        if let Some(existing) =
            self.find_model_by_artifact_hash(ArtifactKind::OriginalGguf, hash)?
        {
            return Ok(existing);
        }

        let display_name = managed_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("downloaded.gguf")
            .to_string();
        let model = ModelRecord {
            id: Uuid::new_v4().to_string(),
            display_name,
            source: source.to_string(),
            source_kind: if source.contains("huggingface.co") {
                ModelSourceKind::HuggingFace
            } else {
                ModelSourceKind::DirectUrl
            },
            family: None,
            parameter_hint: None,
            license: None,
            status: ModelStatus::Imported,
            created_at: Utc::now(),
        };

        let conn = self.conn.lock();
        conn.execute(
            r#"
            INSERT INTO models (id, display_name, source, source_kind, family, parameter_hint, license, status, created_at)
            VALUES (?1, ?2, ?3, ?4, NULL, NULL, NULL, ?5, ?6)
            "#,
            params![
                model.id,
                model.display_name,
                model.source,
                source_kind_str(&model.source_kind),
                model_status_str(&model.status),
                model.created_at.to_rfc3339(),
            ],
        )?;
        drop(conn);

        let artifact = ArtifactRecord {
            model_id: model.id.clone(),
            kind: ArtifactKind::OriginalGguf,
            path: managed_path.to_path_buf(),
            size_bytes: fs::metadata(managed_path)?.len() as i64,
            hash: hash.to_string(),
            created_at: Utc::now(),
        };
        self.upsert_artifact(&artifact)?;
        Ok(model)
    }

    fn list_models(&self) -> Result<Vec<ModelRecord>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            r#"
            SELECT id, display_name, source, source_kind, family, parameter_hint, license, status, created_at
            FROM models
            ORDER BY created_at DESC
            "#,
        )?;
        let rows = stmt.query_map([], map_model_row)?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    fn resolve_artifact(
        &self,
        model_id: &str,
        kind: ArtifactKind,
    ) -> Result<Option<ArtifactRecord>> {
        let conn = self.conn.lock();
        let artifact = conn
            .query_row(
                r#"
                SELECT model_id, kind, path, size_bytes, hash, created_at
                FROM artifacts
                WHERE model_id = ?1 AND kind = ?2
                ORDER BY created_at DESC
                LIMIT 1
                "#,
                params![model_id, artifact_kind_str(&kind)],
                map_artifact_row,
            )
            .optional()?;
        Ok(artifact)
    }

    fn mark_state(&self, model_id: &str, status: ModelStatus) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE models SET status = ?1 WHERE id = ?2",
            params![model_status_str(&status), model_id],
        )?;
        Ok(())
    }

    fn update_model_metadata(
        &self,
        model_id: &str,
        family: Option<&str>,
        parameter_hint: Option<&str>,
        license: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE models SET family = ?1, parameter_hint = ?2, license = ?3 WHERE id = ?4",
            params![family, parameter_hint, license, model_id],
        )?;
        Ok(())
    }
}

#[derive(Default)]
pub struct JobController {
    flags: Mutex<std::collections::HashMap<String, Arc<AtomicBool>>>,
}

impl JobController {
    pub fn register(&self, job_id: &str) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        self.flags
            .lock()
            .insert(job_id.to_string(), Arc::clone(&flag));
        flag
    }

    pub fn cancel(&self, job_id: &str) -> bool {
        if let Some(flag) = self.flags.lock().get(job_id) {
            flag.store(true, Ordering::SeqCst);
            return true;
        }
        false
    }

    pub fn is_cancelled(&self, job_id: &str) -> bool {
        self.flags
            .lock()
            .get(job_id)
            .map(|flag| flag.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    pub fn clear(&self, job_id: &str) {
        self.flags.lock().remove(job_id);
    }
}

pub struct DesktopApp {
    pub paths: AppPaths,
    pub resources: ResourcePaths,
    pub store: Arc<SqliteModelStore>,
    conversion_backend: Arc<dyn ConversionBackend>,
    runtime_backend: Arc<dyn InferenceBackend>,
    http: Client,
    job_controller: Arc<JobController>,
    worker_status: Mutex<WorkerEnvironmentStatus>,
}

impl DesktopApp {
    pub fn new(
        paths: AppPaths,
        resources: ResourcePaths,
        store: Arc<SqliteModelStore>,
        conversion_backend: Arc<dyn ConversionBackend>,
        runtime_backend: Arc<dyn InferenceBackend>,
        worker_status: WorkerEnvironmentStatus,
    ) -> Result<Self> {
        paths.ensure()?;
        Ok(Self {
            paths,
            resources,
            store,
            conversion_backend,
            runtime_backend,
            http: Client::builder().build()?,
            job_controller: Arc::new(JobController::default()),
            worker_status: Mutex::new(worker_status),
        })
    }

    pub fn list_models(&self) -> Result<Vec<ModelRecord>> {
        self.store.list_models()
    }

    pub fn list_jobs(&self) -> Result<Vec<JobRecord>> {
        self.store.list_jobs()
    }

    pub fn list_model_summaries(&self) -> Result<Vec<String>> {
        let models = self.list_models()?;
        let summaries = models
            .into_iter()
            .map(|model| {
                let artifacts = self.store.list_artifacts(&model.id).unwrap_or_default();
                let original = artifacts
                    .iter()
                    .find(|artifact| artifact.kind == ArtifactKind::OriginalGguf);
                let tritpack = artifacts
                    .iter()
                    .find(|artifact| artifact.kind == ArtifactKind::TritpackDir);
                format!(
                    "{} | {} | {:?} | original={} MB | tritpack={} MB",
                    short_id(&model.id),
                    model.display_name,
                    model.status,
                    original
                        .map(|artifact| artifact.size_bytes / (1 << 20))
                        .unwrap_or_default(),
                    tritpack
                        .map(|artifact| artifact.size_bytes / (1 << 20))
                        .unwrap_or_default(),
                )
            })
            .collect();
        Ok(summaries)
    }

    pub fn list_job_summaries(&self) -> Result<Vec<String>> {
        let jobs = self.list_jobs()?;
        Ok(jobs
            .into_iter()
            .map(|job| {
                format!(
                    "{} | {:?} | {:?} | {:.0}% | {}{}",
                    short_id(&job.id),
                    job.kind,
                    job.state,
                    job.progress * 100.0,
                    job.stage,
                    job.error
                        .as_ref()
                        .map(|error| format!(" | error={error}"))
                        .unwrap_or_default()
                )
            })
            .collect())
    }

    pub fn list_chat_sessions(&self) -> Result<Vec<ChatSession>> {
        self.store.list_sessions()
    }

    pub fn list_chat_session_summaries(&self) -> Result<Vec<String>> {
        let models = self.list_models()?;
        let sessions = self.list_chat_sessions()?;
        Ok(sessions
            .into_iter()
            .map(|session| {
                let model_name = models
                    .iter()
                    .find(|model| model.id == session.model_id)
                    .map(|model| model.display_name.as_str())
                    .unwrap_or("Unknown model");
                let preview = session
                    .messages
                    .iter()
                    .find(|message| message.role == ChatRole::User)
                    .map(|message| truncate_preview(&message.content, 44))
                    .unwrap_or_else(|| "Conversation".to_string());
                format!(
                    "{} | {} | {}",
                    session.created_at.format("%b %-d, %-I:%M %p"),
                    model_name,
                    preview
                )
            })
            .collect())
    }

    pub fn set_ui_setting(&self, key: &str, value: &str) -> Result<()> {
        self.store.set_setting(key, value)
    }

    pub fn get_ui_setting(&self, key: &str) -> Result<Option<String>> {
        self.store.get_setting(key)
    }

    pub fn import_local(&self, source_path: impl AsRef<Path>) -> Result<ModelRecord> {
        let source_path = source_path.as_ref();
        let file_name = source_path
            .file_name()
            .ok_or_else(|| anyhow!("source path must include a file name"))?;
        let hash = hash_file(source_path)?;
        let managed_path = self.paths.models_original.join(format!(
            "{}-{}",
            &hash[..12],
            file_name.to_string_lossy()
        ));
        if !managed_path.exists() {
            fs::copy(source_path, &managed_path)
                .with_context(|| format!("failed to import {}", source_path.display()))?;
        }
        self.store.import_local(source_path, &managed_path, &hash)
    }

    pub fn inspect_model(&self, model_id: &str) -> Result<GgufInspection> {
        let current_status = self
            .store
            .list_models()?
            .into_iter()
            .find(|model| model.id == model_id)
            .map(|model| model.status)
            .ok_or_else(|| anyhow!("model {model_id} not found"))?;
        let original = self
            .store
            .resolve_artifact(model_id, ArtifactKind::OriginalGguf)?
            .ok_or_else(|| anyhow!("no original GGUF artifact found for model {model_id}"))?;
        let inspection = self.conversion_backend.inspect_gguf(&original.path)?;
        self.store.update_model_metadata(
            model_id,
            inspection
                .metadata
                .get("general.architecture")
                .map(String::as_str),
            inspection
                .metadata
                .get("general.size_label")
                .or_else(|| inspection.metadata.get("general.basename"))
                .map(String::as_str),
            inspection
                .metadata
                .get("general.license")
                .or_else(|| inspection.metadata.get("general.license.name"))
                .map(String::as_str),
        )?;
        if matches!(
            current_status,
            ModelStatus::Imported | ModelStatus::Inspecting
        ) {
            self.store
                .mark_state(model_id, ModelStatus::ReadyToConvert)?;
        }
        Ok(inspection)
    }

    pub fn model_detail_summary(&self, model_id: &str) -> Result<Vec<String>> {
        let models = self.store.list_models()?;
        let model = models
            .into_iter()
            .find(|record| record.id == model_id)
            .ok_or_else(|| anyhow!("model {model_id} not found"))?;
        let artifacts = self.store.list_artifacts(model_id)?;
        let mut lines = vec![
            format!("Model: {}", model.display_name),
            format!("Status: {:?}", model.status),
            format!("Source: {}", model.source),
            format!(
                "Family: {}",
                model.family.unwrap_or_else(|| "unknown".to_string())
            ),
            format!(
                "Parameter Hint: {}",
                model
                    .parameter_hint
                    .unwrap_or_else(|| "unknown".to_string())
            ),
            format!(
                "License: {}",
                model.license.unwrap_or_else(|| "unknown".to_string())
            ),
        ];

        if let Ok(inspection) = self.inspect_model(model_id) {
            lines.push(format!(
                "GGUF: version {} | tensors {} | metadata {} | file {} MB",
                inspection.version,
                inspection.tensor_count,
                inspection.metadata_count,
                inspection.size_bytes / (1 << 20)
            ));
            if let Ok(estimate) = self
                .conversion_backend
                .estimate_conversion(&inspection, &ConversionProfile::default())
            {
                lines.push(format!(
                    "Estimate: {} MB | {:.2}x compression at alpha={} block_size={}",
                    estimate.estimated_bytes / (1 << 20),
                    estimate.compression_ratio,
                    ConversionProfile::default().alpha,
                    ConversionProfile::default().block_size
                ));
            }
            for (key, value) in inspection.metadata.iter().take(8) {
                lines.push(format!("{key}: {value}"));
            }
            if !inspection.tensor_names.is_empty() {
                lines.push(format!(
                    "Tensors: {}",
                    inspection
                        .tensor_names
                        .iter()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        for artifact in artifacts {
            lines.push(format!(
                "Artifact {:?}: {} MB | {}",
                artifact.kind,
                artifact.size_bytes / (1 << 20),
                artifact.path.display()
            ));
        }

        Ok(lines)
    }

    pub fn runtime_detail_summary(&self) -> Result<Vec<String>> {
        let health = self.runtime_backend.health_check()?;
        let caps = self.runtime_backend.capabilities();
        let worker = self.worker_status.lock().clone();
        let models = self.store.list_models()?;
        let cached = models
            .iter()
            .filter(|model| {
                self.store
                    .resolve_artifact(&model.id, ArtifactKind::ReconstructedCache)
                    .ok()
                    .flatten()
                    .is_some()
            })
            .count();
        Ok(vec![
            format!("Backend: {}", self.runtime_backend.name()),
            format!(
                "Runtime: {}",
                if health.ok { "Ready" } else { "Unavailable" }
            ),
            format!("Runtime detail: {}", summarize_status_line(&health.message)),
            format!(
                "Streaming: {}",
                if caps.supports_streaming {
                    "Supported"
                } else {
                    "Unavailable"
                }
            ),
            format!(
                "Interrupt: {}",
                if caps.supports_interrupt {
                    "Supported"
                } else {
                    "Unavailable"
                }
            ),
            format!(
                "Embeddings: {}",
                if caps.supports_embeddings {
                    "Supported"
                } else {
                    "Unavailable"
                }
            ),
            format!(
                "Reconstructed cache: {}",
                summarize_path(&self.paths.cache_reconstructed)
            ),
            format!("Prepared models cached: {}", cached),
            format!(
                "Worker: {}",
                if worker.ok { "Ready" } else { "Setup needed" }
            ),
            format!("Worker detail: {}", summarize_status_line(&worker.message)),
            format!(
                "Worker interpreter: {}",
                summarize_path(&worker.interpreter_path)
            ),
            format!("Worker venv: {}", summarize_path(&worker.venv_path)),
            format!("Worker marker: {}", summarize_path(&worker.marker_path)),
            format!("Worker python version: {}", worker.python_version),
            format!("Worker lock sha: {}", short_hash(&worker.lock_sha256)),
        ])
    }

    pub fn queue_conversion(&self, model_id: &str, profile: ConversionProfile) -> Result<String> {
        let original = self
            .store
            .resolve_artifact(model_id, ArtifactKind::OriginalGguf)?
            .ok_or_else(|| anyhow!("no original GGUF artifact found for model {model_id}"))?;
        let output_dir = self
            .paths
            .models_tritpack
            .join(format!("{model_id}.tritpack"));
        let job = self.store.create_job(
            JobKind::Convert,
            Some(model_id),
            "Queued for conversion",
            true,
        )?;
        let job_id = job.id.clone();
        let cancel_flag = self.job_controller.register(&job_id);
        self.store.mark_state(model_id, ModelStatus::Converting)?;

        let backend = Arc::clone(&self.conversion_backend);
        let store = Arc::clone(&self.store);
        let job_controller = Arc::clone(&self.job_controller);
        let source_hash = original.hash.clone();
        let model_id = model_id.to_string();

        thread::spawn(move || {
            let mut running = job.clone();
            running.state = JobState::Running;
            running.stage = "Starting conversion".to_string();
            running.updated_at = Utc::now();
            let _ = store.upsert_job(&running);
            let request = tritpack_runtime_api::ConvertRequest {
                job_id: job.id.clone(),
                source_path: original.path.clone(),
                output_dir: output_dir.clone(),
                profile: profile.clone(),
                sample_tensors: match profile.verification_mode {
                    VerificationMode::Sampled => 8,
                    VerificationMode::Full => usize::MAX,
                },
            };

            match backend.convert_to_tritpack(request) {
                Ok(stream) => {
                    while let Ok(event) = stream.receiver.recv() {
                        match event {
                            ConversionEvent::Progress {
                                progress,
                                stage,
                                message,
                                ..
                            } => {
                                let mut updated = job.clone();
                                updated.state = JobState::Running;
                                updated.progress = progress;
                                updated.stage = format!("{stage}: {message}");
                                updated.updated_at = Utc::now();
                                let _ = store.upsert_job(&updated);
                            }
                            ConversionEvent::Complete {
                                output_dir,
                                report_summary,
                                ..
                            } => {
                                if cancel_flag.load(Ordering::SeqCst) {
                                    let quarantine = output_dir
                                        .with_extension(format!("cancelled-{}", &job.id[..8]));
                                    let _ = fs::rename(&output_dir, &quarantine);
                                    let mut updated = job.clone();
                                    updated.state = JobState::Cancelled;
                                    updated.stage = "Conversion cancelled".to_string();
                                    updated.updated_at = Utc::now();
                                    let _ = store.upsert_job(&updated);
                                    let _ = store.mark_state(&model_id, ModelStatus::Converted);
                                    job_controller.clear(&job.id);
                                    break;
                                }
                                let artifact = ArtifactRecord {
                                    model_id: model_id.clone(),
                                    kind: ArtifactKind::TritpackDir,
                                    size_bytes: directory_size(&output_dir).unwrap_or_default()
                                        as i64,
                                    hash: hash_string(&format!(
                                        "{}:{}:{}",
                                        source_hash, profile.alpha, profile.block_size
                                    )),
                                    path: output_dir.clone(),
                                    created_at: Utc::now(),
                                };
                                let _ = store.upsert_artifact(&artifact);

                                let verify = backend.verify_conversion(VerifyRequest {
                                    source_path: original.path.clone(),
                                    tritpack_dir: output_dir.clone(),
                                    sample_tensors: if profile.verification_mode
                                        == VerificationMode::Full
                                    {
                                        usize::MAX
                                    } else {
                                        8
                                    },
                                });

                                let mut updated = job.clone();
                                updated.state = JobState::Completed;
                                updated.progress = 1.0;
                                updated.stage = match verify {
                                    Ok(ref report) if report.ok => {
                                        format!("Verified | {report_summary}")
                                    }
                                    Ok(ref report) => format!(
                                        "Converted with warnings | {}",
                                        report.notes.join("; ")
                                    ),
                                    Err(ref error) => {
                                        format!("Converted, verification skipped | {error}")
                                    }
                                };
                                updated.updated_at = Utc::now();
                                let _ = store.upsert_job(&updated);
                                let _ = store.mark_state(
                                    &model_id,
                                    if verify.as_ref().map(|report| report.ok).unwrap_or(false) {
                                        ModelStatus::Launchable
                                    } else {
                                        ModelStatus::Converted
                                    },
                                );
                                job_controller.clear(&job.id);
                                break;
                            }
                            ConversionEvent::Error { message, .. } => {
                                let mut updated = job.clone();
                                if cancel_flag.load(Ordering::SeqCst) {
                                    updated.state = JobState::Cancelled;
                                    updated.stage = "Conversion cancelled".to_string();
                                } else {
                                    updated.state = JobState::Failed;
                                    updated.error = Some(message.clone());
                                    updated.stage = "Conversion failed".to_string();
                                }
                                updated.updated_at = Utc::now();
                                let _ = store.upsert_job(&updated);
                                if !cancel_flag.load(Ordering::SeqCst) {
                                    let _ = store.mark_state(&model_id, ModelStatus::Error);
                                }
                                job_controller.clear(&job.id);
                                break;
                            }
                        }
                    }
                }
                Err(error) => {
                    let mut updated = job.clone();
                    if cancel_flag.load(Ordering::SeqCst) {
                        updated.state = JobState::Cancelled;
                        updated.stage = "Conversion cancelled".to_string();
                    } else {
                        updated.state = JobState::Failed;
                        updated.error = Some(error.to_string());
                        updated.stage = "Conversion failed".to_string();
                    }
                    updated.updated_at = Utc::now();
                    let _ = store.upsert_job(&updated);
                    if !cancel_flag.load(Ordering::SeqCst) {
                        let _ = store.mark_state(&model_id, ModelStatus::Error);
                    }
                }
            }
            job_controller.clear(&job.id);
        });

        Ok(job_id)
    }

    pub fn queue_prepare_runtime(
        &self,
        model_id: &str,
        _runtime_profile: RuntimeProfile,
    ) -> Result<String> {
        let tritpack_artifact = self
            .store
            .resolve_artifact(model_id, ArtifactKind::TritpackDir)?
            .ok_or_else(|| anyhow!("model {model_id} does not have a TritPack artifact yet"))?;
        let original = self
            .store
            .resolve_artifact(model_id, ArtifactKind::OriginalGguf)?
            .ok_or_else(|| anyhow!("model {model_id} does not have an original GGUF artifact"))?;
        let job = self.store.create_job(
            JobKind::Reconstruct,
            Some(model_id),
            "Queued for runtime preparation",
            true,
        )?;
        let job_id = job.id.clone();
        let store = Arc::clone(&self.store);
        let conversion_backend = Arc::clone(&self.conversion_backend);
        let model_id = model_id.to_string();
        let tritpack_dir = tritpack_artifact.path.clone();
        let original_hash = original.hash.clone();
        let job_controller = Arc::clone(&self.job_controller);
        let output_path = self.paths.cache_reconstructed.join(format!(
            "{}-{}.gguf",
            model_id,
            &original_hash[..12]
        ));
        let cancel_flag = self.job_controller.register(&job_id);

        thread::spawn(move || {
            let mut updated = job.clone();
            updated.state = JobState::Running;
            updated.stage = "Preparing runtime cache".to_string();
            updated.updated_at = Utc::now();
            let _ = store.upsert_job(&updated);

            match conversion_backend.reconstruct_gguf_stream(
                job.id.clone(),
                tritpack_dir.clone(),
                output_path.clone(),
            ) {
                Ok(stream) => {
                    while let Ok(event) = stream.receiver.recv() {
                        match event {
                            ConversionEvent::Progress {
                                progress,
                                stage,
                                message,
                                ..
                            } => {
                                let mut in_progress = job.clone();
                                in_progress.state = JobState::Running;
                                in_progress.progress = progress;
                                in_progress.stage = format!("{stage}: {message}");
                                in_progress.updated_at = Utc::now();
                                let _ = store.upsert_job(&in_progress);
                            }
                            ConversionEvent::Complete { .. } => {
                                if cancel_flag.load(Ordering::SeqCst) {
                                    let quarantine = output_path
                                        .with_extension(format!("cancelled-{}", &job.id[..8]));
                                    let _ = fs::rename(&output_path, &quarantine);
                                    let mut cancelled = job.clone();
                                    cancelled.state = JobState::Cancelled;
                                    cancelled.stage = "Runtime preparation cancelled".to_string();
                                    cancelled.updated_at = Utc::now();
                                    let _ = store.upsert_job(&cancelled);
                                    job_controller.clear(&job.id);
                                    break;
                                }
                                match persist_reconstructed_artifact(
                                    &store,
                                    &model_id,
                                    &original_hash,
                                    &output_path,
                                ) {
                                    Ok(_) => {
                                        updated.state = JobState::Completed;
                                        updated.progress = 1.0;
                                        updated.stage = format!(
                                            "Runtime ready | cached {}",
                                            output_path.display()
                                        );
                                        updated.updated_at = Utc::now();
                                        let _ = store.upsert_job(&updated);
                                        let _ =
                                            store.mark_state(&model_id, ModelStatus::Launchable);
                                    }
                                    Err(error) => {
                                        updated.state = JobState::Failed;
                                        updated.error = Some(error.to_string());
                                        updated.stage = "Runtime preparation failed".to_string();
                                        updated.updated_at = Utc::now();
                                        let _ = store.upsert_job(&updated);
                                    }
                                }
                                job_controller.clear(&job.id);
                                break;
                            }
                            ConversionEvent::Error { message, .. } => {
                                if cancel_flag.load(Ordering::SeqCst) {
                                    updated.state = JobState::Cancelled;
                                    updated.stage = "Runtime preparation cancelled".to_string();
                                } else {
                                    updated.state = JobState::Failed;
                                    updated.error = Some(message);
                                    updated.stage = "Runtime preparation failed".to_string();
                                }
                                updated.updated_at = Utc::now();
                                let _ = store.upsert_job(&updated);
                                job_controller.clear(&job.id);
                                break;
                            }
                        }
                    }
                }
                Err(error) => {
                    updated.state = JobState::Failed;
                    updated.error = Some(error.to_string());
                    updated.stage = "Runtime preparation failed".to_string();
                    updated.updated_at = Utc::now();
                    let _ = store.upsert_job(&updated);
                }
            }
            job_controller.clear(&job.id);
        });

        Ok(job_id)
    }

    pub fn remove_reconstructed_cache(&self, model_id: &str) -> Result<()> {
        if let Some(cache_artifact) = self
            .store
            .resolve_artifact(model_id, ArtifactKind::ReconstructedCache)?
        {
            if cache_artifact.path.exists() {
                if cache_artifact.path.is_dir() {
                    fs::remove_dir_all(&cache_artifact.path)?;
                } else {
                    fs::remove_file(&cache_artifact.path)?;
                }
            }
        }
        Ok(())
    }

    pub fn save_huggingface_token(&self, token: &str) -> Result<()> {
        self.store.set_setting("huggingface.token", token)
    }

    pub fn load_huggingface_token(&self) -> Result<Option<String>> {
        self.store.get_setting("huggingface.token")
    }

    pub fn search_huggingface(&self, query: &str) -> Result<Vec<HuggingFaceFile>> {
        #[derive(Debug, Deserialize)]
        struct HfModel {
            id: String,
            siblings: Option<Vec<HfSibling>>,
        }

        #[derive(Debug, Deserialize)]
        struct HfSibling {
            rfilename: String,
            size: Option<u64>,
        }

        let url = format!("https://huggingface.co/api/models?search={query}&limit=20&full=true");
        let mut request = self.http.get(url);
        if let Some(token) = self.load_huggingface_token()? {
            if !token.trim().is_empty() {
                request = request.header(AUTHORIZATION, format!("Bearer {token}"));
            }
        }
        let response = request.send()?.error_for_status()?;
        let models: Vec<HfModel> = response.json()?;

        let mut results = Vec::new();
        for model in models {
            for sibling in model.siblings.unwrap_or_default() {
                if sibling.rfilename.ends_with(".gguf") {
                    results.push(HuggingFaceFile {
                        download_url: format!(
                            "https://huggingface.co/{}/resolve/main/{}?download=1",
                            model.id, sibling.rfilename
                        ),
                        repo_id: model.id.clone(),
                        filename: sibling.rfilename,
                        size_bytes: sibling.size,
                    });
                }
            }
        }
        Ok(results)
    }

    pub fn download_from_url(&self, url: &str) -> Result<ModelRecord> {
        self.download_managed_file(url, url)
    }

    pub fn download_huggingface_file(&self, file: &HuggingFaceFile) -> Result<ModelRecord> {
        self.download_managed_file(
            &file.download_url,
            &format!("{}/{}", file.repo_id, file.filename),
        )
    }

    pub fn queue_download_from_url(&self, url: &str) -> Result<String> {
        self.queue_download(url, url)
    }

    pub fn queue_download_huggingface_file(&self, file: &HuggingFaceFile) -> Result<String> {
        self.queue_download(
            &file.download_url,
            &format!("{}/{}", file.repo_id, file.filename),
        )
    }

    fn queue_download(&self, url: &str, source_label: &str) -> Result<String> {
        let job = self
            .store
            .create_job(JobKind::Download, None, "Queued for download", true)?;
        let job_id = job.id.clone();
        let cancel_flag = self.job_controller.register(&job_id);
        let url = url.to_string();
        let source_label = source_label.to_string();
        let store = Arc::clone(&self.store);
        let http = self.http.clone();
        let paths = self.paths.clone();
        let token = store.get_setting("huggingface.token")?;
        let job_controller = Arc::clone(&self.job_controller);

        thread::spawn(move || {
            if let Err(error) = perform_managed_download(
                store.clone(),
                &http,
                &paths,
                &job,
                &url,
                &source_label,
                token.as_deref(),
                cancel_flag,
            ) {
                let existing = store
                    .list_jobs()
                    .ok()
                    .and_then(|jobs| jobs.into_iter().find(|candidate| candidate.id == job.id))
                    .unwrap_or_else(|| job.clone());
                if existing.state != JobState::Cancelled {
                    let mut updated = existing;
                    updated.state = JobState::Failed;
                    updated.error = Some(error.to_string());
                    updated.stage = "Download failed".to_string();
                    updated.updated_at = Utc::now();
                    let _ = store.upsert_job(&updated);
                }
            }
            job_controller.clear(&job.id);
        });

        Ok(job_id)
    }

    fn download_managed_file(&self, url: &str, source_label: &str) -> Result<ModelRecord> {
        let job = self
            .store
            .create_job(JobKind::Download, None, "Downloading GGUF", true)?;
        perform_managed_download(
            Arc::clone(&self.store),
            &self.http,
            &self.paths,
            &job,
            url,
            source_label,
            self.load_huggingface_token()?.as_deref(),
            Arc::new(AtomicBool::new(false)),
        )
    }

    pub fn cancel_job(&self, job_id: &str) -> Result<()> {
        let jobs = self.list_jobs()?;
        let job = jobs
            .into_iter()
            .find(|candidate| candidate.id == job_id)
            .ok_or_else(|| anyhow!("job {job_id} not found"))?;
        if !job.cancelable {
            return Err(anyhow!("job {job_id} is not cancellable"));
        }
        self.job_controller.cancel(job_id);
        match job.kind {
            JobKind::Convert | JobKind::Reconstruct => {
                let _ = self.conversion_backend.cancel_job(job_id);
            }
            JobKind::Chat => {}
            JobKind::Download | JobKind::Import | JobKind::Verify => {}
        }
        let mut updated = job;
        updated.state = JobState::Cancelled;
        updated.stage = "Cancellation requested".to_string();
        updated.updated_at = Utc::now();
        self.store.upsert_job(&updated)?;
        Ok(())
    }

    pub fn generate_stream(
        &self,
        model_id: &str,
        prompt: String,
        runtime_profile: RuntimeProfile,
    ) -> Result<std::sync::mpsc::Receiver<InferenceEvent>> {
        let tritpack_artifact = self
            .store
            .resolve_artifact(model_id, ArtifactKind::TritpackDir)?
            .ok_or_else(|| anyhow!("model {model_id} does not have a TritPack artifact yet"))?;
        let original = self
            .store
            .resolve_artifact(model_id, ArtifactKind::OriginalGguf)?
            .ok_or_else(|| anyhow!("model {model_id} does not have an original GGUF artifact"))?;

        let handle = self.runtime_backend.prepare_model(LoadModelRequest {
            model_id: model_id.to_string(),
            tritpack_dir: tritpack_artifact.path.clone(),
            original_hash: original.hash.clone(),
            cache_dir: self.paths.cache_reconstructed.clone(),
            runtime_profile: runtime_profile.clone(),
            backend_binary: None,
        })?;
        persist_reconstructed_artifact(
            &self.store,
            model_id,
            &original.hash,
            &handle.prepared_path,
        )?;

        let session = ChatSession {
            id: Uuid::new_v4().to_string(),
            model_id: model_id.to_string(),
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: prompt.clone(),
            }],
            sampling_params: runtime_profile.clone(),
            created_at: Utc::now(),
        };
        self.store.save_session(&session)?;

        let stream = self.runtime_backend.generate_stream(GenerateRequest {
            generation_id: Uuid::new_v4().to_string(),
            model: handle,
            prompt,
            runtime_profile,
            system_prompt: None,
        })?;
        Ok(stream.receiver)
    }

    pub fn runtime_health(&self) -> Result<String> {
        let health = self.runtime_backend.health_check()?;
        let worker = self.worker_status.lock().clone();
        Ok(if !worker.ok {
            "Worker setup needed".to_string()
        } else if health.ok {
            "Runtime ready".to_string()
        } else {
            "llama.cpp unavailable".to_string()
        })
    }
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|ch| match ch {
            '/' | '\\' | ':' | '?' | '&' | '=' | ' ' => '_',
            _ => ch,
        })
        .collect()
}

fn truncate_preview(value: &str, limit: usize) -> String {
    let trimmed = value.trim();
    let mut chars = trimmed.chars();
    let preview: String = chars.by_ref().take(limit).collect();
    if chars.next().is_some() {
        format!("{preview}...")
    } else {
        preview
    }
}

fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}

fn short_hash(hash: &str) -> String {
    hash.chars().take(12).collect()
}

fn summarize_status_line(message: &str) -> String {
    let line = message
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or(message.trim());

    if line.chars().count() <= 96 {
        return line.to_string();
    }

    let truncated: String = line.chars().take(93).collect();
    format!("{truncated}...")
}

fn summarize_path(path: &Path) -> String {
    if let Some(home) = dirs::home_dir() {
        if let Ok(stripped) = path.strip_prefix(&home) {
            return format!("~/{}", stripped.display());
        }
    }

    let text = path.display().to_string();
    if text.chars().count() <= 72 {
        return text;
    }

    let tail: String = text
        .chars()
        .rev()
        .take(69)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("...{tail}")
}

fn hash_string(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn directory_size(path: &Path) -> Result<u64> {
    if path.is_file() {
        return Ok(fs::metadata(path)?.len());
    }
    let mut total = 0;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let child = entry.path();
        total += directory_size(&child)?;
    }
    Ok(total)
}

fn perform_managed_download(
    store: Arc<SqliteModelStore>,
    http: &Client,
    paths: &AppPaths,
    job: &JobRecord,
    url: &str,
    source_label: &str,
    token: Option<&str>,
    cancel_flag: Arc<AtomicBool>,
) -> Result<ModelRecord> {
    let file_name = sanitize_filename(
        url.split('/')
            .last()
            .unwrap_or("model.gguf")
            .trim_end_matches("?download=1"),
    );
    let partial_path = paths
        .cache_downloads
        .join(format!("{}.part", hash_string(url)));
    let destination = paths.models_original.join(file_name);
    let mut request = http.get(url);
    if let Some(token) = token {
        if !token.trim().is_empty() {
            request = request.header(AUTHORIZATION, format!("Bearer {token}"));
        }
    }
    let mut downloaded = if partial_path.exists() {
        fs::metadata(&partial_path)?.len()
    } else {
        0
    };
    if downloaded > 0 {
        request = request.header(RANGE, format!("bytes={downloaded}-"));
    }
    let mut response = request.send()?.error_for_status()?;
    let total = response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_default()
        + downloaded;

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&partial_path)?;
    let mut updated = job.clone();
    updated.state = JobState::Running;
    updated.stage = "Downloading GGUF".to_string();
    updated.updated_at = Utc::now();
    store.upsert_job(&updated)?;

    let mut buffer = [0_u8; 64 * 1024];
    loop {
        if cancel_flag.load(Ordering::SeqCst) {
            updated.state = JobState::Cancelled;
            updated.stage = "Download cancelled".to_string();
            updated.updated_at = Utc::now();
            store.upsert_job(&updated)?;
            return Err(anyhow!("download cancelled"));
        }
        let read = response.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])?;
        downloaded += read as u64;
        updated.progress = if total > 0 {
            downloaded as f32 / total as f32
        } else {
            0.0
        };
        updated.stage = format!(
            "Downloading {} MB / {} MB",
            downloaded / (1 << 20),
            total / (1 << 20)
        );
        updated.updated_at = Utc::now();
        store.upsert_job(&updated)?;
    }

    fs::rename(&partial_path, &destination)?;
    let hash = hash_file(&destination)?;
    let model = store.import_download(source_label, &destination, &hash)?;
    updated.state = JobState::Completed;
    updated.progress = 1.0;
    updated.model_id = Some(model.id.clone());
    updated.stage = "Download complete".to_string();
    updated.updated_at = Utc::now();
    store.upsert_job(&updated)?;
    Ok(model)
}

fn persist_reconstructed_artifact(
    store: &SqliteModelStore,
    model_id: &str,
    original_hash: &str,
    prepared_path: &Path,
) -> Result<()> {
    if prepared_path.exists() {
        store.upsert_artifact(&ArtifactRecord {
            model_id: model_id.to_string(),
            kind: ArtifactKind::ReconstructedCache,
            path: prepared_path.to_path_buf(),
            size_bytes: fs::metadata(prepared_path)?.len() as i64,
            hash: hash_string(&format!(
                "reconstructed:{}:{}",
                original_hash,
                prepared_path.display()
            )),
            created_at: Utc::now(),
        })?;
    }
    Ok(())
}

fn source_kind_str(kind: &ModelSourceKind) -> &'static str {
    match kind {
        ModelSourceKind::Local => "local",
        ModelSourceKind::DirectUrl => "direct_url",
        ModelSourceKind::HuggingFace => "huggingface",
    }
}

fn parse_source_kind(value: &str) -> ModelSourceKind {
    match value {
        "direct_url" => ModelSourceKind::DirectUrl,
        "huggingface" => ModelSourceKind::HuggingFace,
        _ => ModelSourceKind::Local,
    }
}

fn model_status_str(status: &ModelStatus) -> &'static str {
    match status {
        ModelStatus::Imported => "imported",
        ModelStatus::Inspecting => "inspecting",
        ModelStatus::ReadyToConvert => "ready_to_convert",
        ModelStatus::Converting => "converting",
        ModelStatus::Converted => "converted",
        ModelStatus::Verified => "verified",
        ModelStatus::Launchable => "launchable",
        ModelStatus::Error => "error",
    }
}

fn parse_model_status(value: &str) -> ModelStatus {
    match value {
        "inspecting" => ModelStatus::Inspecting,
        "ready_to_convert" => ModelStatus::ReadyToConvert,
        "converting" => ModelStatus::Converting,
        "converted" => ModelStatus::Converted,
        "verified" => ModelStatus::Verified,
        "launchable" => ModelStatus::Launchable,
        "error" => ModelStatus::Error,
        _ => ModelStatus::Imported,
    }
}

fn artifact_kind_str(kind: &ArtifactKind) -> &'static str {
    match kind {
        ArtifactKind::OriginalGguf => "original_gguf",
        ArtifactKind::TritpackDir => "tritpack_dir",
        ArtifactKind::ReconstructedCache => "reconstructed_cache",
    }
}

fn parse_artifact_kind(value: &str) -> ArtifactKind {
    match value {
        "tritpack_dir" => ArtifactKind::TritpackDir,
        "reconstructed_cache" => ArtifactKind::ReconstructedCache,
        _ => ArtifactKind::OriginalGguf,
    }
}

fn job_kind_str(kind: &JobKind) -> &'static str {
    match kind {
        JobKind::Import => "import",
        JobKind::Download => "download",
        JobKind::Convert => "convert",
        JobKind::Verify => "verify",
        JobKind::Reconstruct => "reconstruct",
        JobKind::Chat => "chat",
    }
}

fn job_state_str(state: &JobState) -> &'static str {
    match state {
        JobState::Queued => "queued",
        JobState::Running => "running",
        JobState::Cancelled => "cancelled",
        JobState::Failed => "failed",
        JobState::Completed => "completed",
    }
}

fn parse_job_state(value: &str) -> JobState {
    match value {
        "running" => JobState::Running,
        "cancelled" => JobState::Cancelled,
        "failed" => JobState::Failed,
        "completed" => JobState::Completed,
        _ => JobState::Queued,
    }
}

fn parse_job_kind(value: &str) -> JobKind {
    match value {
        "download" => JobKind::Download,
        "convert" => JobKind::Convert,
        "verify" => JobKind::Verify,
        "reconstruct" => JobKind::Reconstruct,
        "chat" => JobKind::Chat,
        _ => JobKind::Import,
    }
}

fn parse_dt(value: String) -> rusqlite::Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(&value)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })
}

fn map_model_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ModelRecord> {
    Ok(ModelRecord {
        id: row.get(0)?,
        display_name: row.get(1)?,
        source: row.get(2)?,
        source_kind: parse_source_kind(&row.get::<_, String>(3)?),
        family: row.get(4)?,
        parameter_hint: row.get(5)?,
        license: row.get(6)?,
        status: parse_model_status(&row.get::<_, String>(7)?),
        created_at: parse_dt(row.get(8)?)?,
    })
}

fn map_artifact_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ArtifactRecord> {
    Ok(ArtifactRecord {
        model_id: row.get(0)?,
        kind: parse_artifact_kind(&row.get::<_, String>(1)?),
        path: PathBuf::from(row.get::<_, String>(2)?),
        size_bytes: row.get(3)?,
        hash: row.get(4)?,
        created_at: parse_dt(row.get(5)?)?,
    })
}

fn map_job_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<JobRecord> {
    Ok(JobRecord {
        id: row.get(0)?,
        kind: parse_job_kind(&row.get::<_, String>(1)?),
        state: parse_job_state(&row.get::<_, String>(2)?),
        model_id: row.get(3)?,
        progress: row.get(4)?,
        stage: row.get(5)?,
        cancelable: row.get::<_, i64>(6)? != 0,
        error: row.get(7)?,
        created_at: parse_dt(row.get(8)?)?,
        updated_at: parse_dt(row.get(9)?)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn deduplicates_imported_models_by_hash() -> Result<()> {
        let root = tempdir()?;
        let paths = AppPaths::from_root(root.path().join("state"))?;
        let store = SqliteModelStore::open(&paths.state_db)?;

        let sample_path = root.path().join("sample.gguf");
        fs::write(&sample_path, b"GGUFsample")?;
        let hash = hash_file(&sample_path)?;
        let managed_path = paths.models_original.join("sample.gguf");
        fs::copy(&sample_path, &managed_path)?;

        let first = store.import_local(&sample_path, &managed_path, &hash)?;
        let second = store.import_local(&sample_path, &managed_path, &hash)?;
        assert_eq!(first.id, second.id);
        assert_eq!(store.list_models()?.len(), 1);
        Ok(())
    }
}
