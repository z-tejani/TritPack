use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::Result;
use i_slint_backend_winit::WinitWindowAccessor;
use slint::{ModelRc, PhysicalPosition, SharedString, Timer, TimerMode, VecModel};
use tritpack_app_core::{DesktopApp, HuggingFaceFile};
use tritpack_runtime_api::{ConversionProfile, InferenceEvent, ModelRecord, RuntimeProfile};

slint::include_modules!();

pub fn run(app: Arc<DesktopApp>) -> Result<()> {
    let window = AppWindow::new()?;
    let library_items = std::rc::Rc::new(VecModel::<SharedString>::from(vec![]));
    let job_items = std::rc::Rc::new(VecModel::<SharedString>::from(vec![]));
    let search_items = std::rc::Rc::new(VecModel::<SharedString>::from(vec![]));
    let detail_items = std::rc::Rc::new(VecModel::<SharedString>::from(vec![]));
    let runtime_items = std::rc::Rc::new(VecModel::<SharedString>::from(vec![]));
    let chat_items = std::rc::Rc::new(VecModel::<SharedString>::from(vec![]));
    let chat_state = Arc::new(Mutex::new(Vec::<SharedString>::new()));
    let search_state = Arc::new(Mutex::new(Vec::<HuggingFaceFile>::new()));
    let model_state = Arc::new(Mutex::new(Vec::<ModelRecord>::new()));

    window.set_library_items(ModelRc::from(library_items.clone()));
    window.set_job_items(ModelRc::from(job_items.clone()));
    window.set_search_items(ModelRc::from(search_items.clone()));
    window.set_detail_items(ModelRc::from(detail_items.clone()));
    window.set_runtime_items(ModelRc::from(runtime_items.clone()));
    window.set_chat_items(ModelRc::from(chat_items.clone()));
    window.set_status_text(app.runtime_health()?.into());
    runtime_items.set_vec(
        app.runtime_detail_summary()?
            .into_iter()
            .map(Into::into)
            .collect::<Vec<SharedString>>(),
    );
    if let Some(token) = app.load_huggingface_token()? {
        window.set_hf_token(token.into());
    }

    {
        let weak = window.as_weak();
        let _ = slint::spawn_local(async move {
            if let Some(window) = weak.upgrade() {
                if let Ok(winit_window) = window.window().winit_window().await {
                    if let Some(monitor) = winit_window.current_monitor() {
                        let monitor_size = monitor.size();
                        let outer_size = winit_window.outer_size();
                        let centered_x = monitor.position().x
                            + ((monitor_size.width as i32 - outer_size.width as i32) / 2).max(0);
                        let centered_y = monitor.position().y
                            + ((monitor_size.height as i32 - outer_size.height as i32) / 2).max(0);
                        window
                            .window()
                            .set_position(PhysicalPosition::new(centered_x, centered_y));
                    }
                }
            }
        });
    }

    refresh_models(&app, &window, &library_items, &job_items, &model_state)?;

    let refresh_timer = Timer::default();
    let weak = window.as_weak();
    let app_for_timer = Arc::clone(&app);
    let library_for_timer = library_items.clone();
    let jobs_for_timer = job_items.clone();
    let runtime_for_timer = runtime_items.clone();
    let model_state_for_timer = Arc::clone(&model_state);
    refresh_timer.start(
        TimerMode::Repeated,
        std::time::Duration::from_secs(2),
        move || {
            if let Some(window) = weak.upgrade() {
                let _ = refresh_models(
                    &app_for_timer,
                    &window,
                    &library_for_timer,
                    &jobs_for_timer,
                    &model_state_for_timer,
                );
                let _ = refresh_runtime(&app_for_timer, &window, &runtime_for_timer);
            }
        },
    );

    let weak = window.as_weak();
    let app_for_refresh = Arc::clone(&app);
    let library_for_refresh = library_items.clone();
    let jobs_for_refresh = job_items.clone();
    let runtime_for_refresh = runtime_items.clone();
    let model_state_for_refresh = Arc::clone(&model_state);
    window.on_refresh(move || {
        if let Some(window) = weak.upgrade() {
            let _ = refresh_models(
                &app_for_refresh,
                &window,
                &library_for_refresh,
                &jobs_for_refresh,
                &model_state_for_refresh,
            );
            let _ = refresh_runtime(&app_for_refresh, &window, &runtime_for_refresh);
        }
    });

    let weak = window.as_weak();
    let app_for_runtime = Arc::clone(&app);
    let runtime_for_runtime = runtime_items.clone();
    window.on_refresh_runtime(move || {
        if let Some(window) = weak.upgrade() {
            let _ = refresh_runtime(&app_for_runtime, &window, &runtime_for_runtime);
        }
    });

    let weak = window.as_weak();
    window.on_close_window(move || {
        if let Some(window) = weak.upgrade() {
            let _ = window.window().hide();
        }
    });

    let weak = window.as_weak();
    window.on_minimize_window(move || {
        if let Some(window) = weak.upgrade() {
            window.window().set_minimized(true);
        }
    });

    let weak = window.as_weak();
    window.on_toggle_maximize_window(move || {
        if let Some(window) = weak.upgrade() {
            let slint_window = window.window();
            let maximized = slint_window.is_maximized();
            slint_window.set_maximized(!maximized);
        }
    });

    let weak = window.as_weak();
    window.on_begin_window_drag(move || {
        if let Some(window) = weak.upgrade() {
            let _ = window
                .window()
                .with_winit_window(|winit_window| winit_window.drag_window());
        }
    });

    let weak = window.as_weak();
    window.on_show_about(move || {
        if let Some(window) = weak.upgrade() {
            window.set_selected_view(5);
            window.set_status_text("TritPack Desktop alpha".into());
        }
    });

    window.on_quit_app(move || {
        let _ = slint::quit_event_loop();
    });

    let weak = window.as_weak();
    window.on_open_project_guide(move || {
        let target = repo_root().join("README.md");
        let result = open_path(&target);
        if let Some(window) = weak.upgrade() {
            match result {
                Ok(_) => window.set_status_text("Opened project guide".into()),
                Err(error) => {
                    window.set_status_text(format!("Could not open guide: {error}").into())
                }
            }
        }
    });

    let weak = window.as_weak();
    window.on_open_local_models_guide(move || {
        let target = repo_root().join("LOCAL_MODELS_GUIDE.md");
        let result = open_path(&target);
        if let Some(window) = weak.upgrade() {
            match result {
                Ok(_) => window.set_status_text("Opened local models guide".into()),
                Err(error) => window
                    .set_status_text(format!("Could not open local models guide: {error}").into()),
            }
        }
    });

    let weak = window.as_weak();
    let app_for_cancel = Arc::clone(&app);
    let library_for_cancel = library_items.clone();
    let jobs_for_cancel = job_items.clone();
    let model_state_for_cancel = Arc::clone(&model_state);
    window.on_cancel_job(move |job_id| {
        if job_id.trim().is_empty() {
            return;
        }
        if let Some(window) = weak.upgrade() {
            match app_for_cancel.cancel_job(job_id.trim()) {
                Ok(_) => window.set_status_text(
                    format!("Cancellation requested for {}", job_id.trim()).into(),
                ),
                Err(error) => window.set_status_text(format!("Cancel failed: {error}").into()),
            }
            let _ = refresh_models(
                &app_for_cancel,
                &window,
                &library_for_cancel,
                &jobs_for_cancel,
                &model_state_for_cancel,
            );
        }
    });

    let weak = window.as_weak();
    let app_for_import = Arc::clone(&app);
    let library_for_import = library_items.clone();
    let jobs_for_import = job_items.clone();
    let model_state_for_import = Arc::clone(&model_state);
    window.on_import_local(move |path| {
        if path.trim().is_empty() {
            return;
        }
        let result = app_for_import.import_local(path.trim());
        if let Some(window) = weak.upgrade() {
            match result {
                Ok(model) => {
                    window.set_status_text(format!("Imported {}", model.display_name).into())
                }
                Err(error) => window.set_status_text(format!("Import failed: {error}").into()),
            }
            let _ = refresh_models(
                &app_for_import,
                &window,
                &library_for_import,
                &jobs_for_import,
                &model_state_for_import,
            );
        }
    });

    let weak = window.as_weak();
    let app_for_convert = Arc::clone(&app);
    window.on_queue_convert(move |model_id| {
        if model_id.trim().is_empty() {
            return;
        }
        if let Some(window) = weak.upgrade() {
            match app_for_convert.queue_conversion(model_id.trim(), ConversionProfile::default()) {
                Ok(job_id) => {
                    window.set_status_text(format!("Queued conversion job {}", &job_id[..8]).into())
                }
                Err(error) => {
                    window.set_status_text(format!("Conversion queue failed: {error}").into())
                }
            }
        }
    });

    let weak = window.as_weak();
    let app_for_library_select = Arc::clone(&app);
    let detail_for_library_select = detail_items.clone();
    let model_state_for_select = Arc::clone(&model_state);
    window.on_select_library_item(move |index| {
        let Ok(index) = usize::try_from(index) else {
            return;
        };
        let selected = {
            let state = model_state_for_select.lock().expect("model state poisoned");
            state.get(index).cloned()
        };
        if let Some(window) = weak.upgrade() {
            match selected {
                Some(model) => {
                    window.set_selected_library_index(index as i32);
                    window.set_model_id_input(model.id.clone().into());
                    window.set_chat_model_id(model.id.clone().into());
                    match app_for_library_select.model_detail_summary(&model.id) {
                        Ok(lines) => {
                            detail_for_library_select.set_vec(
                                lines
                                    .into_iter()
                                    .map(Into::into)
                                    .collect::<Vec<SharedString>>(),
                            );
                            window
                                .set_status_text(format!("Selected {}", model.display_name).into());
                        }
                        Err(error) => window.set_status_text(
                            format!(
                                "Selected {}, but detail failed: {error}",
                                model.display_name
                            )
                            .into(),
                        ),
                    }
                }
                None => window.set_status_text("Model selection out of range".into()),
            }
        }
    });

    let weak = window.as_weak();
    let app_for_prepare = Arc::clone(&app);
    let library_for_prepare = library_items.clone();
    let jobs_for_prepare = job_items.clone();
    let model_state_for_prepare = Arc::clone(&model_state);
    window.on_prepare_runtime(move |model_id| {
        if model_id.trim().is_empty() {
            return;
        }
        if let Some(window) = weak.upgrade() {
            match app_for_prepare.queue_prepare_runtime(model_id.trim(), RuntimeProfile::default())
            {
                Ok(job_id) => window
                    .set_status_text(format!("Queued runtime prep job {}", &job_id[..8]).into()),
                Err(error) => {
                    window.set_status_text(format!("Runtime preparation failed: {error}").into())
                }
            }
            let _ = refresh_models(
                &app_for_prepare,
                &window,
                &library_for_prepare,
                &jobs_for_prepare,
                &model_state_for_prepare,
            );
        }
    });

    let weak = window.as_weak();
    let app_for_inspect = Arc::clone(&app);
    let detail_for_inspect = detail_items.clone();
    let library_for_inspect = library_items.clone();
    let jobs_for_inspect = job_items.clone();
    let model_state_for_inspect = Arc::clone(&model_state);
    window.on_inspect_model(move |model_id| {
        if model_id.trim().is_empty() {
            return;
        }
        if let Some(window) = weak.upgrade() {
            match app_for_inspect.model_detail_summary(model_id.trim()) {
                Ok(lines) => {
                    detail_for_inspect.set_vec(
                        lines
                            .into_iter()
                            .map(Into::into)
                            .collect::<Vec<SharedString>>(),
                    );
                    window.set_status_text("Loaded model detail".into());
                }
                Err(error) => window.set_status_text(format!("Inspect failed: {error}").into()),
            }
            let _ = refresh_models(
                &app_for_inspect,
                &window,
                &library_for_inspect,
                &jobs_for_inspect,
                &model_state_for_inspect,
            );
        }
    });

    let weak = window.as_weak();
    let app_for_search = Arc::clone(&app);
    let search_for_search = search_items.clone();
    let search_state_for_search = Arc::clone(&search_state);
    window.on_search_models(move |query| {
        if let Some(window) = weak.upgrade() {
            match app_for_search.search_huggingface(query.trim()) {
                Ok(results) => {
                    *search_state_for_search
                        .lock()
                        .expect("search state poisoned") = results.clone();
                    fill_results(&search_for_search, &results);
                    window.set_selected_search_index(-1);
                    window.set_search_index("".into());
                    window
                        .set_status_text(format!("Found {} GGUF artifacts", results.len()).into());
                }
                Err(error) => window.set_status_text(format!("Search failed: {error}").into()),
            }
        }
    });

    let weak = window.as_weak();
    let search_state_for_select = Arc::clone(&search_state);
    window.on_select_search_item(move |index| {
        let Ok(index) = usize::try_from(index) else {
            return;
        };
        let selected = {
            let state = search_state_for_select
                .lock()
                .expect("search state poisoned");
            state.get(index).cloned()
        };
        if let Some(window) = weak.upgrade() {
            match selected {
                Some(file) => {
                    window.set_selected_search_index(index as i32);
                    window.set_search_index((index + 1).to_string().into());
                    window.set_status_text(
                        format!("Selected result #{}: {}", index + 1, file.filename).into(),
                    );
                }
                None => window.set_status_text("Search result selection out of range".into()),
            }
        }
    });

    let weak = window.as_weak();
    let app_for_hf_download = Arc::clone(&app);
    let search_state_for_download = Arc::clone(&search_state);
    let library_for_hf_download = library_items.clone();
    let jobs_for_hf_download = job_items.clone();
    let model_state_for_hf_download = Arc::clone(&model_state);
    window.on_download_search_result(move |index| {
        let Ok(index) = index.trim().parse::<usize>() else {
            if let Some(window) = weak.upgrade() {
                window.set_status_text("Enter a numeric search result index".into());
            }
            return;
        };
        let index = index.saturating_sub(1);

        let selected = {
            let state = search_state_for_download
                .lock()
                .expect("search state poisoned");
            state.get(index).cloned()
        };

        if let Some(window) = weak.upgrade() {
            match selected {
                Some(file) => match app_for_hf_download.queue_download_huggingface_file(&file) {
                    Ok(job_id) => window.set_status_text(
                        format!("Queued download job {} for {}", &job_id[..8], file.repo_id).into(),
                    ),
                    Err(error) => {
                        window.set_status_text(format!("Download failed: {error}").into())
                    }
                },
                None => window.set_status_text("Search result index out of range".into()),
            }
            let _ = refresh_models(
                &app_for_hf_download,
                &window,
                &library_for_hf_download,
                &jobs_for_hf_download,
                &model_state_for_hf_download,
            );
        }
    });

    let weak = window.as_weak();
    let app_for_download = Arc::clone(&app);
    let library_for_download = library_items.clone();
    let jobs_for_download = job_items.clone();
    let model_state_for_download = Arc::clone(&model_state);
    window.on_download_url(move |url| {
        if url.trim().is_empty() {
            return;
        }
        if let Some(window) = weak.upgrade() {
            match app_for_download.queue_download_from_url(url.trim()) {
                Ok(job_id) => {
                    window.set_status_text(format!("Queued download job {}", &job_id[..8]).into())
                }
                Err(error) => window.set_status_text(format!("Download failed: {error}").into()),
            }
            let _ = refresh_models(
                &app_for_download,
                &window,
                &library_for_download,
                &jobs_for_download,
                &model_state_for_download,
            );
        }
    });

    let weak = window.as_weak();
    let app_for_token = Arc::clone(&app);
    window.on_save_token(move |token| {
        if let Some(window) = weak.upgrade() {
            match app_for_token.save_huggingface_token(token.trim()) {
                Ok(_) => window.set_status_text("Saved Hugging Face token".into()),
                Err(error) => {
                    window.set_status_text(format!("Failed to save token: {error}").into())
                }
            }
        }
    });

    let weak = window.as_weak();
    let app_for_cache = Arc::clone(&app);
    let library_for_cache = library_items.clone();
    let jobs_for_cache = job_items.clone();
    let model_state_for_cache = Arc::clone(&model_state);
    window.on_clear_cache(move |model_id| {
        if model_id.trim().is_empty() {
            return;
        }
        if let Some(window) = weak.upgrade() {
            match app_for_cache.remove_reconstructed_cache(model_id.trim()) {
                Ok(_) => window.set_status_text("Cleared reconstructed cache".into()),
                Err(error) => {
                    window.set_status_text(format!("Failed to clear cache: {error}").into())
                }
            }
            let _ = refresh_models(
                &app_for_cache,
                &window,
                &library_for_cache,
                &jobs_for_cache,
                &model_state_for_cache,
            );
        }
    });

    let weak = window.as_weak();
    let app_for_chat = Arc::clone(&app);
    let chat_for_chat = chat_items.clone();
    let chat_state_for_chat = Arc::clone(&chat_state);
    window.on_send_prompt(move |model_id, prompt| {
        if prompt.trim().is_empty() {
            return;
        }

        if model_id.trim().is_empty() {
            if let Some(window) = weak.upgrade() {
                window.set_selected_view(1);
                window.set_status_text("Select a model in Models before chatting".into());
            }
            return;
        }

        {
            let mut state = chat_state_for_chat.lock().expect("chat state poisoned");
            state.push(format!("User: {}", prompt.trim()).into());
            state.push("Assistant:".into());
            chat_for_chat.set_vec(state.clone());
        }

        if let Some(window) = weak.upgrade() {
            window.set_chat_prompt("".into());
            window.set_status_text("Starting generation".into());
        }

        let app = Arc::clone(&app_for_chat);
        let weak = weak.clone();
        let chat_state_for_thread = Arc::clone(&chat_state_for_chat);
        let model_id = model_id.trim().to_string();
        let prompt = prompt.trim().to_string();
        thread::spawn(move || {
            match app.generate_stream(&model_id, prompt.clone(), RuntimeProfile::default()) {
                Ok(receiver) => {
                    while let Ok(event) = receiver.recv() {
                        let chat_state = Arc::clone(&chat_state_for_thread);
                        let weak = weak.clone();
                        match event {
                            InferenceEvent::Token(token) => {
                                let _ = slint::invoke_from_event_loop(move || {
                                    let mut state = chat_state.lock().expect("chat state poisoned");
                                    if state.is_empty() {
                                        state.push("Assistant:".into());
                                    }
                                    let last = state.pop().unwrap_or_default();
                                    state.push(format!("{last}{token}").into());
                                    if let Some(window) = weak.upgrade() {
                                        let model = std::rc::Rc::new(VecModel::from(state.clone()));
                                        window.set_chat_items(ModelRc::from(model));
                                        window.set_status_text("Streaming response".into());
                                    }
                                });
                            }
                            InferenceEvent::Completed { .. } => {
                                let _ = slint::invoke_from_event_loop(move || {
                                    if let Some(window) = weak.upgrade() {
                                        window.set_status_text("Generation complete".into());
                                    }
                                });
                                break;
                            }
                            InferenceEvent::Error(error) => {
                                let _ = slint::invoke_from_event_loop(move || {
                                    if let Some(window) = weak.upgrade() {
                                        window.set_status_text(
                                            format!("Generation failed: {error}").into(),
                                        );
                                    }
                                });
                                break;
                            }
                            InferenceEvent::Started { .. } | InferenceEvent::Log(_) => {}
                        }
                    }
                }
                Err(error) => {
                    let _ = slint::invoke_from_event_loop(move || {
                        if let Some(window) = weak.upgrade() {
                            window.set_status_text(
                                format!("Could not start generation: {error}").into(),
                            );
                        }
                    });
                }
            }
        });
    });

    window.run()?;
    Ok(())
}

fn refresh_models(
    app: &DesktopApp,
    window: &AppWindow,
    library_items: &VecModel<SharedString>,
    job_items: &VecModel<SharedString>,
    model_state: &Arc<Mutex<Vec<ModelRecord>>>,
) -> Result<()> {
    *model_state.lock().expect("model state poisoned") = app.list_models()?;
    library_items.set_vec(
        app.list_model_summaries()?
            .into_iter()
            .map(Into::into)
            .collect::<Vec<SharedString>>(),
    );
    job_items.set_vec(
        app.list_job_summaries()?
            .into_iter()
            .map(Into::into)
            .collect::<Vec<SharedString>>(),
    );
    window.set_status_text(app.runtime_health()?.into());
    Ok(())
}

fn refresh_runtime(
    app: &DesktopApp,
    window: &AppWindow,
    runtime_items: &VecModel<SharedString>,
) -> Result<()> {
    runtime_items.set_vec(
        app.runtime_detail_summary()?
            .into_iter()
            .map(Into::into)
            .collect::<Vec<SharedString>>(),
    );
    window.set_status_text(app.runtime_health()?.into());
    Ok(())
}

fn fill_results(model: &VecModel<SharedString>, results: &[HuggingFaceFile]) {
    model.set_vec(
        results
            .iter()
            .enumerate()
            .map(|(index, result)| {
                format!(
                    "#{} | {} / {} | {} MB | {}",
                    index + 1,
                    result.repo_id,
                    result.filename,
                    result.size_bytes.unwrap_or_default() / (1 << 20),
                    result.download_url
                )
                .into()
            })
            .collect::<Vec<SharedString>>(),
    );
}

fn repo_root() -> PathBuf {
    let fallback = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    fallback.canonicalize().unwrap_or(fallback)
}

fn open_path(path: &std::path::Path) -> Result<()> {
    Command::new("open").arg(path).status()?;
    Ok(())
}
