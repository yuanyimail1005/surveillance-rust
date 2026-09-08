mod face;

use std::fmt;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Default, Clone, Copy)]
struct WatchdogCleanupStats {
    processes_cleaned: usize,
    files_cleaned: usize,
}

use anyhow::{Context, Result as AnyhowResult};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use dotenvy::dotenv;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::fs;
use tokio::net::UdpSocket;
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;
use webrtc::{
    api::{APIBuilder, media_engine::MediaEngine},
    data_channel::RTCDataChannel,
    ice_transport::ice_server::RTCIceServer,
    peer_connection::{
        configuration::RTCConfiguration,
        peer_connection_state::RTCPeerConnectionState,
        sdp::session_description::RTCSessionDescription,
        RTCPeerConnection,
    },
    rtp::packet::Packet,
    rtp_transceiver::rtp_codec::{RTCRtpCodecCapability, RTPCodecType},
    track::track_local::{
        track_local_static_rtp::TrackLocalStaticRTP,
        TrackLocal, TrackLocalWriter,
    },
};
use webrtc_util::{Marshal, MarshalSize, Unmarshal};

const TALKBACK_RTP_PORT: u16 = 5010;

#[derive(Debug, Clone)]
struct Config {
    server_host: String,
    server_port: u16,
    ssl_cert_path: String,
    ssl_key_path: String,
    camera_device: String,
    camera_width: i32,
    camera_height: i32,
    camera_fps: i32,
    pulse_sink_name: String,
    pulse_capture_source_name: String,
    face_recognition_enabled: bool,
    face_recognition_known_faces_dir: String,
    face_recognition_match_threshold: f64,
    face_recognition_max_faces: i32,
    face_recognition_detect_every_n_frames: i32,
    face_recognition_cascade_path: String,
    face_recognition_min_consecutive_frames: i32,
}

impl Config {
    fn from_env() -> Self {
        dotenv().ok();

        let default_camera_device = select_default_camera_device();

        Self {
            server_host: std::env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            server_port: std::env::var("SERVER_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5000),
            ssl_cert_path: std::env::var("SSL_CERT_PATH")
                .unwrap_or_else(|_| "${HOME}/certs/cert.pem".to_string())
                .replace("${HOME}", std::env::var("HOME").unwrap_or_default().as_str()),
            ssl_key_path: std::env::var("SSL_KEY_PATH")
                .unwrap_or_else(|_| "${HOME}/certs/key.pem".to_string())
                .replace("${HOME}", std::env::var("HOME").unwrap_or_default().as_str()),
            camera_device: std::env::var("CAMERA_DEVICE")
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .unwrap_or(default_camera_device),
            camera_width: std::env::var("CAMERA_WIDTH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1920),
            camera_height: std::env::var("CAMERA_HEIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1080),
            camera_fps: std::env::var("CAMERA_FPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(25),
            pulse_sink_name: std::env::var("PULSE_SINK_NAME").unwrap_or_else(|_| "@DEFAULT_SINK@".to_string()),
            pulse_capture_source_name: std::env::var("PULSE_CAPTURE_SOURCE_NAME")
                .unwrap_or_else(|_| "@DEFAULT_SOURCE@".to_string()),
            face_recognition_enabled: std::env::var("FACE_RECOGNITION_ENABLED")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(false),
            face_recognition_known_faces_dir: std::env::var("FACE_RECOGNITION_KNOWN_FACES_DIR")
                .unwrap_or_else(|_| format!("{}/known_faces", std::env::var("HOME").unwrap_or_default())),
            face_recognition_match_threshold: std::env::var("FACE_RECOGNITION_MATCH_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.6),
            face_recognition_max_faces: std::env::var("FACE_RECOGNITION_MAX_FACES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8),
            face_recognition_detect_every_n_frames: std::env::var("FACE_RECOGNITION_DETECT_EVERY_N_FRAMES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
            face_recognition_cascade_path: std::env::var("FACE_RECOGNITION_CASCADE_PATH")
                .unwrap_or_default(),
            face_recognition_min_consecutive_frames: std::env::var("FACE_RECOGNITION_MIN_CONSECUTIVE_FRAMES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
        }
    }
}

fn select_default_camera_device() -> String {
    for candidate in list_video_nodes() {
        if is_supported_v4l2_camera(&candidate) {
            return candidate;
        }
    }
    if rpicam_camera_available() {
        if let Some((index, _)) = rpicam_camera_indices().first() {
            return format!("rpicam://{index}");
        }
    }
    "/dev/video0".to_string()
}

fn rpicam_camera_available() -> bool {
    command_available("rpicam-hello")
        && command_available("rpicam-vid")
        && !rpicam_camera_indices().is_empty()
}

fn command_available(command: &str) -> bool {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .map(|path| path.join(command))
        .any(|path| path.is_file())
}

fn rpicam_camera_indices() -> Vec<(u32, String)> {
    let Ok(output) = Command::new("rpicam-hello").args(["--list-cameras"]).output() else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    let mut cameras = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let (index, name) = line.trim().split_once(':')?;
            let index = index.trim().parse::<u32>().ok()?;
            let name = name.trim().split('[').next().unwrap_or(name).trim();
            (!name.is_empty()).then(|| (index, name.to_string()))
        })
        .collect::<Vec<_>>();
    cameras.sort_by_key(|(index, _)| *index);
    cameras.dedup_by_key(|(index, _)| *index);
    cameras
}

fn rpicam_index(camera_device: &str) -> Option<String> {
    camera_device.trim().strip_prefix("rpicam://")
        .filter(|index| !index.is_empty() && index.chars().all(|ch| ch.is_ascii_digit()))
        .map(str::to_string)
}

fn list_video_nodes() -> Vec<String> {
    let mut nodes = std::fs::read_dir("/dev")
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            let suffix = name.strip_prefix("video")?;
            suffix.parse::<u32>().ok()?;
            Some(format!("/dev/{name}"))
        })
        .collect::<Vec<_>>();
    nodes.sort_by_key(|path| {
        path.strip_prefix("/dev/video")
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(u32::MAX)
    });
    nodes
}

fn is_supported_v4l2_camera(device_path: &str) -> bool {
    if !std::path::Path::new(device_path).exists() {
        return false;
    }

    let Ok(capabilities) = Command::new("v4l2-ctl")
        .args(["-d", device_path, "--all"])
        .output()
    else {
        return true;
    };
    if !capabilities.status.success() {
        return false;
    }
    let capabilities = String::from_utf8_lossy(&capabilities.stdout).to_lowercase();
    if capabilities.contains("metadata capture") && !capabilities.contains("video capture") {
        return false;
    }
    if !capabilities.contains("video capture") {
        return false;
    }

    let Ok(formats) = Command::new("v4l2-ctl")
        .args(["-d", device_path, "--list-formats-ext"])
        .output()
    else {
        return false;
    };
    if !formats.status.success() {
        return false;
    }
    let formats = String::from_utf8_lossy(&formats.stdout).to_lowercase();
    formats.contains("mjpg") || formats.contains("motion-jpeg")
}

fn available_camera_devices() -> Vec<Value> {
    let resolutions = json!([
        {"width": 640, "height": 480},
        {"width": 1280, "height": 720},
        {"width": 1920, "height": 1080},
        {"width": 2560, "height": 1440}
    ]);
    let mut devices = Vec::new();

    for path in list_video_nodes() {
        if !is_supported_v4l2_camera(&path) {
            continue;
        }
        devices.push(json!({
            "path": path,
            "name": path,
            "supported": true,
            "reason": "v4l2",
            "supported_resolutions": resolutions
        }));
    }

    if command_available("rpicam-hello") && command_available("rpicam-vid") {
        for (index, name) in rpicam_camera_indices() {
            devices.push(json!({
                "path": format!("rpicam://{index}"),
                "name": format!("CSI Camera {index}: {name}"),
                "supported": true,
                "reason": "CSI camera via rpicam-vid mjpeg stream",
                "supported_resolutions": resolutions
            }));
        }
    }

    devices
}

fn camera_source_type_for_device(camera_device: &str) -> &'static str {
    if camera_device.starts_with("rpicam://") {
        "CSI"
    } else if camera_device.starts_with("/dev/video") {
        "V4L2"
    } else {
        "Unknown"
    }
}

#[derive(Debug, Clone, Default)]
struct CameraState {
    width: i32,
    height: i32,
    fps: i32,
    camera_device: String,
}

#[derive(Debug, Clone)]
struct AudioState {
    selected_microphone: String,
    selected_speaker: String,
    volume: i32,
}

impl Default for AudioState {
    fn default() -> Self {
        Self {
            selected_microphone: "@DEFAULT_SOURCE@".to_string(),
            selected_speaker: "@DEFAULT_SINK@".to_string(),
            volume: 70,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct FaceState {
    enabled: bool,
    available: bool,
    initializing: bool,
    backend: String,
    message: String,
    known_faces_count: i32,
    detect_every_n_frames: i32,
    match_threshold: f64,
    max_faces: i32,
    result: FaceResult,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct FaceResult {
    updated_at: i64,
    frame_index: i64,
    broadcast_frame_seq: i64,
    image_width: i32,
    image_height: i32,
    faces: Vec<FaceDetection>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct FaceDetection {
    name: String,
    confidence: f64,
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

#[derive(Debug, Clone)]
struct Manager {
    camera: CameraState,
    audio: AudioState,
    face: FaceState,
}

#[derive(Clone)]
struct SharedMediaPipeline {
    video_ffmpeg: Arc<Mutex<Option<std::process::Child>>>,
    video_encoder_stdin: Arc<Mutex<Option<std::process::ChildStdin>>>,
    camera_capture: Arc<Mutex<Option<std::process::Child>>>,
    camera_pipe_thread: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
    audio_ffmpeg: Arc<Mutex<Option<std::process::Child>>>,
    talkback_ffmpeg: Arc<Mutex<Option<std::process::Child>>>,
    video_rtp_port: u16,
    audio_rtp_port: u16,
    created_at_unix_ms: u128,
    video_packets_in: Arc<AtomicU64>,
    audio_packets_in: Arc<AtomicU64>,
    video_packets_written_by_session: Arc<AsyncMutex<HashMap<u64, u64>>>,
    audio_packets_written_by_session: Arc<AsyncMutex<HashMap<u64, u64>>>,
    metadata_channels: Arc<AsyncMutex<Vec<(u64, String, Arc<RTCDataChannel>)>>>,
    video_frame_seq: Arc<AtomicU64>,
    talkback_sdp_path: PathBuf,
    video_task: Arc<Mutex<Option<JoinHandle<()>>>>,
    audio_task: Arc<Mutex<Option<JoinHandle<()>>>>,
    face_reader_thread: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
    video_tracks: Arc<AsyncMutex<Vec<(u64, Arc<TrackLocalStaticRTP>)>>>,
    audio_tracks: Arc<AsyncMutex<Vec<(u64, Arc<TrackLocalStaticRTP>)>>>,
}

impl SharedMediaPipeline {
    async fn start(config: &Config, face_svc: Arc<face::FaceService>) -> AnyhowResult<Self> {
        let cleaned_workers = cleanup_orphan_media_workers();
        if cleaned_workers > 0 {
            tracing::info!("shared media startup cleanup: killed {} stale media worker process(es)", cleaned_workers);
        }

        let video_socket = UdpSocket::bind("127.0.0.1:0").await?;
        let video_port = video_socket.local_addr()?.port();
        let audio_socket = UdpSocket::bind("127.0.0.1:0").await?;
        let audio_port = audio_socket.local_addr()?.port();
        let created_at_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);

        let video_tracks = Arc::new(AsyncMutex::new(Vec::new()));
        let audio_tracks = Arc::new(AsyncMutex::new(Vec::new()));
        let video_packets_in = Arc::new(AtomicU64::new(0));
        let audio_packets_in = Arc::new(AtomicU64::new(0));
        let video_packets_written_by_session = Arc::new(AsyncMutex::new(HashMap::new()));
        let audio_packets_written_by_session = Arc::new(AsyncMutex::new(HashMap::new()));
        let metadata_channels = Arc::new(AsyncMutex::new(Vec::new()));
        let video_frame_seq = Arc::new(AtomicU64::new(0));

        let (video_child, encoder_stdin, face_pipe) = start_video_encoder(config, video_port)?;
        let video_ffmpeg = Arc::new(Mutex::new(Some(video_child)));
        let video_encoder_stdin = Arc::new(Mutex::new(Some(encoder_stdin)));
        let (camera, camera_stdout) = start_camera_capture(config)?;
        let camera_capture = Arc::new(Mutex::new(Some(camera)));
        let camera_pipe_thread = Arc::new(Mutex::new(Some(spawn_camera_pipe(
            camera_stdout,
            video_encoder_stdin.clone(),
        ))));
        let audio_ffmpeg = Arc::new(Mutex::new(Some(start_audio_ffmpeg(&config.pulse_capture_source_name, audio_port)?)));
        let talkback_sdp_path = write_talkback_sdp(TALKBACK_RTP_PORT)?;
        let talkback_ffmpeg = Arc::new(Mutex::new(Some(start_talkback_ffmpeg(&talkback_sdp_path, &config.pulse_sink_name)?)));

        // Spawn MJPEG reader thread: feeds face recognition service at 2 fps
        let face_reader_thread = Arc::new(Mutex::new(Some(
            face::spawn_mjpeg_reader(face_pipe, move |jpeg, seq| {
                face_svc.process_frame(jpeg, seq);
                true // keep running
            })
        )));

        let video_task = Arc::new(Mutex::new(Some(start_rtp_reader(
            video_socket,
            video_tracks.clone(),
            video_packets_in.clone(),
            video_packets_written_by_session.clone(),
            Some(metadata_channels.clone()),
            Some(video_frame_seq.clone()),
        ).await?)));
        let audio_task = Arc::new(Mutex::new(Some(start_rtp_reader(
            audio_socket,
            audio_tracks.clone(),
            audio_packets_in.clone(),
            audio_packets_written_by_session.clone(),
            None,
            None,
        ).await?)));

        Ok(Self {
            video_ffmpeg,
            video_encoder_stdin,
            camera_capture,
            camera_pipe_thread,
            audio_ffmpeg,
            talkback_ffmpeg,
            video_rtp_port: video_port,
            audio_rtp_port: audio_port,
            created_at_unix_ms,
            video_packets_in,
            audio_packets_in,
            video_packets_written_by_session,
            audio_packets_written_by_session,
            metadata_channels,
            video_frame_seq,
            talkback_sdp_path,
            video_task,
            audio_task,
            face_reader_thread,
            video_tracks,
            audio_tracks,
        })
    }

    fn child_running(child: &Arc<Mutex<Option<std::process::Child>>>) -> bool {
        let mut guard = child.lock().unwrap();
        if let Some(process) = guard.as_mut() {
            process.try_wait().ok().flatten().is_none()
        } else {
            false
        }
    }

    fn has_live_media_workers(&self) -> bool {
        Self::child_running(&self.video_ffmpeg)
            && Self::child_running(&self.audio_ffmpeg)
            && (self.camera_capture.lock().unwrap().is_none()
                || Self::child_running(&self.camera_capture))
    }

    async fn register_tracks(&self, session_id: u64, video_track: Arc<TrackLocalStaticRTP>, audio_track: Arc<TrackLocalStaticRTP>) {
        self.video_tracks.lock().await.push((session_id, video_track));
        self.audio_tracks.lock().await.push((session_id, audio_track));
    }

    async fn unregister_tracks(&self, session_id: u64) {
        self.video_tracks.lock().await.retain(|(sid, _)| *sid != session_id);
        self.audio_tracks.lock().await.retain(|(sid, _)| *sid != session_id);
        self.video_packets_written_by_session.lock().await.remove(&session_id);
        self.audio_packets_written_by_session.lock().await.remove(&session_id);
        self.metadata_channels.lock().await.retain(|(sid, _, _)| *sid != session_id);
    }

    async fn register_metadata_channel(&self, session_id: u64, label: String, dc: Arc<RTCDataChannel>) {
        let mut channels = self.metadata_channels.lock().await;
        channels.retain(|(sid, existing_label, _)| !(*sid == session_id && *existing_label == label));
        channels.push((session_id, label, dc));
    }

    async fn unregister_metadata_channel(&self, session_id: u64, label: &str) {
        self.metadata_channels
            .lock()
            .await
            .retain(|(sid, existing_label, _)| !(*sid == session_id && existing_label == label));
    }

    async fn restart_camera(
        &self,
        config: &Config,
        _face_svc: Arc<face::FaceService>,
    ) -> AnyhowResult<()> {
        if let Some(mut child) = self.camera_capture.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(thread) = self.camera_pipe_thread.lock().unwrap().take() {
            let _ = thread.join();
        }

        let (camera, camera_stdout) = start_camera_capture(config)?;
        *self.camera_capture.lock().unwrap() = Some(camera);
        *self.camera_pipe_thread.lock().unwrap() = Some(spawn_camera_pipe(
            camera_stdout,
            self.video_encoder_stdin.clone(),
        ));
        if !Self::child_running(&self.video_ffmpeg)
            || !Self::child_running(&self.camera_capture)
        {
            return Err(anyhow::anyhow!(
                "camera capture process exited while starting {}",
                config.camera_device
            ));
        }

        Ok(())
    }

    async fn shutdown(&self) {
        if let Some(mut child) = self.camera_capture.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(thread) = self.camera_pipe_thread.lock().unwrap().take() {
            let _ = thread.join();
        }
        self.video_encoder_stdin.lock().unwrap().take();

        if let Some(mut child) = self.video_ffmpeg.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(mut child) = self.audio_ffmpeg.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(mut child) = self.talkback_ffmpeg.lock().unwrap().take() {
            let _ = child.kill();
            let _ = child.wait();
        }

        if let Some(task) = self.video_task.lock().unwrap().take() {
            task.abort();
        }
        if let Some(task) = self.audio_task.lock().unwrap().take() {
            task.abort();
        }
        // Face reader thread will exit when the pipe closes (ffmpeg killed above).
        if let Some(t) = self.face_reader_thread.lock().unwrap().take() {
            let _ = t.join();
        }

        let _ = std::fs::remove_file(&self.talkback_sdp_path);
    }
}

#[derive(Clone)]
struct WebRTCMediaSession {
    id: u64,
    peer: Arc<RTCPeerConnection>,
    media_pipeline: Arc<SharedMediaPipeline>,
}

impl WebRTCMediaSession {
    async fn shutdown(&self) {
        self.media_pipeline.unregister_tracks(self.id).await;
        let _ = self.peer.close().await;
    }
}

#[derive(Clone)]
struct AppState {
    config: Config,
    manager: Arc<Mutex<Manager>>,
    webrtc_sessions: Arc<Mutex<Vec<Arc<WebRTCMediaSession>>>>,
    media_pipeline: Arc<Mutex<Option<Arc<SharedMediaPipeline>>>>,
    next_session_id: Arc<AtomicU64>,
    webrtc_connect_lock: Arc<AsyncMutex<()>>,
    face_service: Arc<face::FaceService>,
}

impl Default for Manager {
    fn default() -> Self {
        let config = Config::from_env();
        Self {
            camera: CameraState {
                width: config.camera_width,
                height: config.camera_height,
                fps: config.camera_fps,
                camera_device: config.camera_device.clone(),
            },
            audio: AudioState {
                selected_microphone: config.pulse_capture_source_name.clone(),
                selected_speaker: config.pulse_sink_name.clone(),
                volume: 70,
            },
            face: FaceState {
                enabled: config.face_recognition_enabled,
                available: false,
                initializing: false,
                backend: "opencv-haar-template".to_string(),
                message: if config.face_recognition_enabled {
                    "not initialized".to_string()
                } else {
                    "disabled".to_string()
                },
                known_faces_count: 0,
                detect_every_n_frames: config.face_recognition_detect_every_n_frames,
                match_threshold: config.face_recognition_match_threshold,
                max_faces: config.face_recognition_max_faces,
                result: FaceResult::default(),
            },
        }
    }
}

#[derive(Debug, Deserialize)]
struct CameraSettingsRequest {
    width: i32,
    height: i32,
    fps: i32,
    camera_device: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AudioSelectionRequest {
    microphone: Option<String>,
    speaker: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SpeakerVolumeRequest {
    volume: i32,
}

#[derive(Debug, Deserialize)]
struct FaceSettingsRequest {
    enabled: bool,
}

#[derive(Debug, Deserialize)]
struct WebRTCConnectRequest {
    sdp: Option<String>,
    #[serde(rename = "type")]
    type_: Option<String>,
}

#[tokio::main]
async fn main() -> AnyhowResult<()> {
    tracing_subscriber::fmt::init();
    let _ = rustls::crypto::ring::default_provider().install_default();

    let config = Config::from_env();
    let manager = Arc::new(Mutex::new(Manager::default()));

    let face_service = Arc::new(face::FaceService::new(
        config.face_recognition_enabled,
        config.face_recognition_known_faces_dir.clone(),
        config.face_recognition_cascade_path.clone(),
        config.face_recognition_detect_every_n_frames,
        config.face_recognition_match_threshold,
        config.face_recognition_max_faces,
        config.face_recognition_min_consecutive_frames,
    ));
    face_service.init();

    let state = Arc::new(AppState {
        config: config.clone(),
        manager: manager.clone(),
        webrtc_sessions: Arc::new(Mutex::new(Vec::new())),
        media_pipeline: Arc::new(Mutex::new(None)),
        next_session_id: Arc::new(AtomicU64::new(1)),
        webrtc_connect_lock: Arc::new(AsyncMutex::new(())),
        face_service,
    });

    let app = build_app(state.clone());

    let addr = SocketAddr::from(([0, 0, 0, 0], config.server_port));
    let cert_path = config.ssl_cert_path.clone();
    let key_path = config.ssl_key_path.clone();

    let tls_config = axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert_path, &key_path)
        .await
        .with_context(|| format!("failed to load TLS cert {:?} and key {:?}", cert_path, key_path))?;

    let handle = axum_server::Handle::new();
    let shutdown_handle = handle.clone();
    tokio::spawn(async move {
        shutdown_signal().await;
        shutdown_handle.graceful_shutdown(Some(Duration::from_secs(5)));
    });

    tracing::info!("server listening at https://{}:{}", config.server_host, config.server_port);
    let serve_result = axum_server::bind_rustls(addr, tls_config)
        .handle(handle)
        .serve(app.into_make_service())
        .await;

    shutdown_runtime_state(state).await;

    serve_result.map_err(anyhow::Error::from)
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate = match signal(SignalKind::terminate()) {
            Ok(sig) => sig,
            Err(err) => {
                tracing::warn!("failed to register SIGTERM handler: {err}");
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };

        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = terminate.recv() => {},
        }
    }

    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

async fn shutdown_runtime_state(state: Arc<AppState>) {
    let sessions = {
        let mut guard = state.webrtc_sessions.lock().unwrap();
        std::mem::take(&mut *guard)
    };

    for session in sessions {
        session.shutdown().await;
    }

    let pipeline = {
        let mut guard = state.media_pipeline.lock().unwrap();
        guard.take()
    };

    if let Some(pipeline) = pipeline {
        pipeline.shutdown().await;
    }
}

fn build_app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/status", get(status_handler))
        .route("/debug/media", get(debug_media_handler))
        .route("/camera_settings", get(camera_settings_get).post(camera_settings_post))
        .route("/server_audio_devices", get(server_audio_devices_get))
        .route("/server_audio_devices/select", post(server_audio_devices_select_post))
        .route("/speaker_volume", get(speaker_volume_get).post(speaker_volume_post))
        .route("/face_status", get(face_status_handler))
        .route("/face_settings", post(face_settings_post))
        .route("/webrtc/connect", post(webrtc_connect_post))
        .route("/webrtc/status", get(webrtc_status_handler))
        .route("/static/{*path}", get(static_handler))
        .with_state(state)
}

async fn start_rtp_reader(
    socket: UdpSocket,
    tracks: Arc<AsyncMutex<Vec<(u64, Arc<TrackLocalStaticRTP>)>>>,
    ingress_counter: Arc<AtomicU64>,
    writes_by_session: Arc<AsyncMutex<HashMap<u64, u64>>>,
    metadata_channels: Option<Arc<AsyncMutex<Vec<(u64, String, Arc<RTCDataChannel>)>>>>,
    video_frame_seq: Option<Arc<AtomicU64>>,
) -> AnyhowResult<JoinHandle<()>> {
    let port = socket.local_addr()?.port();

    let task = tokio::spawn(async move {
        let mut buf = [0u8; 8192];
        loop {
            match socket.recv_from(&mut buf).await {
                Ok((len, _)) => {
                    let mut packet_data = &buf[..len];
                    match Packet::unmarshal(&mut packet_data) {
                        Ok(packet) => {
                            ingress_counter.fetch_add(1, Ordering::Relaxed);

                            let targets = {
                                let guard = tracks.lock().await;
                                guard.clone()
                            };

                            let mut successful_session_writes = Vec::new();

                            for (session_id, track) in targets {
                                if let Err(err) = track.write_rtp(&packet).await {
                                    tracing::warn!("rtp reader on port {} write failed: {err}", port);
                                } else {
                                    successful_session_writes.push(session_id);
                                }
                            }

                            if !successful_session_writes.is_empty() {
                                let mut counters = writes_by_session.lock().await;
                                for session_id in successful_session_writes {
                                    *counters.entry(session_id).or_insert(0) += 1;
                                }
                            }

                            if packet.header.marker {
                                if let (Some(channels), Some(frame_seq)) = (&metadata_channels, &video_frame_seq) {
                                    let seq = frame_seq.fetch_add(1, Ordering::Relaxed) + 1;
                                    let payload = format!(
                                        "{{\"type\":\"frame_meta\",\"broadcast_frame_seq\":{seq}}}"
                                    );

                                    let targets = {
                                        let guard = channels.lock().await;
                                        guard.clone()
                                    };

                                    let mut stale = Vec::new();
                                    for (session_id, label, dc) in targets {
                                        if let Err(err) = dc.send_text(payload.clone()).await {
                                            tracing::debug!(
                                                "frame_meta send failed for session {} channel {}: {}",
                                                session_id,
                                                label,
                                                err
                                            );
                                            stale.push((session_id, label));
                                        }
                                    }

                                    if !stale.is_empty() {
                                        let mut guard = channels.lock().await;
                                        guard.retain(|(sid, label, _)| {
                                            !stale.iter().any(|(stale_sid, stale_label)| stale_sid == sid && stale_label == label)
                                        });
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            tracing::warn!("failed to parse RTP on port {}: {err}", port);
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!("UDP reader on port {} stopped: {err}", port);
                    break;
                }
            }
        }
    });

    Ok(task)
}

fn start_video_encoder(
    config: &Config,
    rtp_port: u16,
) -> AnyhowResult<(std::process::Child, std::process::ChildStdin, std::process::ChildStdout)> {
    let fps = config.camera_fps.max(1).to_string();
    let rtp_url = format!("rtp://127.0.0.1:{rtp_port}");

    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-f", "mjpeg",
        "-framerate",
        &fps,
        "-i",
        "pipe:0",
        // output 0: VP8 RTP for WebRTC
        "-an",
        "-c:v",
        "libvpx",
        "-deadline",
        "realtime",
        "-cpu-used",
        "5",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rtp",
        &rtp_url,
        // output 1: low-fps MJPEG pipe for face recognition
        "-an",
        "-vf",
        "fps=2,scale=640:-2",
        "-c:v",
        "mjpeg",
        "-q:v",
        "5",
        "-f",
        "image2pipe",
        "pipe:1",
    ])
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());

    let mut child = cmd.spawn().map_err(anyhow::Error::from)?;
    let stdin = child.stdin.take()
        .ok_or_else(|| anyhow::anyhow!("video encoder: no stdin pipe"))?;
    let stdout = child.stdout.take()
        .ok_or_else(|| anyhow::anyhow!("video encoder: no stdout pipe"))?;
    Ok((child, stdin, stdout))
}

fn start_camera_capture(
    config: &Config,
) -> AnyhowResult<(std::process::Child, std::process::ChildStdout)> {
    let mut cmd = if let Some(camera_index) = rpicam_index(&config.camera_device) {
        let width = config.camera_width.max(2).to_string();
        let height = config.camera_height.max(2).to_string();
        let fps = config.camera_fps.max(1).to_string();
        let mut command = Command::new("rpicam-vid");
        command.args([
            "--camera", &camera_index, "--codec", "mjpeg", "--width", &width,
            "--height", &height, "--framerate", &fps, "--timeout", "0",
            "--nopreview", "--output", "-",
        ]);
        command
    } else {
        let video_size = format!("{}x{}", config.camera_width.max(2), config.camera_height.max(2));
        let fps = config.camera_fps.max(1).to_string();
        let mut command = Command::new("ffmpeg");
        command.args([
            "-hide_banner", "-loglevel", "error", "-f", "v4l2",
            "-input_format", "mjpeg", "-video_size", &video_size,
            "-framerate", &fps, "-i", &config.camera_device,
            "-vcodec", "copy", "-f", "mjpeg", "pipe:1",
        ]);
        command
    };
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = cmd.spawn().map_err(anyhow::Error::from)?;
    let stdout = child.stdout.take()
        .ok_or_else(|| anyhow::anyhow!("camera capture: no stdout pipe"))?;
    Ok((child, stdout))
}

fn spawn_camera_pipe(
    mut camera_stdout: std::process::ChildStdout,
    encoder_stdin: Arc<Mutex<Option<std::process::ChildStdin>>>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = match camera_stdout.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(read) => read,
            };
            let mut stdin = encoder_stdin.lock().unwrap();
            let Some(stdin) = stdin.as_mut() else { break };
            if stdin.write_all(&buffer[..read]).is_err() {
                break;
            }
        }
    })
}

fn start_audio_ffmpeg(source: &str, rtp_port: u16) -> AnyhowResult<std::process::Child> {
    let mut cmd = Command::new("ffmpeg");
    let input_name = if source.is_empty() || source == "@DEFAULT_SOURCE@" { "default" } else { source };
    let rtp_url = format!("rtp://127.0.0.1:{rtp_port}");

    cmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-f",
        "pulse",
        "-i",
        input_name,
        "-ar",
        "48000",
        "-ac",
        "1",
        // match Go's volume+compressor chain so quiet mics are audible
        "-af",
        "volume=18dB,acompressor=threshold=-24dB:ratio=8:attack=5:release=100:makeup=6dB",
        "-c:a",
        "libopus",
        "-application",
        "lowdelay",
        "-frame_duration",
        "20",
        "-f",
        "rtp",
        &rtp_url,
    ])
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::null());

    cmd.spawn().map_err(anyhow::Error::from)
}

fn write_talkback_sdp(port: u16) -> AnyhowResult<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("surveillance-rust-talkback-{stamp}.sdp"));
    let sdp = format!(
        concat!(
            "v=0\n",
            "o=- 0 0 IN IP4 127.0.0.1\n",
            "s=Surveillance Rust Talkback\n",
            "c=IN IP4 127.0.0.1\n",
            "t=0 0\n",
            "m=audio {} RTP/AVP 111\n",
            "a=rtpmap:111 opus/48000/2\n"
        ),
        port
    );
    std::fs::write(&path, sdp)?;
    Ok(path)
}

fn start_talkback_ffmpeg(sdp_path: &std::path::Path, sink_name: &str) -> AnyhowResult<std::process::Child> {
    let mut cmd = Command::new("ffmpeg");
    let pulse_sink = if sink_name.is_empty() || sink_name == "@DEFAULT_SINK@" {
        "default"
    } else {
        sink_name
    };

    cmd.args([
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file,udp,rtp",
        "-f",
        "sdp",
        "-i",
        sdp_path.to_string_lossy().as_ref(),
        "-af",
        "volume=5.0",
        "-f",
        "pulse",
        pulse_sink,
    ])
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::null());

    cmd.spawn().map_err(anyhow::Error::from)
}

fn watchdog_cleanup_stale_ffmpeg() -> WatchdogCleanupStats {
    // Only remove stale temp artifacts. Do not kill ffmpeg here because active
    // concurrent sessions share a single media pipeline process tree.
    let mut stats = WatchdogCleanupStats::default();

    if let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("surveillance-rust-talkback-") && name.ends_with(".sdp") {
                    if std::fs::remove_file(path).is_ok() {
                        stats.files_cleaned += 1;
                    }
                }
            }
        }
    }

    stats
}

fn cleanup_orphan_media_workers() -> usize {
    let patterns = [
        "rtp://127.0.0.1:",
        "surveillance-rust-talkback-",
    ];

    let mut cleaned = 0usize;
    for pattern in patterns {
        let count = Command::new("pgrep")
            .args(["-f", pattern])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .count()
            })
            .unwrap_or(0);

        let _ = Command::new("pkill")
            .args(["-9", "-f", pattern])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();

        cleaned += count;
    }

    cleaned
}

async fn index_handler() -> std::result::Result<Html<String>, StatusCode> {
    let public_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("public");
    let html = fs::read_to_string(public_dir.join("index.html"))
        .await
        .map_err(|_| StatusCode::NOT_FOUND)?;
    Ok(Html(html))
}

async fn static_handler(Path(path): Path<String>) -> std::result::Result<Response, StatusCode> {
    let safe_path = path.trim_start_matches('/');
    let public_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("public");
    let file_path = public_dir.join(safe_path);

    if !file_path.starts_with(&public_dir) {
        return Err(StatusCode::FORBIDDEN);
    }

    let bytes = fs::read(&file_path).await.map_err(|_| StatusCode::NOT_FOUND)?;
    let mime = match file_path.extension().and_then(|s| s.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("svg") => "image/svg+xml",
        Some("json") => "application/json; charset=utf-8",
        _ => "application/octet-stream",
    };

    Ok(([(axum::http::header::CONTENT_TYPE, mime)], bytes).into_response())
}

async fn status_handler(State(state): State<Arc<AppState>>) -> Json<Value> {
    let manager = state.manager.lock().unwrap();
    let camera_source_type = camera_source_type_for_device(&manager.camera.camera_device);
    Json(json!({
        "camera": true,
        "audio": true,
        "queue_size": 0,
        "camera_device": manager.camera.camera_device,
        "camera_source_type": camera_source_type,
        "camera_device_preference": manager.camera.camera_device,
        "camera_width": manager.camera.width,
        "camera_height": manager.camera.height,
        "camera_fps": manager.camera.fps,
    }))
}

async fn camera_settings_get(State(state): State<Arc<AppState>>) -> Json<Value> {
    let manager = state.manager.lock().unwrap();
    let camera_source_type = camera_source_type_for_device(&manager.camera.camera_device);
    let available_devices = available_camera_devices();
    Json(json!({
        "status": "ok",
        "width": manager.camera.width,
        "height": manager.camera.height,
        "fps": manager.camera.fps,
        "camera_device": manager.camera.camera_device,
        "selected_camera_device": manager.camera.camera_device,
        "camera_source_type": camera_source_type,
        "available_camera_devices": available_devices,
        "supported_resolutions": [
            {"width": 640, "height": 480},
            {"width": 1280, "height": 720},
            {"width": 1920, "height": 1080},
            {"width": 2560, "height": 1440}
        ],
        "allowed_resolutions": [
            {"width": 640, "height": 480},
            {"width": 1280, "height": 720},
            {"width": 1920, "height": 1080},
            {"width": 2560, "height": 1440}
        ],
        "fps_range": {"min": 1, "max": 60}
    }))
}

async fn camera_settings_post(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CameraSettingsRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if req.width <= 0 || req.height <= 0 || req.fps <= 0 {
        return Err((StatusCode::BAD_REQUEST, Json(json!({"status":"error","message":"width, height and fps must be integers"}))))
    }

    let is_supported = matches!((req.width, req.height), (640, 480) | (1280, 720) | (1920, 1080) | (2560, 1440));
    if !is_supported {
        return Err((StatusCode::BAD_REQUEST, Json(json!({"status":"error","message": format!("unsupported resolution {}x{}", req.width, req.height)}))));
    }
    if req.fps < 1 || req.fps > 60 {
        return Err((StatusCode::BAD_REQUEST, Json(json!({"status":"error","message":"fps must be between 1 and 60"}))));
    }

    let selected_camera_device = req.camera_device.clone().filter(|v| !v.trim().is_empty()).unwrap_or_else(|| state.config.camera_device.clone());

    {
        let mut manager = state.manager.lock().unwrap();
        manager.camera.width = req.width;
        manager.camera.height = req.height;
        manager.camera.fps = req.fps;
        if let Some(camera_device) = req.camera_device.clone().filter(|v| !v.trim().is_empty()) {
            manager.camera.camera_device = camera_device;
        }
    }

    let active_pipeline = state.media_pipeline.lock().unwrap().clone();
    if let Some(pipeline) = active_pipeline {
        let mut pipeline_config = state.config.clone();
        pipeline_config.camera_device = selected_camera_device.clone();
        pipeline_config.camera_width = req.width;
        pipeline_config.camera_height = req.height;
        pipeline_config.camera_fps = req.fps;
        if let Err(err) = pipeline
            .restart_camera(&pipeline_config, state.face_service.clone())
            .await
        {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({
                    "status": "error",
                    "message": format!("camera restart failed: {err}")
                })),
            ));
        }
    }

    let camera_source_type = camera_source_type_for_device(&selected_camera_device);
    let available_devices = available_camera_devices();

    Ok(Json(json!({
        "status": "ok",
        "width": req.width,
        "height": req.height,
        "fps": req.fps,
        "camera_device": selected_camera_device.clone(),
        "selected_camera_device": selected_camera_device.clone(),
        "camera_source_type": camera_source_type,
        "available_camera_devices": available_devices,
        "supported_resolutions": [
            {"width": 640, "height": 480},
            {"width": 1280, "height": 720},
            {"width": 1920, "height": 1080},
            {"width": 2560, "height": 1440}
        ],
        "allowed_resolutions": [
            {"width": 640, "height": 480},
            {"width": 1280, "height": 720},
            {"width": 1920, "height": 1080},
            {"width": 2560, "height": 1440}
        ],
        "fps_range": {"min": 1, "max": 60}
    })))
}

async fn server_audio_devices_get(State(state): State<Arc<AppState>>) -> Json<Value> {
    let manager = state.manager.lock().unwrap();
    Json(json!({
        "status": "ok",
        "microphones": [
            {"id": "@DEFAULT_SOURCE@", "name": "Default microphone", "kind": "default"},
            {"id": "alsa_input.pci-0000_00_1f.3.analog-stereo", "name": "alsa_input.pci-0000_00_1f.3.analog-stereo", "kind": "pulseaudio-source"}
        ],
        "speakers": [
            {"id": "@DEFAULT_SINK@", "name": "Default speaker", "kind": "default"},
            {"id": "alsa_output.pci-0000_00_1f.3.analog-stereo", "name": "alsa_output.pci-0000_00_1f.3.analog-stereo", "kind": "pulseaudio-sink"}
        ],
        "selected_microphone": manager.audio.selected_microphone,
        "selected_speaker": manager.audio.selected_speaker,
    }))
}

async fn server_audio_devices_select_post(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AudioSelectionRequest>,
) -> Json<Value> {
    let mut manager = state.manager.lock().unwrap();
    if let Some(microphone) = req.microphone.filter(|v| !v.trim().is_empty()) {
        manager.audio.selected_microphone = microphone;
    }
    if let Some(speaker) = req.speaker.filter(|v| !v.trim().is_empty()) {
        manager.audio.selected_speaker = speaker;
    }
    Json(json!({
        "status": "ok",
        "selected_microphone": manager.audio.selected_microphone,
        "selected_speaker": manager.audio.selected_speaker,
    }))
}

async fn speaker_volume_get(State(state): State<Arc<AppState>>) -> Json<Value> {
    let manager = state.manager.lock().unwrap();
    Json(json!({
        "status": "ok",
        "available": true,
        "volume": manager.audio.volume,
        "control": "pactl"
    }))
}

async fn speaker_volume_post(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SpeakerVolumeRequest>,
) -> Json<Value> {
    let mut manager = state.manager.lock().unwrap();
    manager.audio.volume = req.volume.clamp(0, 100);
    Json(json!({
        "status": "ok",
        "volume": manager.audio.volume,
        "control": "pactl"
    }))
}

async fn face_status_handler(State(state): State<Arc<AppState>>) -> Json<Value> {
    let status = state.face_service.get_status();
    Json(json!({
        "enabled": status.enabled,
        "available": status.available,
        "initializing": status.initializing,
        "backend": status.backend,
        "message": status.message,
        "known_faces_count": status.known_faces_count,
        "detect_every_n_frames": status.detect_every_n_frames,
        "match_threshold": status.match_threshold,
        "max_faces": status.max_faces,
        "result": status.result,
    }))
}

async fn face_settings_post(
    State(state): State<Arc<AppState>>,
    Json(req): Json<FaceSettingsRequest>,
) -> Json<Value> {
    state.face_service.set_enabled(req.enabled);
    let status = state.face_service.get_status();
    Json(json!({
        "status": "ok",
        "enabled": status.enabled,
        "available": status.available,
        "initializing": status.initializing,
        "backend": status.backend,
        "message": status.message,
        "known_faces_count": status.known_faces_count,
        "detect_every_n_frames": status.detect_every_n_frames,
        "match_threshold": status.match_threshold,
        "max_faces": status.max_faces,
        "result": status.result,
    }))
}

async fn webrtc_status_handler(State(state): State<Arc<AppState>>) -> Json<Value> {
    let sessions = {
        let active = state.webrtc_sessions.lock().unwrap();
        active.len()
    };

    Json(json!({
        "status": "ok",
        "sessions": sessions,
        "media_port_range": {
            "min": 0,
            "max": TALKBACK_RTP_PORT
        }
    }))
}

fn child_debug_info(child: &Arc<Mutex<Option<std::process::Child>>>) -> Value {
    let mut guard = child.lock().unwrap();
    if let Some(process) = guard.as_mut() {
        let running = process.try_wait().ok().flatten().is_none();
        json!({
            "pid": process.id(),
            "running": running
        })
    } else {
        json!({
            "pid": Value::Null,
            "running": false
        })
    }
}

async fn debug_media_handler(State(state): State<Arc<AppState>>) -> Json<Value> {
    let active_session_ids = {
        let sessions = state.webrtc_sessions.lock().unwrap();
        sessions.iter().map(|session| session.id).collect::<Vec<_>>()
    };

    let maybe_pipeline = {
        let pipeline = state.media_pipeline.lock().unwrap();
        pipeline.clone()
    };

    if let Some(pipeline) = maybe_pipeline {
        let video_counts = {
            let mut counts: HashMap<u64, usize> = HashMap::new();
            let guard = pipeline.video_tracks.lock().await;
            for (session_id, _) in guard.iter() {
                *counts.entry(*session_id).or_insert(0) += 1;
            }
            counts
        };

        let audio_counts = {
            let mut counts: HashMap<u64, usize> = HashMap::new();
            let guard = pipeline.audio_tracks.lock().await;
            for (session_id, _) in guard.iter() {
                *counts.entry(*session_id).or_insert(0) += 1;
            }
            counts
        };

        let video_writes_by_session = {
            let guard = pipeline.video_packets_written_by_session.lock().await;
            guard.clone()
        };

        let audio_writes_by_session = {
            let guard = pipeline.audio_packets_written_by_session.lock().await;
            guard.clone()
        };

        let metadata_counts = {
            let mut counts: HashMap<u64, usize> = HashMap::new();
            let guard = pipeline.metadata_channels.lock().await;
            for (session_id, _, _) in guard.iter() {
                *counts.entry(*session_id).or_insert(0) += 1;
            }
            counts
        };

        let mut all_session_ids: BTreeSet<u64> = active_session_ids.iter().copied().collect();
        all_session_ids.extend(video_counts.keys().copied());
        all_session_ids.extend(audio_counts.keys().copied());
        all_session_ids.extend(video_writes_by_session.keys().copied());
        all_session_ids.extend(audio_writes_by_session.keys().copied());
        all_session_ids.extend(metadata_counts.keys().copied());

        let per_session = all_session_ids
            .into_iter()
            .map(|session_id| {
                json!({
                    "session_id": session_id,
                    "active_webrtc_session": active_session_ids.contains(&session_id),
                    "video_tracks": video_counts.get(&session_id).copied().unwrap_or(0),
                    "audio_tracks": audio_counts.get(&session_id).copied().unwrap_or(0),
                    "metadata_channels": metadata_counts.get(&session_id).copied().unwrap_or(0),
                    "video_packets_written": video_writes_by_session.get(&session_id).copied().unwrap_or(0),
                    "audio_packets_written": audio_writes_by_session.get(&session_id).copied().unwrap_or(0)
                })
            })
            .collect::<Vec<_>>();

        return Json(json!({
            "status": "ok",
            "pipeline": {
                "active": true,
                "healthy": pipeline.has_live_media_workers(),
                "video_rtp_port": pipeline.video_rtp_port,
                "audio_rtp_port": pipeline.audio_rtp_port,
                "created_at_unix_ms": pipeline.created_at_unix_ms,
                "talkback_rtp_port": TALKBACK_RTP_PORT,
                "talkback_sdp_path": pipeline.talkback_sdp_path,
                "ffmpeg": {
                    "video": child_debug_info(&pipeline.video_ffmpeg),
                    "audio": child_debug_info(&pipeline.audio_ffmpeg),
                    "talkback": child_debug_info(&pipeline.talkback_ffmpeg)
                },
                "telemetry": {
                    "video_packets_in": pipeline.video_packets_in.load(Ordering::Relaxed),
                    "audio_packets_in": pipeline.audio_packets_in.load(Ordering::Relaxed),
                    "video_frame_seq": pipeline.video_frame_seq.load(Ordering::Relaxed),
                    "metadata_channel_total": metadata_counts.values().sum::<usize>()
                },
                "track_registry": {
                    "video_total": video_counts.values().sum::<usize>(),
                    "audio_total": audio_counts.values().sum::<usize>()
                }
            },
            "sessions": {
                "active_webrtc_total": active_session_ids.len(),
                "per_session_track_counts": per_session
            }
        }));
    }

    Json(json!({
        "status": "ok",
        "pipeline": {
            "active": false,
            "healthy": false,
            "video_rtp_port": Value::Null,
            "audio_rtp_port": Value::Null,
            "created_at_unix_ms": Value::Null,
            "talkback_rtp_port": TALKBACK_RTP_PORT,
            "talkback_sdp_path": Value::Null,
            "ffmpeg": {
                "video": {"pid": Value::Null, "running": false},
                "audio": {"pid": Value::Null, "running": false},
                "talkback": {"pid": Value::Null, "running": false}
            },
            "telemetry": {
                "video_packets_in": 0,
                "audio_packets_in": 0,
                "video_frame_seq": 0,
                "metadata_channel_total": 0
            },
            "track_registry": {
                "video_total": 0,
                "audio_total": 0
            }
        },
        "sessions": {
            "active_webrtc_total": active_session_ids.len(),
            "per_session_track_counts": []
        }
    }))
}

#[axum::debug_handler]
async fn webrtc_connect_post(
    State(state): State<Arc<AppState>>,
    Json(req): Json<WebRTCConnectRequest>,
) -> Response {
    let _connect_guard = state.webrtc_connect_lock.lock().await;

    let result = async {
        let cleanup_stats = watchdog_cleanup_stale_ffmpeg();
        tracing::info!(
            "watchdog cleanup: killed {} stale ffmpeg process(es), removed {} stale talkback sdp file(s)",
            cleanup_stats.processes_cleaned,
            cleanup_stats.files_cleaned
        );

        let session_id = state.next_session_id.fetch_add(1, Ordering::SeqCst);

        let sdp = req.sdp.ok_or_else(|| (StatusCode::BAD_REQUEST, Json(json!({ "status": "error", "message": "missing sdp" }))))?;
        let offer_type = req.type_.ok_or_else(|| (StatusCode::BAD_REQUEST, Json(json!({ "status": "error", "message": "missing type" }))))?;

        if offer_type != "offer" {
            return Err((StatusCode::BAD_REQUEST, Json(json!({ "status": "error", "message": "expected offer type" }))));
        }

        let mut media_engine = MediaEngine::default();
        media_engine.register_default_codecs()
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "status": "error", "message": format!("media engine error: {e}") }))))?;

        let api = APIBuilder::new()
            .with_media_engine(media_engine)
            .build();

        let config = RTCConfiguration {
            ice_servers: vec![RTCIceServer {
                urls: vec!["stun:stun.l.google.com:19302".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        };

        let pc = Arc::new(
            api.new_peer_connection(config)
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "status": "error", "message": format!("peer init error: {e}") }))))?,
        );

        let state_for_close = state.clone();
        pc.on_peer_connection_state_change(Box::new(move |connection_state| {
            let state_for_close = state_for_close.clone();
            Box::pin(async move {
                if connection_state == RTCPeerConnectionState::Closed || connection_state == RTCPeerConnectionState::Failed {
                    let removed = {
                        let mut sessions = state_for_close.webrtc_sessions.lock().unwrap();
                        sessions
                            .iter()
                            .position(|session| session.id == session_id)
                            .map(|index| sessions.remove(index))
                    };

                    if let Some(session) = removed {
                        session.shutdown().await;
                    }
                }
            })
        }));

        let talkback_target = format!("127.0.0.1:{TALKBACK_RTP_PORT}");
        pc.on_track(Box::new(move |track, _, _| {
            let talkback_target = talkback_target.clone();
            tokio::spawn(async move {
                if track.kind() != RTPCodecType::Audio {
                    return;
                }

                let socket = match UdpSocket::bind("127.0.0.1:0").await {
                    Ok(sock) => sock,
                    Err(err) => {
                        tracing::warn!("talkback bridge UDP bind failed: {err}");
                        return;
                    }
                };

                let mut packet_buf = vec![0_u8; 2000];
                while let Ok((packet, _)) = track.read_rtp().await {
                    let size = packet.marshal_size();
                    if packet_buf.len() < size {
                        packet_buf.resize(size, 0);
                    }

                    match packet.marshal_to(&mut packet_buf[..size]) {
                        Ok(written) => {
                            if let Err(err) = socket.send_to(&packet_buf[..written], &talkback_target).await {
                                tracing::warn!("talkback bridge send failed: {err}");
                                break;
                            }
                        }
                        Err(err) => {
                            tracing::warn!("talkback bridge packet marshal failed: {err}");
                        }
                    }
                }
            });

            Box::pin(async {})
        }));

        let state_for_data_channel = state.clone();
        pc.on_data_channel(Box::new(move |dc| {
            let state_for_data_channel = state_for_data_channel.clone();
            Box::pin(async move {
                let label = dc.label().to_string();
                if label != "video-text" && label != "face-data" {
                    return;
                }

                let state_for_open = state_for_data_channel.clone();
                let dc_for_open = dc.clone();
                let label_for_open = label.clone();
                dc.on_open(Box::new(move || {
                    let state_for_open = state_for_open.clone();
                    let dc_for_open = dc_for_open.clone();
                    let label_for_open = label_for_open.clone();
                    Box::pin(async move {
                        let maybe_pipeline = {
                            let active = state_for_open.media_pipeline.lock().unwrap();
                            active.clone()
                        };

                        if let Some(pipeline) = maybe_pipeline {
                            pipeline
                                .register_metadata_channel(session_id, label_for_open.clone(), dc_for_open.clone())
                                .await;
                        }

                        // Subscribe this datachannel to live face recognition results.
                        let mut face_rx = state_for_open.face_service.broadcast.subscribe();
                        let dc_face = dc_for_open.clone();
                        tokio::spawn(async move {
                            while let Ok(json_str) = face_rx.recv().await {
                                if dc_face.send_text(json_str).await.is_err() {
                                    break;
                                }
                            }
                        });
                    })
                }));

                let state_for_close = state_for_data_channel.clone();
                let label_for_close = label.clone();
                dc.on_close(Box::new(move || {
                    let state_for_close = state_for_close.clone();
                    let label_for_close = label_for_close.clone();
                    Box::pin(async move {
                        let maybe_pipeline = {
                            let active = state_for_close.media_pipeline.lock().unwrap();
                            active.clone()
                        };

                        if let Some(pipeline) = maybe_pipeline {
                            pipeline
                                .unregister_metadata_channel(session_id, &label_for_close)
                                .await;
                        }
                    })
                }));
            })
        }));

        let video_track = Arc::new(TrackLocalStaticRTP::new(
            RTCRtpCodecCapability {
                mime_type: "video/vp8".to_string(),
                clock_rate: 90000,
                channels: 0,
                sdp_fmtp_line: String::new(),
                rtcp_feedback: vec![],
            },
            "camera-video".to_string(),
            "surveillance-rust".to_string(),
        ));
        let audio_track = Arc::new(TrackLocalStaticRTP::new(
            RTCRtpCodecCapability {
                mime_type: "audio/opus".to_string(),
                clock_rate: 48000,
                channels: 1,
                sdp_fmtp_line: "minptime=10;useinbandfec=1".to_string(),
                rtcp_feedback: vec![],
            },
            "camera-audio".to_string(),
            "surveillance-rust".to_string(),
        ));

        pc.add_track(video_track.clone() as Arc<dyn TrackLocal + Send + Sync>)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "status": "error", "message": format!("video track init error: {e}") }))))?;
        pc.add_track(audio_track.clone() as Arc<dyn TrackLocal + Send + Sync>)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "status": "error", "message": format!("audio track init error: {e}") }))))?;

        let pipeline_config = {
            let manager = state.manager.lock().unwrap();
            let mut config = state.config.clone();
            config.camera_device = manager.camera.camera_device.clone();
            config.camera_width = manager.camera.width;
            config.camera_height = manager.camera.height;
            config.camera_fps = manager.camera.fps;
            config
        };

        let media_pipeline = {
            let existing = {
                let active = state.media_pipeline.lock().unwrap();
                active.clone()
            };

            if let Some(pipeline) = existing {
                if pipeline.has_live_media_workers() {
                    pipeline
                } else {
                    tracing::warn!("existing media pipeline is unhealthy; recreating before attaching new session");
                    pipeline.shutdown().await;
                    {
                        let mut active = state.media_pipeline.lock().unwrap();
                        *active = None;
                    }

                    let replacement = Arc::new(
                        SharedMediaPipeline::start(&pipeline_config, state.face_service.clone())
                            .await
                            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "status": "error", "message": format!("media pipeline restart error: {e}") }))))?,
                    );

                    let mut active = state.media_pipeline.lock().unwrap();
                    if active.is_none() {
                        *active = Some(replacement.clone());
                        replacement
                    } else {
                        active.as_ref().unwrap().clone()
                    }
                }
            } else {
                let pipeline = Arc::new(
                    SharedMediaPipeline::start(&pipeline_config, state.face_service.clone())
                        .await
                        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "status": "error", "message": format!("media pipeline init error: {e}") }))))?,
                );
                let mut active = state.media_pipeline.lock().unwrap();
                if active.is_none() {
                    *active = Some(pipeline.clone());
                    pipeline
                } else {
                    active.as_ref().unwrap().clone()
                }
            }
        };

        media_pipeline.register_tracks(session_id, video_track.clone(), audio_track.clone()).await;

        let offer = RTCSessionDescription::offer(sdp.clone())
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "status": "error", "message": format!("offer parse error: {e}") }))))?;

        pc.set_remote_description(offer)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "status": "error", "message": format!("remote description error: {e}") }))))?;

        let answer = pc
            .create_answer(None)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "status": "error", "message": format!("answer creation error: {e}") }))))?;

        // Must register the promise before set_local_description triggers gathering.
        let mut gather_complete = pc.gathering_complete_promise().await;

        pc.set_local_description(answer.clone())
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "status": "error", "message": format!("local description error: {e}") }))))?;

        // Wait for all ICE candidates to be included in the SDP before returning it.
        let _ = gather_complete.recv().await;

        let local = pc
            .local_description()
            .await
            .ok_or_else(|| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "status": "error", "message": "missing local description" }))))?;

        let session = Arc::new(WebRTCMediaSession {
            id: session_id,
            peer: pc.clone(),
            media_pipeline: media_pipeline.clone(),
        });

        {
            let mut sessions = state.webrtc_sessions.lock().unwrap();
            sessions.push(session);
        }

        Ok(Json(json!({
            "sdp": local.sdp,
            "type": "answer"
        })) as Json<Value>)
    };

    match result.await {
        Ok(response) => response.into_response(),
        Err(err) => err.into_response(),
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.server_host, self.server_port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn status_route_responds() {
        let app = build_app(Arc::new(AppState {
            config: Config::from_env(),
            manager: Arc::new(Mutex::new(Manager::default())),
            webrtc_sessions: Arc::new(Mutex::new(Vec::new())),
            media_pipeline: Arc::new(Mutex::new(None)),
            next_session_id: Arc::new(AtomicU64::new(1)),
            webrtc_connect_lock: Arc::new(AsyncMutex::new(())),
            face_service: Arc::new(face::FaceService::new(false, String::new(), String::new(), 1, 0.6, 8, 1)),
        }));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn webrtc_connect_returns_valid_answer_sdp() {
        let app = build_app(Arc::new(AppState {
            config: Config::from_env(),
            manager: Arc::new(Mutex::new(Manager::default())),
            webrtc_sessions: Arc::new(Mutex::new(Vec::new())),
            media_pipeline: Arc::new(Mutex::new(None)),
            next_session_id: Arc::new(AtomicU64::new(1)),
            webrtc_connect_lock: Arc::new(AsyncMutex::new(())),
            face_service: Arc::new(face::FaceService::new(false, String::new(), String::new(), 1, 0.6, 8, 1)),
        }));

        let mut media_engine = MediaEngine::default();
        media_engine.register_default_codecs().unwrap();
        let api = APIBuilder::new()
            .with_media_engine(media_engine)
            .build();
        let pc = api
            .new_peer_connection(RTCConfiguration::default())
            .await
            .unwrap();

        pc.create_data_channel("face-data", None).await.unwrap();
        pc.add_transceiver_from_kind(webrtc::rtp_transceiver::rtp_codec::RTPCodecType::Video, None)
            .await
            .unwrap();
        pc.add_transceiver_from_kind(webrtc::rtp_transceiver::rtp_codec::RTPCodecType::Audio, None)
            .await
            .unwrap();

        let offer = pc.create_offer(None).await.unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/webrtc/connect")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&json!({
                            "sdp": offer.sdp,
                            "type": "offer"
                        }))
                        .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let payload: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();

        assert_eq!(payload["type"], "answer");
        assert!(payload["sdp"].as_str().unwrap().starts_with("v=0"));
    }
}
