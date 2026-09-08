//! Face recognition: OpenCV Haar cascade detection + L2 template matching.
//! Mirrors the Go implementation in surveillance-go/internal/face/face.go.

use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use opencv::{
    core::{self, Mat, Rect, Size, Vector},
    imgcodecs, imgproc,
    objdetect::CascadeClassifier,
    prelude::*,
};
use serde::{Deserialize, Serialize};

// ─── Public result types ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DetectedFace {
    pub name: String,
    pub confidence: f64,
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FaceDetectionResult {
    pub updated_at: i64,
    pub frame_index: i64,
    pub broadcast_frame_seq: u64,
    pub image_width: i32,
    pub image_height: i32,
    pub faces: Vec<DetectedFace>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FaceStatus {
    pub enabled: bool,
    pub available: bool,
    pub initializing: bool,
    pub backend: String,
    pub message: String,
    pub known_faces_count: i32,
    pub detect_every_n_frames: i32,
    pub match_threshold: f64,
    pub max_faces: i32,
    pub result: FaceDetectionResult,
}

// ─── Internal types ───────────────────────────────────────────────────────────

struct KnownFaceSample {
    name: String,
    path: String,     // relative path (cache key)
    signature: String, // "size:mtime_ns"
    mat: Mat,
}

// Mat is explicitly Send in the opencv crate (unsafe impl Send for Mat).
unsafe impl Send for KnownFaceSample {}

struct FaceTrack {
    face: DetectedFace,
    rect: Rect,
    streak: i32,
    missed_count: i32,
}

struct CandidateFace {
    face: DetectedFace,
    rect: Rect,
}

// ─── Service state (non-OpenCV fields, fast mutex) ────────────────────────────

struct ServiceState {
    enabled: bool,
    available: bool,
    initializing: bool,
    message: String,
    known_faces_count: i32,
    detect_every_n_frames: i32,
    match_threshold: f64,
    max_faces: i32,
    min_consecutive_frames: i32,
    last_result: FaceDetectionResult,
    tracks: Vec<FaceTrack>,
    frame_counter: i64,
    process_in_flight: bool,
    known_faces_dir: String,
    cascade_path_override: String,
}

// ─── OpenCV models (separate mutex so status queries never block detection) ───

struct Models {
    classifier: CascadeClassifier,
    samples: Vec<KnownFaceSample>,
}

unsafe impl Send for Models {}

// ─── Frame job ────────────────────────────────────────────────────────────────

struct FrameJob {
    jpeg: Vec<u8>,
    frame_seq: u64,
}

// ─── Public service ───────────────────────────────────────────────────────────

pub struct FaceService {
    state: Arc<Mutex<ServiceState>>,
    models: Arc<Mutex<Option<Models>>>,
    frame_tx: std::sync::mpsc::SyncSender<FrameJob>,
    /// Broadcast channel – sends the JSON face_data payload for every detection.
    pub broadcast: tokio::sync::broadcast::Sender<String>,
}

impl FaceService {
    pub fn new(
        enabled: bool,
        known_faces_dir: String,
        cascade_path_override: String,
        detect_every_n_frames: i32,
        match_threshold: f64,
        max_faces: i32,
        min_consecutive_frames: i32,
    ) -> Self {
        let message = if enabled {
            "not initialized".to_string()
        } else {
            "disabled".to_string()
        };

        let state = Arc::new(Mutex::new(ServiceState {
            enabled,
            available: false,
            initializing: false,
            message,
            known_faces_count: 0,
            detect_every_n_frames: detect_every_n_frames.max(1),
            match_threshold,
            max_faces,
            min_consecutive_frames: min_consecutive_frames.max(1),
            last_result: FaceDetectionResult { faces: vec![], ..Default::default() },
            tracks: vec![],
            frame_counter: 0,
            process_in_flight: false,
            known_faces_dir,
            cascade_path_override,
        }));

        let models: Arc<Mutex<Option<Models>>> = Arc::new(Mutex::new(None));
        let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel::<FrameJob>(4);
        let (broadcast_tx, _) = tokio::sync::broadcast::channel::<String>(16);

        // Two concurrent detection workers (mirrors Go)
        let frame_rx = Arc::new(Mutex::new(frame_rx));
        for _ in 0..2 {
            let state_w = state.clone();
            let models_w = models.clone();
            let rx_w = frame_rx.clone();
            let tx_w = broadcast_tx.clone();
            std::thread::spawn(move || {
                detection_worker(state_w, models_w, rx_w, tx_w);
            });
        }

        FaceService { state, models, frame_tx, broadcast: broadcast_tx }
    }

    /// Start model loading in a background thread if enabled.
    pub fn init(&self) {
        {
            let s = self.state.lock().unwrap();
            if !s.enabled { return; }
        }
        self.trigger_load_models();
    }

    /// Queue a JPEG frame for detection.  Returns false if the queue is full.
    pub fn process_frame(&self, jpeg: Vec<u8>, frame_seq: u64) -> bool {
        self.frame_tx.try_send(FrameJob { jpeg, frame_seq }).is_ok()
    }

    pub fn get_status(&self) -> FaceStatus {
        let s = self.state.lock().unwrap();
        FaceStatus {
            enabled: s.enabled,
            available: s.available,
            initializing: s.initializing,
            backend: "opencv-haar-template".to_string(),
            message: s.message.clone(),
            known_faces_count: s.known_faces_count,
            detect_every_n_frames: s.detect_every_n_frames,
            match_threshold: s.match_threshold,
            max_faces: s.max_faces,
            result: s.last_result.clone(),
        }
    }

    /// Enable or disable at runtime; triggers lazy model loading on first enable.
    pub fn set_enabled(&self, enabled: bool) {
        let needs_init = {
            let mut s = self.state.lock().unwrap();
            let was = s.enabled;
            s.enabled = enabled;
            if !enabled {
                s.message = "disabled".to_string();
                s.last_result = FaceDetectionResult { faces: vec![], ..Default::default() };
            } else if s.available {
                s.message = format!("ready ({} known face sample(s))", s.known_faces_count);
            } else if !s.initializing {
                s.message = "initializing".to_string();
            }
            !was && enabled && !s.available && !s.initializing
        };
        if needs_init {
            self.trigger_load_models();
        }
    }

    /// Reload known faces from disk (e.g. after new images are added).
    pub fn reload_known_faces(&self) {
        self.trigger_load_models();
    }

    pub fn shutdown(&self) {
        // Frame workers stop when the sender is dropped.
        // The broadcast channel stays open until all subscribers drain.
    }

    fn trigger_load_models(&self) {
        let state_c = self.state.clone();
        let models_c = self.models.clone();
        std::thread::spawn(move || {
            load_models(state_c, models_c);
        });
    }
}

// ─── Detection worker ─────────────────────────────────────────────────────────

fn detection_worker(
    state: Arc<Mutex<ServiceState>>,
    models: Arc<Mutex<Option<Models>>>,
    frame_rx: Arc<Mutex<std::sync::mpsc::Receiver<FrameJob>>>,
    result_tx: tokio::sync::broadcast::Sender<String>,
) {
    loop {
        let job = {
            let rx = frame_rx.lock().unwrap();
            match rx.recv() {
                Ok(job) => job,
                Err(_) => return, // channel closed → shut down
            }
        };

        // Quick checks before doing expensive OpenCV work.
        {
            let mut s = state.lock().unwrap();
            if !s.enabled || !s.available || s.initializing || s.process_in_flight {
                continue;
            }
            s.frame_counter += 1;
            if s.detect_every_n_frames > 1 && s.frame_counter % s.detect_every_n_frames as i64 != 0 {
                continue;
            }
            s.process_in_flight = true;
        }

        if let Some(status) = run_detection(&state, &models, &job.jpeg, job.frame_seq) {
            if let Ok(json) = serde_json::to_string(&serde_json::json!({
                "type": "face_data",
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
            })) {
                result_tx.send(json).ok();
            }
        }

        state.lock().unwrap().process_in_flight = false;
    }
}

// ─── Single-frame detection ───────────────────────────────────────────────────

fn run_detection(
    state: &Mutex<ServiceState>,
    models: &Mutex<Option<Models>>,
    jpeg: &[u8],
    frame_seq: u64,
) -> Option<FaceStatus> {
    let (match_threshold, max_faces, min_consecutive) = {
        let s = state.lock().unwrap();
        (s.match_threshold, s.max_faces, s.min_consecutive_frames)
    };

    let jpeg_vec = jpeg.to_vec();
    let buf = core::Vector::<u8>::from(jpeg_vec);

    let img = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR).ok()?;
    if img.empty() { return None; }

    let img_w = img.cols();
    let img_h = img.rows();

    let mut gray = Mat::default();
    imgproc::cvt_color_def(&img, &mut gray, imgproc::COLOR_BGR2GRAY).ok()?;

    let min_dim = img_w.min(img_h);
    let min_face = (min_dim as f64 * 0.10).max(36.0) as i32;
    let max_face = (min_dim as f64 * 0.80) as i32;

    let mut detected_rects: Vector<Rect> = Vector::new();
    {
        let mut mods = models.lock().unwrap();
        let mods = mods.as_mut()?;
        mods.classifier.detect_multi_scale(
            &gray,
            &mut detected_rects,
            1.10,
            7,
            0,
            Size::new(min_face, min_face),
            Size::new(max_face, max_face),
        ).ok()?;
    }

    let rects: Vec<Rect> = if max_faces > 0 {
        detected_rects.iter().take(max_faces as usize).collect()
    } else {
        detected_rects.iter().collect()
    };

    let mut candidates = Vec::new();
    for rect in rects {
        if !is_plausible_face_rect(&rect, img_w, img_h) { continue; }
        let normalized = match normalized_face_from_gray(&gray, &rect) {
            Some(m) => m,
            None => continue,
        };

        let (name, confidence) = {
            let mods = models.lock().unwrap();
            match mods.as_ref() {
                Some(m) if !m.samples.is_empty() => {
                    let (n, score) = best_sample_match(&normalized, &m.samples);
                    let conf = round3(score);
                    let matched = if score >= match_threshold { n } else { "Unknown".to_string() };
                    (matched, conf)
                }
                _ => ("Unknown".to_string(), 0.0),
            }
        };

        candidates.push(CandidateFace {
            face: DetectedFace {
                name,
                confidence,
                left: rect.x.max(0),
                top: rect.y.max(0),
                right: (rect.x + rect.width).min(img_w - 1),
                bottom: (rect.y + rect.height).min(img_h - 1),
            },
            rect,
        });
    }

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    let mut s = state.lock().unwrap();
    let faces = update_tracks(&mut s.tracks, candidates, min_consecutive);
    s.last_result = FaceDetectionResult {
        updated_at: now_ms,
        frame_index: s.frame_counter,
        broadcast_frame_seq: frame_seq,
        image_width: img_w,
        image_height: img_h,
        faces,
    };

    Some(FaceStatus {
        enabled: s.enabled,
        available: s.available,
        initializing: s.initializing,
        backend: "opencv-haar-template".to_string(),
        message: s.message.clone(),
        known_faces_count: s.known_faces_count,
        detect_every_n_frames: s.detect_every_n_frames,
        match_threshold: s.match_threshold,
        max_faces: s.max_faces,
        result: s.last_result.clone(),
    })
}

// ─── Track management ─────────────────────────────────────────────────────────

fn update_tracks(
    tracks: &mut Vec<FaceTrack>,
    candidates: Vec<CandidateFace>,
    min_consecutive: i32,
) -> Vec<DetectedFace> {
    for t in tracks.iter_mut() { t.missed_count += 1; }

    let mut used = vec![false; tracks.len()];
    for cand in candidates {
        let mut best_idx = None;
        let mut best_iou = 0.0f64;
        for (i, t) in tracks.iter().enumerate() {
            if used[i] { continue; }
            let iou = rect_iou(&cand.rect, &t.rect);
            if iou > 0.10 && iou > best_iou {
                best_iou = iou;
                best_idx = Some(i);
            }
        }
        if let Some(i) = best_idx {
            used[i] = true;
            tracks[i].rect = cand.rect;
            tracks[i].face = cand.face;
            tracks[i].streak += 1;
            tracks[i].missed_count = 0;
        } else {
            tracks.push(FaceTrack { face: cand.face, rect: cand.rect, streak: 1, missed_count: 0 });
            used.push(true);
        }
    }

    tracks.retain(|t| t.missed_count <= 2);

    tracks.iter()
        .filter(|t| t.missed_count == 0 && t.streak >= min_consecutive)
        .map(|t| t.face.clone())
        .collect()
}

// ─── OpenCV helpers ───────────────────────────────────────────────────────────

fn normalized_face_from_gray(gray: &Mat, rect: &Rect) -> Option<Mat> {
    if gray.empty() { return None; }
    let bounds = Rect::new(0, 0, gray.cols(), gray.rows());
    let intersection = bounds & *rect;
    if intersection.width <= 0 || intersection.height <= 0 { return None; }

    let region = gray.roi(intersection).ok()?;
    let mut resized = Mat::default();
    imgproc::resize(&region, &mut resized, Size::new(128, 128), 0.0, 0.0, imgproc::INTER_LINEAR).ok()?;
    if resized.empty() { return None; }

    let mut equalized = Mat::default();
    imgproc::equalize_hist(&resized, &mut equalized).ok()?;
    if equalized.empty() { return None; }
    Some(equalized)
}fn best_sample_match(face: &Mat, samples: &[KnownFaceSample]) -> (String, f64) {
    let mut best_name = "Unknown".to_string();
    let mut best_score = 0.0f64;
    for sample in samples {
        if let Ok(distance) = core::norm2(face, &sample.mat, core::NORM_L2, &core::no_array()) {
            let score = l2_distance_to_confidence(distance, face.rows(), face.cols());
            if score > best_score {
                best_score = score;
                best_name = sample.name.clone();
            }
        }
    }
    (best_name, best_score)
}

fn is_plausible_face_rect(rect: &Rect, frame_w: i32, frame_h: i32) -> bool {
    let w = rect.width;
    let h = rect.height;
    if w <= 0 || h <= 0 { return false; }
    let area = w * h;
    if area < 1400 { return false; }
    let aspect = w as f64 / h as f64;
    if aspect < 0.70 || aspect > 1.45 { return false; }
    let mx = (frame_w as f64 * 0.01) as i32;
    let my = (frame_h as f64 * 0.01) as i32;
    if rect.x <= mx || rect.y <= my
        || rect.x + rect.width >= frame_w - mx
        || rect.y + rect.height >= frame_h - my
    {
        return false;
    }
    true
}

fn rect_iou(a: &Rect, b: &Rect) -> f64 {
    let inter = *a & *b;
    if inter.width <= 0 || inter.height <= 0 { return 0.0; }
    let inter_area = (inter.width * inter.height) as f64;
    let union_area = (a.width * a.height + b.width * b.height) as f64 - inter_area;
    if union_area <= 0.0 { return 0.0; }
    inter_area / union_area
}

fn l2_distance_to_confidence(distance: f64, rows: i32, cols: i32) -> f64 {
    if rows <= 0 || cols <= 0 { return 0.0; }
    let max_dist = rows.max(cols) as f64 * 255.0;
    if max_dist <= 0.0 { return 0.0; }
    ((1.0 - distance / max_dist) as f64).clamp(0.0, 1.0)
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

// ─── Model loading ────────────────────────────────────────────────────────────

fn load_models(state: Arc<Mutex<ServiceState>>, models: Arc<Mutex<Option<Models>>>) {
    {
        let mut s = state.lock().unwrap();
        if s.initializing { return; }
        s.initializing = true;
        s.available = false;
        s.message = "loading OpenCV classifier".to_string();
    }

    let (known_faces_dir, cascade_override) = {
        let s = state.lock().unwrap();
        (s.known_faces_dir.clone(), s.cascade_path_override.clone())
    };

    let cascade_path = match resolve_cascade_path(&cascade_override) {
        Some(p) => p,
        None => {
            finish_init_error(&state, "could not find haarcascade_frontalface_default.xml");
            return;
        }
    };

    let mut classifier = match CascadeClassifier::new(&cascade_path) {
        Ok(c) => c,
        Err(e) => { finish_init_error(&state, &format!("CascadeClassifier::new failed: {e}")); return; }
    };

    let persisted_cache = load_face_cache(&known_faces_dir).unwrap_or_default();
    let samples = match load_known_face_samples(&mut classifier, &known_faces_dir, &persisted_cache) {
        Ok(s) => s,
        Err(e) => { finish_init_error(&state, &e.to_string()); return; }
    };
    let sample_count = samples.len() as i32;

    // Save updated cache
    let cache_entries: Vec<CacheEntry> = samples.iter().filter_map(|s| {
        sample_to_cache_entry(s).ok().map(|mut e| { e.path = s.path.clone(); e.signature = s.signature.clone(); e })
    }).collect();
    if let Err(e) = save_face_cache(&known_faces_dir, &cache_entries) {
        log::warn!("face: failed to save cache: {e}");
    }

    *models.lock().unwrap() = Some(Models { classifier, samples });

    let mut s = state.lock().unwrap();
    s.initializing = false;
    s.available = true;
    s.known_faces_count = sample_count;
    s.tracks = vec![];
    s.message = if s.enabled {
        format!("ready ({sample_count} known face sample(s))")
    } else {
        "disabled".to_string()
    };
    log::info!("face: ready with {sample_count} known face sample(s)");
}

fn finish_init_error(state: &Mutex<ServiceState>, msg: &str) {
    let mut s = state.lock().unwrap();
    s.initializing = false;
    s.available = false;
    s.message = format!("initialization failed: {msg}");
    log::error!("face: {}", s.message);
}

// ─── Known face loading ────────────────────────────────────────────────────────

fn load_known_face_samples(
    classifier: &mut CascadeClassifier,
    known_faces_dir: &str,
    persisted_cache: &HashMap<String, CacheEntry>,
) -> anyhow::Result<Vec<KnownFaceSample>> {
    let dir = std::path::Path::new(known_faces_dir);
    if !dir.exists() {
        log::warn!("face: known_faces dir not found: {known_faces_dir} — all detections labelled Unknown");
        return Ok(vec![]);
    }

    let mut persons: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    persons.sort();

    let mut samples = Vec::new();

    for person in &persons {
        let person_dir = dir.join(person);
        log::info!("face: loading known faces for {person}");
        let mut images: Vec<std::fs::DirEntry> = std::fs::read_dir(&person_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name().to_string_lossy().to_lowercase();
                name.ends_with(".jpg") || name.ends_with(".jpeg") || name.ends_with(".png")
            })
            .collect();
        images.sort_by_key(|e| e.file_name());

        let mut count = 0;
        for img_entry in &images {
            let img_path = img_entry.path();
            let rel = img_path.strip_prefix(dir).unwrap_or(&img_path).to_string_lossy().to_string();
            let meta = img_entry.metadata()?;
            let sig = format!(
                "{}:{}",
                meta.len(),
                meta.modified().map(|t| t.duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)).unwrap_or(0)
            );

            // Check persisted cache
            if let Some(cached) = persisted_cache.get(&rel) {
                if cached.signature == sig {
                    if cached.no_face {
                        continue; // known-bad image
                    }
                    if let Ok(mat) = sample_from_cache_entry(cached) {
                        log::info!("face: cache hit for {rel}");
                        samples.push(KnownFaceSample { name: person.clone(), path: rel.clone(), signature: sig, mat });
                        count += 1;
                        continue;
                    }
                }
            }

            // Load and process
            log::info!("face: reading known face image {}", img_path.display());
            let path_str = img_path.to_string_lossy().to_string();
            let img = imgcodecs::imread(&path_str, imgcodecs::IMREAD_COLOR).unwrap_or_default();
            if img.empty() {
                log::warn!("face: skipping unreadable image {path_str}");
                continue;
            }
            let mut gray = Mat::default();
            if imgproc::cvt_color_def(&img, &mut gray, imgproc::COLOR_BGR2GRAY).is_err() { continue; }

            let mut face_rects: Vector<Rect> = Vector::new();
            let _ = classifier.detect_multi_scale(
                &gray,
                &mut face_rects,
                1.10, 5, 0,
                Size::new(36, 36),
                Size::new(gray.cols(), gray.rows()),
            );
            if face_rects.is_empty() { continue; }

            // Largest face
            let best_rect = face_rects.iter()
                .max_by_key(|r| r.width * r.height)
                .unwrap();

            let normalized = match normalized_face_from_gray(&gray, &best_rect) {
                Some(m) => m,
                None => continue,
            };

            samples.push(KnownFaceSample { name: person.clone(), path: rel.clone(), signature: sig, mat: normalized });
            count += 1;
        }
        log::info!("face: loaded {count} sample(s) for {person}");
    }

    log::info!("face: finished loading, total samples={}", samples.len());
    Ok(samples)
}

// ─── Cache ────────────────────────────────────────────────────────────────────

const CACHE_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Clone, Default)]
struct CacheEntry {
    path: String,
    signature: String,
    #[serde(default)]
    no_face: bool,
    #[serde(default)]
    rows: i32,
    #[serde(default)]
    cols: i32,
    #[serde(default)]
    mat_type: i32,
    #[serde(default)]
    data: String,
}

#[derive(Serialize, Deserialize)]
struct FaceCache {
    version: u32,
    entries: Vec<CacheEntry>,
}

fn cache_file_path(known_faces_dir: &str) -> PathBuf {
    PathBuf::from(known_faces_dir).join(".face_sample_cache.json")
}

fn load_face_cache(known_faces_dir: &str) -> anyhow::Result<HashMap<String, CacheEntry>> {
    let path = cache_file_path(known_faces_dir);
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
        Err(e) => return Err(e.into()),
    };
    let cache: FaceCache = serde_json::from_slice(&data)?;
    if cache.version != CACHE_VERSION {
        log::info!("face: cache version mismatch; regenerating");
        return Ok(HashMap::new());
    }
    Ok(cache.entries.into_iter().map(|e| (e.path.clone(), e)).collect())
}

fn save_face_cache(known_faces_dir: &str, entries: &[CacheEntry]) -> anyhow::Result<()> {
    let path = cache_file_path(known_faces_dir);
    let mut sorted = entries.to_vec();
    sorted.sort_by(|a, b| a.path.cmp(&b.path));
    let cache = FaceCache { version: CACHE_VERSION, entries: sorted };
    let json = serde_json::to_vec_pretty(&cache)?;
    std::fs::write(&path, json)?;
    log::info!("face: saved {} cache entries to {}", entries.len(), path.display());
    Ok(())
}

fn sample_to_cache_entry(sample: &KnownFaceSample) -> anyhow::Result<CacheEntry> {
    let data = sample.mat.data_bytes()?;
    Ok(CacheEntry {
        path: sample.path.clone(),
        signature: sample.signature.clone(),
        no_face: false,
        rows: sample.mat.rows(),
        cols: sample.mat.cols(),
        mat_type: sample.mat.typ(),
        data: BASE64.encode(data),
    })
}

fn sample_from_cache_entry(entry: &CacheEntry) -> anyhow::Result<Mat> {
    if entry.rows <= 0 || entry.cols <= 0 || entry.data.is_empty() {
        return Err(anyhow::anyhow!("invalid cache entry"));
    }
    let bytes = BASE64.decode(&entry.data)?;
    let mut mat = unsafe { Mat::new_rows_cols(entry.rows, entry.cols, entry.mat_type)? };
    let dst = mat.data_bytes_mut()?;
    if dst.len() != bytes.len() {
        return Err(anyhow::anyhow!("cache byte count mismatch: {} vs {}", bytes.len(), dst.len()));
    }
    dst.copy_from_slice(&bytes);
    Ok(mat)
}

// ─── Cascade path resolution ──────────────────────────────────────────────────

fn resolve_cascade_path(override_path: &str) -> Option<String> {
    let candidate = override_path.trim();
    if !candidate.is_empty() {
        if std::path::Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }
    let standard_paths = [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ];
    for p in &standard_paths {
        if std::path::Path::new(p).exists() {
            return Some(p.to_string());
        }
    }
    None
}

// ─── MJPEG frame parser ───────────────────────────────────────────────────────

/// Extract complete JPEG frames from a byte buffer.
/// Returns (list of JPEG frames, unconsumed prefix).
pub fn extract_jpeg_frames(buf: &[u8]) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut frames = Vec::new();
    let mut remaining = buf;

    loop {
        // Find SOI marker
        let start = match remaining.windows(2).position(|w| w == [0xFF, 0xD8]) {
            Some(i) => i,
            None => break,
        };
        let search_from = start + 2;
        if search_from >= remaining.len() {
            remaining = &remaining[start..];
            break;
        }
        // Find EOI marker after SOI
        let end_offset = match remaining[search_from..].windows(2).position(|w| w == [0xFF, 0xD9]) {
            Some(i) => i,
            None => {
                remaining = &remaining[start..];
                break;
            }
        };
        let end = search_from + end_offset + 2;
        frames.push(remaining[start..end].to_vec());
        remaining = &remaining[end..];
    }

    (frames, remaining.to_vec())
}

/// Spawn a thread that reads MJPEG from `reader`, extracts frames and calls
/// `on_frame` for each.  The thread runs until EOF or `on_frame` returns false.
pub fn spawn_mjpeg_reader<R, F>(reader: R, on_frame: F) -> std::thread::JoinHandle<()>
where
    R: Read + Send + 'static,
    F: Fn(Vec<u8>, u64) -> bool + Send + 'static,
{
    std::thread::spawn(move || {
        let mut reader = reader;
        let mut buf = Vec::new();
        let mut tmp = [0u8; 65536];
        let mut seq = 0u64;
        loop {
            match reader.read(&mut tmp) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    buf.extend_from_slice(&tmp[..n]);
                    let (frames, leftover) = extract_jpeg_frames(&buf);
                    buf = leftover;
                    // Keep buffer bounded to 5 MB.
                    if buf.len() > 5 * 1024 * 1024 {
                        buf.clear();
                    }
                    for frame in frames {
                        seq += 1;
                        if !on_frame(frame, seq) {
                            return;
                        }
                    }
                }
            }
        }
    })
}
