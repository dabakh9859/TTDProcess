//! Spawns and talks to the Python AI sidecar over JSON-RPC (stdin/stdout).
//!
//! Lifecycle:
//!   - The sidecar is spawned lazily on the first `call()`.
//!   - A background reader thread parses every line of the child's stdout,
//!     routes responses to the matching pending call, and forwards events
//!     to the frontend via Tauri's event system.
//!   - The child stays alive for the rest of the app session. We don't
//!     restart it automatically — if it dies, subsequent calls will fail
//!     and the user will need to reload the app (rare, only on Python crashes).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::oneshot;

/// One in-flight request waiting on a response from the sidecar.
type Pending = oneshot::Sender<Result<Value, String>>;

struct Inner {
    /// Child handle so we can wait/kill on shutdown. None until the first call.
    child: Option<Child>,
    /// Write side of the child's stdin. None when the child isn't running yet.
    stdin: Option<ChildStdin>,
    /// req_id → channel waiting for the response.
    pending: HashMap<u64, Pending>,
}

impl Inner {
    fn new() -> Self {
        Self { child: None, stdin: None, pending: HashMap::new() }
    }
}

/// Async client to the AI sidecar. Cheap to clone (it's all Arc inside).
pub struct AiSidecar {
    inner: Arc<Mutex<Inner>>,
    next_id: AtomicU64,
}

impl AiSidecar {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::new())),
            next_id: AtomicU64::new(1),
        }
    }

    /// Send a JSON-RPC call and await the response. Spawns the sidecar
    /// process on first invocation. Mid-call progress events are *not*
    /// returned here — they are emitted on the Tauri event bus as
    /// `"ai:event"` payloads.
    pub async fn call(
        &self,
        app: &AppHandle,
        method: &str,
        params: Value,
    ) -> Result<Value, String> {
        self.ensure_spawned(app)?;

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();

        // Register this call as pending, then write the request line.
        {
            let mut inner = self.inner.lock().map_err(|e| format!("sidecar lock poisoned: {e}"))?;
            inner.pending.insert(id, tx);

            let stdin = inner.stdin.as_mut()
                .ok_or_else(|| "sidecar stdin gone".to_string())?;
            let request = json!({ "id": id, "method": method, "params": params });
            let line = request.to_string() + "\n";
            stdin.write_all(line.as_bytes())
                .map_err(|e| format!("write to sidecar failed: {e}"))?;
            stdin.flush().map_err(|e| format!("flush sidecar failed: {e}"))?;
        }

        rx.await.map_err(|_| "sidecar dropped response channel".to_string())?
    }

    /// Spawn the sidecar process on first use. Subsequent calls are no-ops.
    fn ensure_spawned(&self, app: &AppHandle) -> Result<(), String> {
        let mut inner = self.inner.lock().map_err(|e| format!("sidecar lock poisoned: {e}"))?;
        if inner.child.is_some() {
            return Ok(());
        }

        let python = python_executable(app);
        let service = service_script(app);

        if !service.exists() {
            return Err(format!("sidecar script not found at {}", service.display()));
        }

        // Force UTF-8 on Python's stdio; the parent reads stdout line-by-line
        // and assumes UTF-8.
        let mut cmd = Command::new(&python);
        cmd.arg(&service)
           .env("PYTHONIOENCODING", "utf-8")
           .env("PYTHONUNBUFFERED", "1")
           .stdin(Stdio::piped())
           .stdout(Stdio::piped())
           // null instead of inherit: GUI release builds have no console,
           // so an "inherited" stderr ends up unbuffered into the void and
           // eventually fills the OS pipe buffer, blocking the child after
           // ~100 log lines. The pypots logger logs every epoch, so this
           // hangs reproducibly during training. Discarding stderr is the
           // simplest fix; Phase 1.B can revisit with a tee-to-file logger.
           .stderr(Stdio::null());

        // Hide console window on Windows release builds.
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = cmd.spawn()
            .map_err(|e| format!("failed to spawn sidecar ({}): {e}", python.display()))?;

        let stdin = child.stdin.take()
            .ok_or_else(|| "child stdin missing".to_string())?;
        let stdout = child.stdout.take()
            .ok_or_else(|| "child stdout missing".to_string())?;

        inner.child = Some(child);
        inner.stdin = Some(stdin);

        // Start the reader thread that fans responses back to pending channels.
        let inner_arc = Arc::clone(&self.inner);
        let app_handle = app.clone();
        thread::spawn(move || reader_loop(stdout, inner_arc, app_handle));

        Ok(())
    }
}

/// Background thread: reads one JSON object per line from the sidecar's
/// stdout and dispatches it (response → pending channel, event → Tauri bus).
fn reader_loop(
    stdout: std::process::ChildStdout,
    inner: Arc<Mutex<Inner>>,
    app: AppHandle,
) {
    let reader = BufReader::new(stdout);
    for line in reader.lines() {
        let Ok(line) = line else { break };
        if line.is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("ai sidecar: bad JSON on stdout: {e}: {line}");
                continue;
            }
        };

        // Mid-call progress event — forward to the frontend, no pending to wake.
        if msg.get("event").is_some() {
            let _ = app.emit("ai:event", &msg);
            continue;
        }

        let id = match msg.get("id").and_then(|v| v.as_u64()) {
            Some(i) => i,
            None => continue,
        };

        let pending = {
            let mut g = match inner.lock() {
                Ok(g) => g,
                Err(_) => break,
            };
            g.pending.remove(&id)
        };
        let Some(tx) = pending else { continue };

        if let Some(err) = msg.get("error") {
            let m = err.get("message").and_then(|v| v.as_str()).unwrap_or("unknown error");
            let _ = tx.send(Err(format!("sidecar error: {m}")));
        } else if let Some(result) = msg.get("result") {
            let _ = tx.send(Ok(result.clone()));
        } else {
            let _ = tx.send(Err(format!("sidecar response missing result/error: {msg}")));
        }
    }
}

/// Resolve which Python interpreter to use.
///
/// Priority:
///   1. `TTD_AI_PYTHON` env var (override for dev/testing).
///   2. The Python embeddable shipped with the MSI in the resource_dir
///      (`<install_dir>/resources/sidecars/python-embed/python.exe`).
///   3. The dev conda env (Aziz's machine — convenience for local dev).
///   4. `python` from PATH — last resort; expected to fail on a clean machine.
fn python_executable(app: &AppHandle) -> PathBuf {
    if let Ok(p) = std::env::var("TTD_AI_PYTHON") {
        return PathBuf::from(p);
    }
    // Bundled python embed: resource_dir/sidecars/python-embed/python.exe.
    if let Ok(res_dir) = app.path().resource_dir() {
        let bundled = res_dir.join("sidecars").join("python-embed").join("python.exe");
        if bundled.exists() {
            return bundled;
        }
    }
    let dev_default = PathBuf::from(r"C:\Users\Aziz\miniconda3\envs\ttd-ai\python.exe");
    if dev_default.exists() {
        return dev_default;
    }
    PathBuf::from("python")
}

/// Resolve the path to `service.py`. Priority:
///   1. `TTD_AI_SERVICE` env var (dev override).
///   2. Resource dir of the installed MSI
///      (`<install_dir>/resources/sidecars/ttd-ai/service.py`).
///   3. `CARGO_MANIFEST_DIR/sidecars/ttd-ai/service.py` — dev tree.
fn service_script(app: &AppHandle) -> PathBuf {
    if let Ok(p) = std::env::var("TTD_AI_SERVICE") {
        return PathBuf::from(p);
    }

    // Release: bundled into the resource dir by Tauri's bundle.resources.
    if let Ok(res_dir) = app.path().resource_dir() {
        let bundled = res_dir.join("sidecars").join("ttd-ai").join("service.py");
        if bundled.exists() {
            return bundled;
        }
    }

    // Dev: project root has src-tauri/sidecars/ttd-ai/service.py
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("sidecars")
        .join("ttd-ai")
        .join("service.py")
}
