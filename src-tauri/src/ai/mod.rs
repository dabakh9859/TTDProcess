//! AI sidecar module.
//!
//! Owns a long-running Python child process (`sidecars/ttd-ai/service.py`)
//! that hosts SAITS for training and inference. Communication is JSON-RPC
//! over stdin/stdout, one JSON object per line. This module exposes an
//! ergonomic async client (`AiSidecar::call`) used by Tauri commands.

pub mod sidecar;

pub use sidecar::AiSidecar;
