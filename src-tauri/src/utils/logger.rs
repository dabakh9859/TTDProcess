use crate::core::types::{LogEntry, LogLevel};

/// Add a new log entry to the logs vector.
pub fn add_log(logs: &mut Vec<LogEntry>, level: LogLevel, message: String) {
    let entry = LogEntry {
        level,
        message,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };
    logs.push(entry);
}
