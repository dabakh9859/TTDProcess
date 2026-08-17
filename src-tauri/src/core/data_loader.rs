use calamine::{open_workbook, Data, Reader, Xlsx};
use polars::prelude::*;
use anyhow::{Context, Result};
use std::path::Path;

/// Detect old-style sensor column names (e.g. `SF_4-TA-1`) and return the
/// new-style equivalent (`SF_4a-TA-1`).  Returns `None` for columns that
/// already follow the new convention or aren't sensor columns.
fn legacy_to_new_name(col: &str) -> Option<String> {
    let prefix = if col.starts_with("SF_") {
        "SF_"
    } else if col.starts_with("TC_") {
        "TC_"
    } else {
        return None;
    };
    let rest = &col[prefix.len()..];
    let dash = rest.find('-')?;
    let station = &rest[..dash];
    if station.is_empty() || !station.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let suffix = &rest[dash..];
    Some(format!("{}{}a{}", prefix, station, suffix))
}

/// Harmonise legacy sensor column names so that data from files with different
/// naming conventions (e.g. pre-2021 `SF_4-TA-1` vs post-2021 `SF_4a-TA-1`)
/// forms a single continuous series.
///
/// * When both old and new columns exist → coalesce (prefer new, fill from old)
///   and drop the old column.
/// * When only the old column exists → rename it to the new convention.
pub fn merge_legacy_sensor_columns(mut df: DataFrame) -> (DataFrame, usize) {
    let col_names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut to_drop: Vec<String> = Vec::new();
    let mut count = 0usize;

    for old_name in &col_names {
        if let Some(new_name) = legacy_to_new_name(old_name) {
            if col_names.contains(&new_name) {
                if let (Ok(new_col), Ok(old_col)) =
                    (df.column(&new_name).cloned(), df.column(old_name).cloned())
                {
                    let mask = new_col.is_not_null();
                    if let Ok(coalesced) = new_col.zip_with(&mask, &old_col) {
                        let _ = df.replace(&new_name, coalesced);
                        to_drop.push(old_name.clone());
                        count += 1;
                    }
                }
            } else {
                let _ = df.rename(old_name, new_name.as_str().into());
                count += 1;
            }
        }
    }

    if !to_drop.is_empty() {
        df = df.drop_many(&to_drop);
    }
    (df, count)
}

/// True if the path's extension is .csv or .dat (case-insensitive).
fn is_csv(file_path: &str) -> bool {
    Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("csv") || e.eq_ignore_ascii_case("dat"))
        .unwrap_or(false)
}

/// Returns the list of sheet names. For CSV files there are no sheets, so
/// we return a single "data" entry to keep the rest of the flow generic.
pub fn get_sheet_names(file_path: &str) -> Result<Vec<String>> {
    let path = Path::new(file_path);

    if !path.exists() {
        anyhow::bail!("File not found: {}", file_path);
    }

    if is_csv(file_path) {
        return Ok(vec!["data".to_string()]);
    }

    let workbook: Xlsx<_> =
        open_workbook(path).context("Could not open Excel file")?;

    Ok(workbook.sheet_names().to_vec())
}

/// Read a CSV file into a tolerant `String` regardless of encoding. Tries
/// UTF-8 first; on failure falls back to Latin-1 (each byte → its Unicode
/// codepoint of the same value), which never errors and preserves all
/// ASCII content perfectly. Most field-data exports are either UTF-8 or
/// Windows-1252/Latin-1 — this covers both without bringing in a heavy
/// encoding crate.
///
/// Strips the UTF-8 BOM (EF BB BF) if present. Excel "Save As CSV UTF-8"
/// and PowerShell `Out-File -Encoding utf8` both add one by default — left
/// in place, the BOM gets attached to the first cell of the header row
/// (e.g. `\u{feff}DateSemih` instead of `DateSemih`) and silently breaks
/// every downstream column lookup.
fn read_csv_text(file_path: &str) -> Result<String> {
    let mut bytes = std::fs::read(file_path)
        .with_context(|| format!("Could not open CSV file: {}", file_path))?;
    if bytes.starts_with(b"\xEF\xBB\xBF") {
        bytes.drain(..3);
    }
    match String::from_utf8(bytes) {
        Ok(s) => Ok(s),
        Err(e) => {
            // Recover the bytes and map them via Latin-1.
            let bytes = e.into_bytes();
            Ok(bytes.iter().map(|&b| b as char).collect())
        }
    }
}

/// Read a CSV file as raw rows of strings (no header parsing yet). Auto-
/// detects the delimiter by looking at the first non-empty line: comma,
/// semicolon, or tab — covers ~all field-data exports we've seen.
///
/// Splits the row-parsing across threads via rayon: tokenizing 1M cells
/// is the bottleneck on wide flux-station exports, and it's trivially
/// parallel once the delimiter is chosen.
fn read_csv_raw(file_path: &str) -> Result<Vec<Vec<String>>> {
    use rayon::prelude::*;

    let text = read_csv_text(file_path)?;
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Ok(Vec::new());
    }

    // Detect the delimiter from the first non-empty line — same logic
    // as before, just kept sequential because it's cheap.
    let delim = lines
        .iter()
        .find(|l| !l.trim().is_empty())
        .map(|line| {
            let line = line.trim_end_matches('\r');
            let counts = [
                (',', line.matches(',').count()),
                (';', line.matches(';').count()),
                ('\t', line.matches('\t').count()),
            ];
            counts.iter().max_by_key(|(_, n)| *n).map(|(c, _)| *c).unwrap_or(',')
        })
        .unwrap_or(',');

    // Parse every line in parallel — each `split` + trim is pure CPU
    // work and lines are independent.
    let all_rows: Vec<Vec<String>> = lines
        .par_iter()
        .map(|raw_line| {
            let line = raw_line.trim_end_matches('\r');
            line.split(delim)
                .map(|s| s.trim().trim_matches('"').to_string())
                .collect()
        })
        .collect();
    Ok(all_rows)
}

/// Preview the first `num_rows` rows of a sheet as Vec<Vec<String>>.
pub fn preview_file(
    file_path: &str,
    sheet_name: &str,
    num_rows: usize,
) -> Result<Vec<Vec<String>>> {
    let path = Path::new(file_path);

    if !path.exists() {
        anyhow::bail!("File not found: {}", file_path);
    }

    // CSV path — sheet_name is ignored (single-sheet format).
    if is_csv(file_path) {
        let rows = read_csv_raw(file_path)?;
        return Ok(rows.into_iter().take(num_rows).collect());
    }

    let mut workbook: Xlsx<_> =
        open_workbook(path).context("Could not open Excel file")?;

    let range = workbook
        .worksheet_range(sheet_name)
        .context(format!("Could not read sheet: {}", sheet_name))?;

    let mut preview: Vec<Vec<String>> = Vec::new();

    for (idx, row) in range.rows().enumerate() {
        if idx >= num_rows {
            break;
        }

        let row_data: Vec<String> = row
            .iter()
            .map(|cell| {
                let cell_str = cell_to_string(cell);

                // Attempt to convert Excel serial dates (range ~40000-50000 => years 2009-2036)
                if let Ok(num) = cell_str.trim().parse::<f64>() {
                    if num > 40000.0 && num < 50000.0 {
                        return excel_date_to_string(num);
                    }
                }
                cell_str
            })
            .collect();

        preview.push(row_data);
    }

    Ok(preview)
}

/// Load a tabular file (Excel or CSV) into a Polars DataFrame.
///
/// `header_row` is the 0-based row index for column names.
/// `data_start_row` is the 0-based row index where actual data begins.
/// For CSV files, `sheet_name` is ignored.
pub fn load_excel_file(
    file_path: &str,
    sheet_name: &str,
    header_row: usize,
    data_start_row: usize,
) -> Result<DataFrame> {
    let path = Path::new(file_path);

    if !path.exists() {
        anyhow::bail!("File not found: {}", file_path);
    }

    // CSV path — read all rows then split into header + data using the same
    // header_row / data_start_row semantics as Excel.
    if is_csv(file_path) {
        let all_rows = read_csv_raw(file_path)?;
        if all_rows.len() <= header_row {
            anyhow::bail!("No header found at row {}", header_row);
        }
        let header = all_rows[header_row].clone();
        let rows: Vec<Vec<String>> = all_rows
            .into_iter()
            .enumerate()
            .filter_map(|(i, r)| if i >= data_start_row { Some(r) } else { None })
            .collect();
        if rows.is_empty() {
            anyhow::bail!("No data rows found starting at row {}", data_start_row);
        }
        return create_dataframe_from_rows(&header, &rows);
    }

    let mut workbook: Xlsx<_> =
        open_workbook(path).context("Could not open Excel file")?;

    let range = workbook
        .worksheet_range(sheet_name)
        .context(format!("Could not read sheet: {}", sheet_name))?;

    let mut header: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<String>> = Vec::new();

    for (row_idx, row) in range.rows().enumerate() {
        let row_data: Vec<String> = row.iter().map(|cell| cell_to_string(cell)).collect();

        if row_idx == header_row {
            header = row_data;
        } else if row_idx >= data_start_row {
            rows.push(row_data);
        }
    }

    if header.is_empty() {
        anyhow::bail!("No header found at row {}", header_row);
    }

    if rows.is_empty() {
        anyhow::bail!("No data rows found starting at row {}", data_start_row);
    }

    create_dataframe_from_rows(&header, &rows)
}

/// Export a DataFrame to CSV.
pub fn export_to_csv(df: &DataFrame, path: &str) -> Result<()> {
    let mut file = std::fs::File::create(path)?;
    let mut df_mut = df.clone();
    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_mut)?;
    Ok(())
}

/// Export a DataFrame to Excel.
/// Excel sheet names: max 31 chars, none of []:*?/\ , and non-empty.
fn sanitize_sheet_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if matches!(c, '[' | ']' | ':' | '*' | '?' | '/' | '\\') { '_' } else { c })
        .collect();
    let trimmed = cleaned.trim();
    let out: String = trimmed.chars().take(31).collect();
    if out.is_empty() { "Feuille".to_string() } else { out }
}

/// Write a DataFrame (header row + data) into an already-created worksheet.
fn write_df_to_worksheet(
    worksheet: &mut rust_xlsxwriter::Worksheet,
    df: &DataFrame,
) -> Result<()> {
    for (col_idx, col_name) in df.get_column_names().iter().enumerate() {
        worksheet.write_string(0, col_idx as u16, *col_name)?;
    }
    for row_idx in 0..df.height() {
        for (col_idx, series) in df.get_columns().iter().enumerate() {
            let value = series.get(row_idx).unwrap();
            match value {
                AnyValue::Float64(f) => { worksheet.write_number((row_idx + 1) as u32, col_idx as u16, f)?; }
                AnyValue::Float32(f) => { worksheet.write_number((row_idx + 1) as u32, col_idx as u16, f as f64)?; }
                AnyValue::Int64(i) => { worksheet.write_number((row_idx + 1) as u32, col_idx as u16, i as f64)?; }
                AnyValue::Int32(i) => { worksheet.write_number((row_idx + 1) as u32, col_idx as u16, i as f64)?; }
                AnyValue::String(s) => { worksheet.write_string((row_idx + 1) as u32, col_idx as u16, s)?; }
                AnyValue::Null => { /* leave cell empty */ }
                other => { worksheet.write_string((row_idx + 1) as u32, col_idx as u16, &format!("{}", other))?; }
            }
        }
    }
    Ok(())
}

const EXCEL_MAX_ROWS: usize = 1_048_576;

pub fn export_to_excel(df: &DataFrame, path: &str, sheet_name: &str) -> Result<()> {
    if df.height() + 1 > EXCEL_MAX_ROWS {
        let csv_path = Path::new(path).with_extension("csv");
        export_to_csv(df, csv_path.to_str().unwrap())?;
        anyhow::bail!(
            "Le dataset contient {} lignes (limite Excel : {}). \
             Fichier exporté automatiquement en CSV : {}",
            df.height(), EXCEL_MAX_ROWS - 1,
            csv_path.display()
        );
    }
    use rust_xlsxwriter::*;
    let mut workbook = Workbook::new();
    let worksheet = workbook.add_worksheet().set_name(&sanitize_sheet_name(sheet_name))?;
    write_df_to_worksheet(worksheet, df)?;
    workbook.save(path)?;
    Ok(())
}

/// Write several DataFrames into ONE workbook, one sheet each.
/// If any sheet exceeds the Excel row limit, all sheets are exported as
/// individual CSV files in the same directory instead.
pub fn export_to_excel_multi(sheets: &[(String, &DataFrame)], path: &str) -> Result<()> {
    let oversized: Vec<&str> = sheets.iter()
        .filter(|(_, df)| df.height() + 1 > EXCEL_MAX_ROWS)
        .map(|(name, _)| name.as_str())
        .collect();
    if !oversized.is_empty() {
        let dir = Path::new(path).parent().unwrap_or(Path::new("."));
        for (name, df) in sheets {
            let safe = name.replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '_', "_");
            let csv_path = dir.join(format!("{}.csv", safe));
            export_to_csv(df, csv_path.to_str().unwrap())?;
        }
        anyhow::bail!(
            "Feuille(s) trop volumineuse(s) pour Excel ({}) : {}. \
             Tous les datasets ont été exportés en CSV dans {}.",
            EXCEL_MAX_ROWS - 1,
            oversized.join(", "),
            dir.display()
        );
    }
    use rust_xlsxwriter::*;
    let mut workbook = Workbook::new();
    for (name, df) in sheets {
        let worksheet = workbook.add_worksheet().set_name(&sanitize_sheet_name(name))?;
        write_df_to_worksheet(worksheet, df)?;
    }
    workbook.save(path)?;
    Ok(())
}

// =============================================================================
// Internal helpers
// =============================================================================

fn cell_to_string(cell: &Data) -> String {
    match cell {
        Data::Int(i) => i.to_string(),
        Data::Float(f) => f.to_string(),
        Data::String(s) => s.clone(),
        Data::Bool(b) => b.to_string(),
        Data::DateTime(dt) => format!("{}", dt),
        Data::Error(e) => format!("ERROR: {:?}", e),
        Data::Empty => String::new(),
        Data::DateTimeIso(s) => s.clone(),
        Data::DurationIso(s) => s.clone(),
    }
}

fn excel_date_to_string(excel_date: f64) -> String {
    use chrono::{Duration, NaiveDate};

    let base_date = NaiveDate::from_ymd_opt(1899, 12, 30).unwrap();
    let days = excel_date.floor() as i64;
    let fraction = excel_date - excel_date.floor();

    let date = base_date + Duration::days(days);

    let total_seconds = (fraction * 86400.0).round() as i64;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    format!(
        "{} {:02}:{:02}:{:02}",
        date.format("%Y-%m-%d"),
        hours,
        minutes,
        seconds
    )
}

/// Parse a timestamp cell to microseconds since epoch.
///
/// Delegates to `timestamp_utils::parse_string_to_datetime` — the canonical
/// format list — instead of keeping a second, shorter one here. The two had
/// drifted apart: this loader accepted only space-separated formats, while our
/// OWN exports write ISO-8601 (`2019-04-10T13:00:00.000000`). Re-importing an
/// exported T600 / sap-flow file therefore parsed every TIMESTAMP as null, and
/// `drop_null_timestamp_rows` then dropped every row — a silent total data
/// loss that also wiped the session's pipeline results.
fn parse_datetime_flexible(s: &str) -> Option<i64> {
    crate::core::timestamp_utils::parse_string_to_datetime(s)
        .map(|dt| dt.timestamp_micros())
}

fn is_numeric_column(data: &[String]) -> bool {
    let non_empty: Vec<&String> = data.iter().filter(|s| !s.is_empty()).collect();
    if non_empty.is_empty() {
        return false;
    }
    let numeric_count = non_empty.iter().filter(|s| s.parse::<f64>().is_ok()).count();
    numeric_count as f64 / non_empty.len() as f64 > 0.8
}

fn is_datetime_column(data: &[String]) -> bool {
    let non_empty: Vec<&String> = data.iter().filter(|s| !s.is_empty()).collect();
    if non_empty.is_empty() {
        return false;
    }
    let datetime_count = non_empty
        .iter()
        .filter(|s| parse_datetime_flexible(s).is_some())
        .count();
    datetime_count as f64 / non_empty.len() as f64 > 0.8
}

fn create_dataframe_from_rows(header: &[String], rows: &[Vec<String>]) -> Result<DataFrame> {
    use rayon::prelude::*;

    if header.is_empty() {
        anyhow::bail!("Empty header");
    }
    if rows.is_empty() {
        anyhow::bail!("No data rows");
    }

    // De-duplicate column names — Polars rejects DataFrames with repeated
    // names. Field-data exports sometimes duplicate columns (e.g. two
    // "Timestamp" columns or repeated metrics). Pandas-style fix: append
    // "_2", "_3", … to subsequent occurrences while preserving the first.
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let header: Vec<String> = header
        .iter()
        .map(|name| {
            let trimmed = name.trim().to_string();
            let count = seen.entry(trimmed.clone()).or_insert(0);
            *count += 1;
            if *count == 1 { trimmed } else { format!("{}_{}", trimmed, count) }
        })
        .collect();

    let num_rows = rows.len();

    // Parallelize the per-column work: extracting + type-detecting +
    // parsing each column is independent of the others. Wide flux-station
    // exports (600+ cols × 1k+ rows) used to spend several seconds here
    // single-threaded — rayon brings it to fractions of a second.
    //
    // We collect (col_idx, Series) pairs and re-sort to preserve the
    // original column order, since `par_iter` order is not deterministic.
    let columns_with_idx: Vec<(usize, Result<Series>)> = header
        .par_iter()
        .enumerate()
        .filter(|(_, name)| !name.trim().is_empty())
        .map(|(col_idx, col_name)| {
            let col_data: Vec<String> = rows
                .iter()
                .map(|row| row.get(col_idx).cloned().unwrap_or_default())
                .collect();

            if col_data.len() != num_rows {
                return (col_idx, Err(anyhow::anyhow!(
                    "Column '{}' has {} values instead of {}",
                    col_name, col_data.len(), num_rows
                )));
            }

            let series = if col_name.to_uppercase() == "TIMESTAMP" {
                // Parse TIMESTAMP column as Datetime(Microseconds).
                let timestamps: Vec<Option<i64>> = col_data
                    .iter()
                    .map(|s| {
                        let trimmed = s.trim();
                        if trimmed.is_empty() {
                            None
                        } else if let Ok(excel_date) = trimmed.parse::<f64>() {
                            let date_str = excel_date_to_string(excel_date);
                            parse_datetime_flexible(&date_str)
                        } else {
                            parse_datetime_flexible(trimmed)
                        }
                    })
                    .collect();
                let ts_series = Series::new(col_name.as_str(), timestamps);
                ts_series
                    .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                    .unwrap_or(ts_series)
            } else if is_datetime_column(&col_data) {
                Series::new(col_name.as_str(), col_data)
            } else if is_numeric_column(&col_data) {
                let numeric_data: Vec<Option<f64>> = col_data
                    .iter()
                    .map(|s| {
                        let trimmed = s.trim();
                        if trimmed.is_empty() {
                            None
                        } else {
                            trimmed.parse::<f64>().ok()
                        }
                    })
                    .collect();
                Series::new(col_name.as_str(), numeric_data)
            } else {
                Series::new(col_name.as_str(), col_data)
            };
            (col_idx, Ok(series))
        })
        .collect();

    // Re-sort to original header order, surface the first error if any.
    let mut sorted = columns_with_idx;
    sorted.sort_by_key(|(idx, _)| *idx);
    let mut series_vec: Vec<Series> = Vec::with_capacity(sorted.len());
    for (_, res) in sorted {
        series_vec.push(res?);
    }

    if series_vec.is_empty() {
        anyhow::bail!("No valid columns found");
    }

    DataFrame::new(series_vec).map_err(|e| anyhow::anyhow!("DataFrame creation error: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_numeric_column() {
        let data = vec!["1.5".into(), "2.3".into(), "3.7".into()];
        assert!(is_numeric_column(&data));

        let data = vec!["abc".into(), "def".into()];
        assert!(!is_numeric_column(&data));
    }

    #[test]
    fn test_parse_datetime_flexible() {
        assert!(parse_datetime_flexible("2020-01-01 12:00:00").is_some());
        assert!(parse_datetime_flexible("not a date").is_none());
    }
}
