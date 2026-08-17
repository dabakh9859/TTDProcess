use anyhow::Result;
use polars::prelude::*;
use std::collections::HashMap;

/// Convert a DataFrame (or a page/slice of one) into a JSON-serializable
/// structure: Vec of rows, where each row is a HashMap<column_name, Value>.
pub fn dataframe_page_to_json(
    df: &DataFrame,
) -> Result<Vec<HashMap<String, serde_json::Value>>> {
    let col_names: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
    let mut rows: Vec<HashMap<String, serde_json::Value>> = Vec::with_capacity(df.height());

    for row_idx in 0..df.height() {
        let mut row_map: HashMap<String, serde_json::Value> = HashMap::new();

        for (col_idx, col_name) in col_names.iter().enumerate() {
            let series = &df.get_columns()[col_idx];
            let value = anyvalue_to_json(series.get(row_idx).unwrap());
            row_map.insert(col_name.clone(), value);
        }

        rows.push(row_map);
    }

    Ok(rows)
}

/// Convert a Polars AnyValue into a serde_json::Value.
fn anyvalue_to_json(val: AnyValue) -> serde_json::Value {
    match val {
        AnyValue::Null => serde_json::Value::Null,
        AnyValue::Boolean(b) => serde_json::Value::Bool(b),
        AnyValue::Int8(i) => serde_json::json!(i),
        AnyValue::Int16(i) => serde_json::json!(i),
        AnyValue::Int32(i) => serde_json::json!(i),
        AnyValue::Int64(i) => serde_json::json!(i),
        AnyValue::UInt8(i) => serde_json::json!(i),
        AnyValue::UInt16(i) => serde_json::json!(i),
        AnyValue::UInt32(i) => serde_json::json!(i),
        AnyValue::UInt64(i) => serde_json::json!(i),
        AnyValue::Float32(f) => {
            if f.is_nan() || f.is_infinite() {
                serde_json::Value::Null
            } else {
                serde_json::json!(f)
            }
        }
        AnyValue::Float64(f) => {
            if f.is_nan() || f.is_infinite() {
                serde_json::Value::Null
            } else {
                serde_json::json!(f)
            }
        }
        AnyValue::String(s) => serde_json::Value::String(s.to_string()),
        AnyValue::StringOwned(s) => serde_json::Value::String(s.to_string()),
        _ => serde_json::Value::String(format!("{}", val)),
    }
}
