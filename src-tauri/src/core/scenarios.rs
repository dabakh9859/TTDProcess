use anyhow::Result;
use std::fs;
use std::path::Path;

use crate::core::types::Scenario;

/// Save a scenario to a JSON file in the given folder.
#[allow(dead_code)]
pub fn save_scenario(scenario: &Scenario, folder: &str) -> Result<()> {
    fs::create_dir_all(folder)?;

    let filename = format!("{}/{}.json", folder, scenario.id);
    let json = serde_json::to_string_pretty(scenario)?;
    fs::write(filename, json)?;

    Ok(())
}

/// Load all scenarios from a folder. Returns them sorted by creation date (newest first).
#[allow(dead_code)]
pub fn load_scenarios(folder: &str) -> Result<Vec<Scenario>> {
    let mut scenarios = Vec::new();

    if !Path::new(folder).exists() {
        return Ok(scenarios);
    }

    for entry in fs::read_dir(folder)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            let content = fs::read_to_string(&path)?;
            if let Ok(scenario) = serde_json::from_str::<Scenario>(&content) {
                scenarios.push(scenario);
            }
        }
    }

    // Sort newest first
    scenarios.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    Ok(scenarios)
}

/// Load a single scenario by ID from the given folder.
#[allow(dead_code)]
pub fn load_scenario(scenario_id: &str, folder: &str) -> Result<Scenario> {
    let filename = format!("{}/{}.json", folder, scenario_id);
    let content = fs::read_to_string(&filename)?;
    let scenario: Scenario = serde_json::from_str(&content)?;
    Ok(scenario)
}

/// Delete a scenario by ID from the given folder.
#[allow(dead_code)]
pub fn delete_scenario(scenario_id: &str, folder: &str) -> Result<()> {
    let filename = format!("{}/{}.json", folder, scenario_id);
    if Path::new(&filename).exists() {
        fs::remove_file(filename)?;
    }
    Ok(())
}
