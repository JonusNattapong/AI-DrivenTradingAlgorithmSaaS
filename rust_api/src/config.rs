//! Configuration module for the API server

use serde::Deserialize;
use std::env;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Environment variable not found: {0}")]
    EnvVarNotFound(String),
    
    #[error("Failed to parse environment variable: {0}")]
    ParseError(String),
}

#[derive(Clone, Debug, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PythonConfig {
    pub model_path: String,
    pub data_cache_path: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub python: PythonConfig,
}

use std::path::Path;

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        // Load .env file from the parent directory (project root)
        let env_path = Path::new("..").join(".env");
        match dotenv::from_path(env_path.as_path()) {
            Ok(_) => log::info!("Loaded .env file from: {}", env_path.display()),
            Err(e) => log::warn!("Could not load .env file from {}: {}", env_path.display(), e),
        };

        // Server configuration
        let host = env::var("SERVER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port = env::var("SERVER_PORT")
            .unwrap_or_else(|_| "8080".to_string())
            .parse::<u16>()
            .map_err(|e| ConfigError::ParseError(format!("Failed to parse SERVER_PORT: {}", e)))?;

        // Python configuration
        let model_path = env::var("PYTHON_MODEL_PATH")
            .unwrap_or_else(|_| "./python_model".to_string());
        let data_cache_path = env::var("PYTHON_DATA_CACHE_PATH")
            .unwrap_or_else(|_| "./python_model/data/cache".to_string());

        Ok(Config {
            server: ServerConfig { host, port },
            python: PythonConfig { model_path, data_cache_path },
        })
    }
}
