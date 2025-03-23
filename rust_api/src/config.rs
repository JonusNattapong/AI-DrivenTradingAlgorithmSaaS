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
pub struct SecurityConfig {
    pub jwt_secret: String,
    pub token_expiration: u64, // in seconds
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub python: PythonConfig,
    pub security: SecurityConfig,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
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

        // Security configuration
        let jwt_secret = env::var("JWT_SECRET")
            .map_err(|_| ConfigError::EnvVarNotFound("JWT_SECRET".to_string()))?;
        let token_expiration = env::var("TOKEN_EXPIRATION")
            .unwrap_or_else(|_| "86400".to_string()) // Default: 24 hours
            .parse::<u64>()
            .map_err(|e| ConfigError::ParseError(format!("Failed to parse TOKEN_EXPIRATION: {}", e)))?;

        Ok(Config {
            server: ServerConfig { host, port },
            python: PythonConfig { model_path, data_cache_path },
            security: SecurityConfig { jwt_secret, token_expiration },
        })
    }
}