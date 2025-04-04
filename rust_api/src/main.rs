//! Main entry point for the trading algorithm API server

use actix_web::{App, HttpServer, middleware, web};
use env_logger::Env;
use log::{error, info};
use std::sync::{Arc, Mutex};

mod api;
mod config;
mod python;

use config::Config;
use python::ModelManager;

// Custom error type for application errors
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),
    
    #[error("Model manager error: {0}")]
    ModelManager(String),
    
    #[error("Server error: {0}")]
    Server(#[from] std::io::Error),
}

#[actix_web::main]
async fn main() -> Result<(), AppError> {
    // Initialize logger with custom format
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .format_module_path(true)
        .init();
    
    info!("Starting AI-Driven Trading Algorithm SaaS");
    
    // Load configuration
    info!("Loading configuration...");
    let config = Config::from_env()
        .map(Arc::new)
        .map_err(AppError::Config)?;
    
    // Initialize Python model manager
    info!("Initializing model manager...");
    let model_manager = ModelManager::new(Arc::clone(&config))
        .map(|manager| web::Data::new(Mutex::new(manager)))
        .map_err(|e| AppError::ModelManager(e.to_string()))?;
    
    // Start HTTP server
    let server_config = config.server.clone();
    info!("Starting server at http://{}:{}", server_config.host, server_config.port);
    
    let server = HttpServer::new(move || {
        App::new()
            // Enable verbose logger middleware
            .wrap(
                middleware::Logger::new(
                    r#"%a "%r" %s %b "%{Referer}i" "%{User-Agent}i" %T"#
                )
            )
            // Enable compression
            .wrap(middleware::Compress::default())
            // Register model manager as app data
            .app_data(model_manager.clone())
            // Configure API routes
            .configure(api::configure_routes)
    })
    .workers(4) // Fixed number of workers
    .shutdown_timeout(30) // Allow 30 seconds for graceful shutdown
    .bind((server_config.host, server_config.port))
    .map_err(AppError::Server)?;
    
    // Run server
    info!("Server is ready to accept connections");
    server.run()
        .await
        .map_err(AppError::Server)?;
        
    info!("Server shutdown complete");
    Ok(())
}
