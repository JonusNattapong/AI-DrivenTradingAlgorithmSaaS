//! Main entry point for the trading algorithm API server

use actix_web::{App, HttpServer, middleware, web};
use std::sync::{Arc, Mutex};

mod api;
mod config;
mod python;

use config::Config;
use python::ModelManager;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logger
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));
    
    // Load configuration
    let config = match Config::from_env() {
        Ok(config) => Arc::new(config),
        Err(e) => {
            eprintln!("Failed to load configuration: {}", e);
            std::process::exit(1);
        }
    };
    
    // Initialize Python model manager
    let model_manager = match ModelManager::new(Arc::clone(&config)) {
        Ok(manager) => web::Data::new(Mutex::new(manager)),
        Err(e) => {
            eprintln!("Failed to initialize model manager: {}", e);
            std::process::exit(1);
        }
    };
    
    // Start HTTP server
    let server_config = config.server.clone();
    println!("Starting server at http://{}:{}", server_config.host, server_config.port);
    
    HttpServer::new(move || {
        App::new()
            // Enable logger middleware
            .wrap(middleware::Logger::default())
            // Register model manager as app data
            .app_data(model_manager.clone())
            // Configure API routes
            .configure(api::configure_routes)
    })
    .bind((server_config.host, server_config.port))?
    .run()
    .await
}