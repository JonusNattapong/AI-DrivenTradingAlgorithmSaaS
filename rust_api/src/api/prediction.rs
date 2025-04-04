//! Prediction API module for the trading algorithm service

use actix_web::{web, HttpResponse, Responder};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::collections::HashMap;

use crate::python::model::{ModelManager, PredictionError};

#[derive(Debug, Deserialize)]
pub struct PredictionRequest {
    /// Symbol to predict (e.g., "AAPL", "BTC-USD")
    pub symbol: String,
    /// Number of days to predict into the future
    pub days: Option<u32>,
    /// Sequence length for the model input
    pub sequence_length: Option<u32>,
    /// Model type to use ("lstm" or "gru")
    pub model_type: Option<String>,
    /// Start date for historical data (format: YYYY-MM-DD)
    pub start_date: Option<String>,
    /// End date for historical data (format: YYYY-MM-DD)
    pub end_date: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct PredictionResponse {
    /// Symbol that was predicted
    pub symbol: String,
    /// Timestamp of the prediction
    pub timestamp: DateTime<Utc>,
    /// Predicted values
    pub predictions: Vec<PredictionPoint>,
    /// Model information
    pub model_info: ModelInfo,
}

#[derive(Debug, Serialize)]
pub struct PredictionPoint {
    /// Date of the prediction
    pub date: String,
    /// Predicted value
    pub value: f64,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    /// Type of model used
    pub model_type: String,
    /// Sequence length used
    pub sequence_length: u32,
    /// Training accuracy metrics
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Status of the API
    pub status: String,
    /// Version of the API
    pub version: String,
    /// Timestamp of the health check
    pub timestamp: DateTime<Utc>,
}

/// Prediction endpoint
pub async fn predict(
    request: web::Json<PredictionRequest>,
    model_manager: web::Data<Mutex<ModelManager>>,
) -> impl Responder {
    let req = request.into_inner();
    
    // Set default values
    let days = req.days.unwrap_or(5);
    let sequence_length = req.sequence_length.unwrap_or(60);
    let model_type = req.model_type.unwrap_or_else(|| "lstm".to_string());
    
    // Get model manager
    let mut model_manager = match model_manager.lock() {
        Ok(manager) => manager,
        Err(_) => return HttpResponse::InternalServerError().json("Failed to acquire model manager lock"),
    };
    
    // Make prediction
    match model_manager.predict(
        &req.symbol,
        days,
        sequence_length,
        &model_type,
        req.start_date.as_deref(),
        req.end_date.as_deref(),
    ) {
        Ok((predictions, metrics)) => {
            // Convert predictions to response format
            let prediction_points = predictions
                .iter()
                .enumerate()
                .map(|(i, &value)| {
                    let date = chrono::Utc::now() + chrono::Duration::days(i as i64 + 1);
                    PredictionPoint {
                        date: date.format("%Y-%m-%d").to_string(),
                        value,
                    }
                })
                .collect();
            
            // Create response
            let response = PredictionResponse {
                symbol: req.symbol,
                timestamp: Utc::now(),
                predictions: prediction_points,
                model_info: ModelInfo {
                    model_type,
                    sequence_length,
                    metrics,
                },
            };
            
            HttpResponse::Ok().json(response)
        }
        Err(err) => {
            match err {
                PredictionError::ModelError => HttpResponse::InternalServerError().json("Model error"),
                PredictionError::Other(msg) => {
                    // Log the specific error message for debugging
                    log::error!("Prediction failed: {}", msg);
                    // Return a generic error message to the client
                    HttpResponse::InternalServerError().json(format!("Prediction failed: {}", msg))
                }
            }
        }
    }
}

/// Health check endpoint
pub async fn health_check() -> impl Responder {
    let response = HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: Utc::now(),
    };
    
    HttpResponse::Ok().json(response)
}
