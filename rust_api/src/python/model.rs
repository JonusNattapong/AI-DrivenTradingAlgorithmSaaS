//! Python model interface for the trading algorithm service

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

use crate::config::Config;

#[derive(Error, Debug)]
pub enum PredictionError {
    #[error("Invalid symbol")]
    InvalidSymbol,
    
    #[error("Failed to fetch data")]
    DataFetchError,
    
    #[error("Model error")]
    ModelError,
    
    #[error("Invalid model type")]
    InvalidModelType,
    
    #[error("{0}")]
    Other(String),
}

pub struct ModelManager {
    config: Arc<Config>,
    py_runtime: PyObject,
}

impl ModelManager {
    pub fn new(config: Arc<Config>) -> Result<Self, PredictionError> {
        // Initialize Python interpreter
        pyo3::prepare_freethreaded_python();
        
        let result = Python::with_gil(|py| -> Result<PyObject, PredictionError> {
            // Import required Python modules
            let sys = PyModule::import(py, "sys")?;
            let os = PyModule::import(py, "os")?;
            
            // Add model path to Python path
            let model_path = Path::new(&config.python.model_path).canonicalize()
                .map_err(|e| PredictionError::Other(format!("Failed to canonicalize model path: {}", e)))?;
            
            let py_path = sys.getattr("path")?
                .downcast::<PyList>()
                .map_err(|_| PredictionError::Other("Failed to get Python path".to_string()))?;
            
            py_path.append(model_path.to_str().unwrap())?;
            
            // Create runtime module
            let runtime_code = r#"
# Runtime module for model management
import os
import sys
import torch
import numpy as np
import json
from datetime import datetime, timedelta

# Import model classes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_model.models.lstm_model import LSTMModel, GRUModel
from python_model.utils.data_loader import DataLoader
from python_model.training.trainer import ModelTrainer

class ModelRuntime:
    def __init__(self, data_cache_path):
        self.data_loader = DataLoader(cache_dir=data_cache_path)
        self.models = {}
        self.metrics = {}
        
    def predict(self, symbol, days, seq_length, model_type, start_date=None, end_date=None):
        """Make prediction for a symbol"""
        # Set default dates if not provided
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Download data
        try:
            data = self.data_loader.download_data(symbol, start_date, end_date)
            if data.empty or len(data) < seq_length + days:
                return None, {"error": "Insufficient data for prediction"}
        except Exception as e:
            return None, {"error": f"Failed to download data: {str(e)}"}
            
        # Prepare data
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        X_train, y_train, X_test, y_test, scaler = self.data_loader.prepare_data(
            data, seq_length, 'Close', feature_cols
        )
        
        # Get or create model
        model_key = f"{symbol}_{model_type}_{seq_length}"
        if model_key not in self.models:
            # Create model
            input_dim = X_train.shape[2]
            hidden_dim = 64
            num_layers = 2
            output_dim = 1
            
            if model_type.lower() == 'lstm':
                model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
            elif model_type.lower() == 'gru':
                model = GRUModel(input_dim, hidden_dim, num_layers, output_dim)
            else:
                return None, {"error": "Invalid model type"}
                
            # Train model
            trainer = ModelTrainer(model)
            trainer.train(
                X_train, y_train, X_test, y_test,
                epochs=50, batch_size=32, learning_rate=0.001,
                early_stopping=True, patience=5
            )
            
            # Evaluate model
            metrics = trainer.evaluate(X_test, y_test)
            
            # Store model and metrics
            self.models[model_key] = model
            self.metrics[model_key] = metrics
        
        # Make prediction
        model = self.models[model_key]
        metrics = self.metrics[model_key]
        
        # Prepare input for prediction
        last_sequence = data[feature_cols].values[-seq_length:]
        last_sequence = scaler.transform(last_sequence)
        last_sequence = np.expand_dims(last_sequence, axis=0)
        
        # Predict next 'days' values
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Predict next value
            pred = model.predict(current_sequence)[0][0]
            predictions.append(pred)
            
            # Update sequence for next prediction
            new_row = current_sequence[0, -1, :].copy()
            new_row[feature_cols.index('Close')] = pred
            
            # Remove first row and add new prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = new_row
        
        # Convert predictions to original scale
        predictions = self.data_loader.inverse_transform(
            np.array(predictions), feature_cols.index('Close')
        )
        
        # Convert metrics to serializable format
        metrics_dict = {}
        for k, v in metrics.items():
            if isinstance(v, (np.float32, np.float64)):
                metrics_dict[k] = float(v)
            else:
                metrics_dict[k] = v
                
        return predictions.tolist(), metrics_dict
"#;
            
            // Create runtime module
            let runtime = PyModule::from_code(py, runtime_code, "runtime.py", "runtime")?;
            
            // Create runtime instance
            let runtime_instance = runtime.getattr("ModelRuntime")?.call1((config.python.data_cache_path.clone(),))?;
            
            Ok(runtime_instance.into())
        });
        
        match result {
            Ok(py_runtime) => Ok(ModelManager {
                config,
                py_runtime,
            }),
            Err(e) => Err(PredictionError::Other(format!("Failed to initialize Python runtime: {:?}", e))),
        }
    }
    
    pub fn predict(
        &mut self,
        symbol: &str,
        days: u32,
        seq_length: u32,
        model_type: &str,
        start_date: Option<&str>,
        end_date: Option<&str>,
    ) -> Result<(Vec<f64>, HashMap<String, f64>), PredictionError> {
        Python::with_gil(|py| {
            // Call predict method
            let result = self.py_runtime.call_method(
                py,
                "predict",
                (symbol, days, seq_length, model_type, start_date, end_date),
                None,
            );
            
            match result {
                Ok(result) => {
                    // Extract predictions and metrics
                    let (predictions, metrics) = result.extract::<(Option<Vec<f64>>, HashMap<String, String>)>(py)
                        .map_err(|e| PredictionError::Other(format!("Failed to extract prediction result: {:?}", e)))?;
                    
                    // Check for errors
                    if let Some(error) = metrics.get("error") {
                        return Err(PredictionError::Other(error.clone()));
                    }
                    
                    // Convert metrics to f64
                    let metrics_f64 = metrics.iter()
                        .filter_map(|(k, v)| {
                            v.parse::<f64>().ok().map(|val| (k.clone(), val))
                        })
                        .collect::<HashMap<String, f64>>();
                    
                    // Return predictions and metrics
                    match predictions {
                        Some(preds) => Ok((preds, metrics_f64)),
                        None => Err(PredictionError::ModelError),
                    }
                },
                Err(e) => Err(PredictionError::Other(format!("Failed to call predict method: {:?}", e))),
            }
        })
    }
}

impl From<PyErr> for PredictionError {
    fn from(err: PyErr) -> Self {
        PredictionError::Other(format!("Python error: {:?}", err))
    }
}