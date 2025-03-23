# AI-Driven Trading Algorithm SaaS

A powerful platform that combines Python with PyTorch for stock/crypto market prediction and Rust for a fast, secure API for investors.

## Project Overview

This project implements an AI-driven trading algorithm as a Software-as-a-Service (SaaS) platform. It uses deep learning models (LSTM and GRU) to predict stock and cryptocurrency prices and provides a fast, secure API for investors to access these predictions.

## Architecture

The project consists of two main components:

1. **Python Model (PyTorch)**: Implements LSTM and GRU models for time series prediction of financial data.
2. **Rust API**: Provides a fast and secure interface for accessing the prediction models.

## Features

- **Deep Learning Models**: LSTM and GRU models for accurate time series prediction
- **Data Preprocessing**: Automatic downloading and preprocessing of financial data
- **Model Training**: Efficient training with early stopping and model checkpointing
- **RESTful API**: Fast and secure API for accessing predictions
- **Caching**: Efficient caching of downloaded data and trained models

## Getting Started

### Prerequisites

- Python 3.8+
- Rust 1.50+
- PyTorch 1.10+

### Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/AI-DrivenTradingAlgorithmSaaS.git
cd AI-DrivenTradingAlgorithmSaaS
```

2. Install Python dependencies

```bash
cd python_model
pip install -r requirements.txt
```

3. Build the Rust API

```bash
cd ../rust_api
cargo build --release
```

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
SERVER_HOST=127.0.0.1
SERVER_PORT=8080
PYTHON_MODEL_PATH=./python_model
PYTHON_DATA_CACHE_PATH=./python_model/data/cache
JWT_SECRET=your_jwt_secret_here
TOKEN_EXPIRATION=86400
```

## Usage

### Running the Example Script

```bash
cd python_model
python example.py --symbol AAPL --model-type lstm --prediction-days 30
```

### Starting the API Server

```bash
cd rust_api
cargo run --release
```

### Making Predictions via API

```bash
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL", "days":5, "model_type":"lstm"}'
```

## API Endpoints

### POST /api/v1/predict

Make a prediction for a stock or cryptocurrency.

**Request Body:**

```json
{
  "symbol": "AAPL",
  "days": 5,
  "sequence_length": 60,
  "model_type": "lstm",
  "start_date": "2020-01-01",
  "end_date": "2023-01-01"
}
```

**Response:**

```json
{
  "symbol": "AAPL",
  "timestamp": "2023-06-01T12:34:56Z",
  "predictions": [
    {"date": "2023-06-02", "value": 150.25},
    {"date": "2023-06-03", "value": 151.30},
    {"date": "2023-06-04", "value": 149.80},
    {"date": "2023-06-05", "value": 152.10},
    {"date": "2023-06-06", "value": 153.45}
  ],
  "model_info": {
    "model_type": "lstm",
    "sequence_length": 60,
    "metrics": {
      "mse": 0.0025,
      "rmse": 0.05,
      "mae": 0.04
    }
  }
}
```

### GET /api/v1/health

Check the health of the API.

**Response:**

```json
{
  "status": "ok",
  "version": "0.1.0",
  "timestamp": "2023-06-01T12:34:56Z"
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.