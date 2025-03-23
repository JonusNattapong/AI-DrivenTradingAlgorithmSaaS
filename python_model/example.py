#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for using the AI-Driven Trading Algorithm

This script demonstrates how to use the LSTM/GRU models to predict
stock or cryptocurrency prices.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from models.lstm_model import LSTMModel, GRUModel
from utils.data_loader import DataLoader
from training.trainer import ModelTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='AI-Driven Trading Algorithm Example')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock/crypto symbol (e.g., AAPL, BTC-USD)')
    parser.add_argument('--model-type', type=str, default='lstm', choices=['lstm', 'gru'], help='Model type')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date for historical data')
    parser.add_argument('--seq-length', type=int, default=60, help='Sequence length for prediction')
    parser.add_argument('--prediction-days', type=int, default=30, help='Number of days to predict')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension of the model')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM/GRU layers')
    parser.add_argument('--cache-dir', type=str, default='./data/cache', help='Directory to cache downloaded data')
    parser.add_argument('--save-dir', type=str, default='./data/models', help='Directory to save models')
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Create cache and save directories if they don't exist
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader(cache_dir=args.cache_dir)
    
    # Download data
    print(f"Downloading data for {args.symbol}...")
    data = data_loader.download_data(args.symbol, args.start_date)
    
    # Prepare data for training
    print("Preparing data...")
    X_train, y_train, X_test, y_test, scaler = data_loader.prepare_data(
        data, args.seq_length, 'Close', 
        feature_cols=['Open', 'High', 'Low', 'Close', 'Volume']
    )
    
    # Create model
    print(f"Creating {args.model_type.upper()} model...")
    input_dim = X_train.shape[2]  # Number of features
    output_dim = 1  # Predicting a single value (Close price)
    
    if args.model_type.lower() == 'lstm':
        model = LSTMModel(input_dim, args.hidden_dim, args.num_layers, output_dim)
    else:  # GRU
        model = GRUModel(input_dim, args.hidden_dim, args.num_layers, output_dim)
    
    # Train model
    print("Training model...")
    trainer = ModelTrainer(model)
    history = trainer.train(
        X_train, y_train, X_test, y_test,
        epochs=100, batch_size=32, learning_rate=0.001,
        early_stopping=True, patience=10,
        save_dir=args.save_dir, verbose=5
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test)
    print(f"Test metrics: {metrics}")
    
    # Plot training history
    trainer.plot_history(save_path=os.path.join(args.save_dir, f"{args.symbol}_{args.model_type}_history.png"))
    
    # Make predictions for future days
    print(f"Predicting next {args.prediction_days} days...")
    
    # Prepare input for prediction (last sequence from data)
    last_sequence = data[['Open', 'High', 'Low', 'Close', 'Volume']].values[-args.seq_length:]
    last_sequence = scaler.transform(last_sequence)
    last_sequence = np.expand_dims(last_sequence, axis=0)  # Add batch dimension
    
    # Predict next N days
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(args.prediction_days):
        # Predict next value
        pred = model.predict(current_sequence)[0][0]
        predictions.append(pred)
        
        # Update sequence for next prediction
        new_row = current_sequence[0, -1, :].copy()
        new_row[3] = pred  # Assuming Close price is at index 3
        
        # Remove first row and add new prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, :] = new_row
    
    # Convert predictions to original scale
    predictions = data_loader.inverse_transform(np.array(predictions), 3)  # 3 is the index of Close price
    
    # Plot predictions
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(data.index[-100:], data['Close'].values[-100:], label='Historical Data')
    
    # Plot predictions
    future_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),
        periods=args.prediction_days,
        freq='D'
    )
    plt.plot(future_dates, predictions, label='Predictions', color='red')
    
    plt.title(f"{args.symbol} Stock Price Prediction")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(args.save_dir, f"{args.symbol}_{args.model_type}_prediction.png"))
    plt.show()
    
    print("Done!")


if __name__ == '__main__':
    main()