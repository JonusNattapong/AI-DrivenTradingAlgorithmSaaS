import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
import datetime

class DataLoader:
    """
    Class for loading and preprocessing financial data for the LSTM model
    """
    def __init__(self, cache_dir=None):
        """
        Initialize the DataLoader
        
        Args:
            cache_dir (str, optional): Directory to cache downloaded data
        """
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.cache_dir = cache_dir
        
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def download_data(self, symbol, start_date, end_date=None, interval='1d'):
        """
        Download financial data from Yahoo Finance
        
        Args:
            symbol (str): Stock/crypto symbol
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str, optional): End date in format 'YYYY-MM-DD'
            interval (str, optional): Data interval ('1d', '1h', etc.)
            
        Returns:
            pandas.DataFrame: DataFrame containing the downloaded data
        """
        if end_date is None:
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            
        # Check if data is cached
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{symbol}_{start_date}_{end_date}_{interval}.csv")
            if os.path.exists(cache_file):
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Download data
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        
        # Cache data if cache_dir is provided
        if self.cache_dir:
            data.to_csv(cache_file)
            
        return data
    
    def prepare_data(self, data, seq_length, target_col='Close', feature_cols=None, train_split=0.8):
        """
        Prepare data for LSTM model
        
        Args:
            data (pandas.DataFrame): DataFrame containing the financial data
            seq_length (int): Sequence length for LSTM input
            target_col (str, optional): Target column to predict
            feature_cols (list, optional): List of feature columns to use
            train_split (float, optional): Train/test split ratio
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, scaler)
        """
        if feature_cols is None:
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
        # Select features
        data_filtered = data[feature_cols].copy()
        
        # Handle missing values
        data_filtered.fillna(method='ffill', inplace=True)
        data_filtered.fillna(method='bfill', inplace=True)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data_filtered)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data, seq_length, feature_cols.index(target_col) if target_col in feature_cols else 3)
        
        # Split data
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, y_train, X_test, y_test, self.scaler
    
    def _create_sequences(self, data, seq_length, target_idx):
        """
        Create sequences for LSTM input
        
        Args:
            data (numpy.ndarray): Scaled data
            seq_length (int): Sequence length
            target_idx (int): Index of target column
            
        Returns:
            tuple: (X, y) where X is the input sequences and y is the target values
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, target_idx])
            
        return np.array(X), np.array(y)
    
    def inverse_transform(self, scaled_value, feature_idx):
        """
        Inverse transform scaled value to original scale
        
        Args:
            scaled_value (float or numpy.ndarray): Scaled value(s)
            feature_idx (int): Index of the feature
            
        Returns:
            float or numpy.ndarray: Original scale value(s)
        """
        # Create a dummy array with zeros
        dummy = np.zeros((len(scaled_value) if hasattr(scaled_value, '__len__') else 1, self.scaler.scale_.shape[0]))
        
        # Set the scaled value at the feature index
        if hasattr(scaled_value, '__len__'):
            dummy[:, feature_idx] = scaled_value
        else:
            dummy[0, feature_idx] = scaled_value
            
        # Inverse transform
        inverse_scaled = self.scaler.inverse_transform(dummy)
        
        # Return the value at the feature index
        if hasattr(scaled_value, '__len__'):
            return inverse_scaled[:, feature_idx]
        else:
            return inverse_scaled[0, feature_idx]