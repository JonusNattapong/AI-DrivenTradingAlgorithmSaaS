import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    """
    LSTM model for time series prediction of stock/crypto prices
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        """
        Initialize the LSTM model
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden units
            num_layers (int): Number of LSTM layers
            output_dim (int): Number of output features
            dropout (float): Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_dim)
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Apply fully connected layer
        out = self.fc(out)
        
        return out
    
    def predict(self, x):
        """
        Make prediction using the model
        
        Args:
            x (numpy.ndarray): Input data of shape (batch_size, seq_len, input_dim)
            
        Returns:
            numpy.ndarray: Predicted values
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
            output = self.forward(x)
            return output.cpu().numpy()


class GRUModel(nn.Module):
    """
    GRU model for time series prediction of stock/crypto prices
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_dim)
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Apply fully connected layer
        out = self.fc(out)
        
        return out
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
            output = self.forward(x)
            return output.cpu().numpy()