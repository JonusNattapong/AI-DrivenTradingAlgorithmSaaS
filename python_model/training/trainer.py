import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class ModelTrainer:
    """
    Class for training and evaluating LSTM/GRU models
    """
    def __init__(self, model, device=None):
        """
        Initialize the trainer
        
        Args:
            model (torch.nn.Module): PyTorch model to train
            device (str, optional): Device to use for training ('cuda' or 'cpu')
        """
        self.model = model
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': {},
            'val_metrics': {}
        }
        
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
              learning_rate=0.001, criterion=None, optimizer=None, scheduler=None,
              early_stopping=True, patience=10, min_delta=0.001,
              save_dir=None, save_best=True, verbose=1):
        """
        Train the model
        
        Args:
            X_train (numpy.ndarray): Training input data
            y_train (numpy.ndarray): Training target data
            X_val (numpy.ndarray): Validation input data
            y_val (numpy.ndarray): Validation target data
            epochs (int, optional): Number of epochs to train
            batch_size (int, optional): Batch size
            learning_rate (float, optional): Learning rate
            criterion (torch.nn.Module, optional): Loss function
            optimizer (torch.optim.Optimizer, optional): Optimizer
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
            early_stopping (bool, optional): Whether to use early stopping
            patience (int, optional): Patience for early stopping
            min_delta (float, optional): Minimum change in validation loss to be considered as improvement
            save_dir (str, optional): Directory to save model checkpoints
            save_best (bool, optional): Whether to save the best model
            verbose (int, optional): Verbosity level
            
        Returns:
            dict: Training history
        """
        # Convert data to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Set criterion if not provided
        if criterion is None:
            criterion = nn.MSELoss()
            
        # Set optimizer if not provided
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        counter = 0
        
        # Create save directory if not exists
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
            train_loss /= len(train_loader.dataset)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    val_loss += loss.item() * inputs.size(0)
                    
            val_loss /= len(val_loader.dataset)
            self.history['val_loss'].append(val_loss)
            
            # Step scheduler if provided
            if scheduler is not None:
                scheduler.step(val_loss)
                
            # Print progress
            if verbose > 0 and (epoch + 1) % verbose == 0:
                print(f'Epoch {epoch+1}/{epochs} - '
                      f'Loss: {train_loss:.4f} - '
                      f'Val Loss: {val_loss:.4f}')
                
            # Check if current model is the best
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                counter = 0
                
                # Save best model
                if save_dir and save_best:
                    self._save_model(save_dir, 'best_model.pth')
            else:
                counter += 1
                
            # Early stopping
            if early_stopping and counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
                
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        # Save final model
        if save_dir:
            self._save_model(save_dir, 'final_model.pth')
            self._save_history(save_dir)
            
        return self.history
    
    def evaluate(self, X_test, y_test, criterion=None, batch_size=32):
        """
        Evaluate the model on test data
        
        Args:
            X_test (numpy.ndarray): Test input data
            y_test (numpy.ndarray): Test target data
            criterion (torch.nn.Module, optional): Loss function
            batch_size (int, optional): Batch size
            
        Returns:
            dict: Evaluation metrics
        """
        # Convert data to PyTorch tensors
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        # Set criterion if not provided
        if criterion is None:
            criterion = nn.MSELoss()
            
        # Create data loader
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        
        # Evaluate
        self.model.eval()
        test_loss = 0.0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                test_loss += loss.item() * inputs.size(0)
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
                
        test_loss /= len(test_loader.dataset)
        
        # Calculate metrics
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals)
        
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        
        metrics = {
            'loss': test_loss,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions using the model
        
        Args:
            X (numpy.ndarray): Input data
            
        Returns:
            numpy.ndarray: Predicted values
        """
        return self.model.predict(X)
    
    def plot_history(self, figsize=(12, 6), save_path=None):
        """
        Plot training history
        
        Args:
            figsize (tuple, optional): Figure size
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def _save_model(self, save_dir, filename):
        """
        Save model to file
        
        Args:
            save_dir (str): Directory to save model
            filename (str): Filename
        """
        model_path = os.path.join(save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'model_config': {
                'input_dim': self.model.lstm.input_size if hasattr(self.model, 'lstm') else self.model.gru.input_size,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'output_dim': self.model.fc.out_features
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, model_path)
    
    def _save_history(self, save_dir):
        """
        Save training history to file
        
        Args:
            save_dir (str): Directory to save history
        """
        history_path = os.path.join(save_dir, 'training_history.json')
        
        # Convert numpy arrays to lists for JSON serialization
        history_json = {}
        for key, value in self.history.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                history_json[key] = [v.tolist() for v in value]
            elif isinstance(value, dict):
                history_json[key] = {}
                for k, v in value.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                        history_json[key][k] = [val.tolist() for val in v]
                    else:
                        history_json[key][k] = v
            else:
                history_json[key] = value
                
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=4)
    
    @classmethod
    def load_model(cls, model_path, model_class, device=None):
        """
        Load model from file
        
        Args:
            model_path (str): Path to model file
            model_class (class): Model class
            device (str, optional): Device to load model to
            
        Returns:
            ModelTrainer: Trainer with loaded model
        """
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Create model instance
        model_config = checkpoint['model_config']
        model = model_class(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            output_dim=model_config['output_dim']
        )
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create trainer
        trainer = cls(model, device)
        
        return trainer