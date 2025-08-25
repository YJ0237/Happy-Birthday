"""
Neural Network-Based Inverse Kinematics Implementation

This implementation reproduces the neural network architecture described in the paper:
- Input layer: 6 neurons (Px, Py, Pz, Rx, Ry, Rz)
- Hidden layers: 48 -> 120 -> 60 neurons
- Output layer: 5 neurons (θ1, θ2, θ3, θ4, θ5)
- Activation: Sigmoid for hidden layers, Linear for output
- Dropout: 0.05 in last two hidden layers
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


class InverseKinematicsNN:
    """Neural Network for Inverse Kinematics"""
    
    def __init__(self):
        self.model = None
        self.scaler_input = StandardScaler()
        self.scaler_output = StandardScaler()
        self.history = None
        
    def build_model(self):
        """Build the neural network architecture as specified"""
        model = models.Sequential([
            # Input layer (6 neurons for Px, Py, Pz, Rx, Ry, Rz)
            layers.Input(shape=(6,)),
            
            # First hidden layer (48 neurons)
            layers.Dense(48, activation='sigmoid', name='hidden_layer_1'),
            
            # Second hidden layer (120 neurons) with dropout
            layers.Dense(120, activation='sigmoid', name='hidden_layer_2'),
            layers.Dropout(0.05, name='dropout_1'),
            
            # Third hidden layer (60 neurons) with dropout
            layers.Dense(60, activation='sigmoid', name='hidden_layer_3'),
            layers.Dropout(0.05, name='dropout_2'),
            
            # Output layer (5 neurons for θ1, θ2, θ3, θ4, θ5)
            layers.Dense(5, activation='linear', name='output_layer')
        ])
        
        # Compile with hyperparameters from Table 4
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Mean Squared Error
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def get_model_summary(self):
        """Display model architecture"""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def prepare_data(self, X, y, test_size=0.2, validation_size=0.2, normalize=True):
        """
        Prepare data for training
        
        Args:
            X: Input data (N x 6) - [Px, Py, Pz, Rx, Ry, Rz]
            y: Output data (N x 5) - [θ1, θ2, θ3, θ4, θ5]
            test_size: Fraction of data for testing
            validation_size: Fraction of training data for validation
            normalize: Whether to normalize the data
        """
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=42
        )
        
        if normalize:
            # Normalize input features
            X_train = self.scaler_input.fit_transform(X_train)
            X_val = self.scaler_input.transform(X_val)
            X_test = self.scaler_input.transform(X_test)
            
            # Normalize output targets
            y_train = self.scaler_output.fit_transform(y_train)
            y_val = self.scaler_output.transform(y_val)
            y_test = self.scaler_output.transform(y_test)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=2000, 
              batch_size=32, verbose=1, save_best=True):
        """
        Train the neural network
        
        Args:
            X_train: Training input data
            y_train: Training output data
            X_val: Validation input data
            y_val: Validation output data
            epochs: Number of training epochs (default from Table 4: 2000)
            batch_size: Batch size for training
            verbose: Verbosity level
            save_best: Whether to save the best model
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = []
        
        if save_best and X_val is not None:
            callbacks.append(
                ModelCheckpoint(
                    'best_inverse_kinematics_model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=1
                )
            )
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=100,
                    restore_best_weights=True,
                    verbose=1
                )
            )
        
        # Training
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X, denormalize=True):
        """
        Make predictions
        
        Args:
            X: Input data to predict
            denormalize: Whether to denormalize the predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Normalize input if scalers are fitted
        if hasattr(self.scaler_input, 'mean_'):
            X_scaled = self.scaler_input.transform(X)
        else:
            X_scaled = X
        
        # Predict
        y_pred = self.model.predict(X_scaled)
        
        # Denormalize output if requested
        if denormalize and hasattr(self.scaler_output, 'mean_'):
            y_pred = self.scaler_output.inverse_transform(y_pred)
        
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Normalize test data if scalers are fitted
        if hasattr(self.scaler_input, 'mean_'):
            X_test_scaled = self.scaler_input.transform(X_test)
        else:
            X_test_scaled = X_test
            
        if hasattr(self.scaler_output, 'mean_'):
            y_test_scaled = self.scaler_output.transform(y_test)
        else:
            y_test_scaled = y_test
        
        # Evaluate
        results = self.model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
        
        # Get predictions for additional metrics
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.scaler_output.inverse_transform(y_pred_scaled) if hasattr(self.scaler_output, 'mean_') else y_pred_scaled
        
        # Calculate additional metrics
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        
        print(f"Test Results:")
        print(f"Loss: {results[0]:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        
        return results, y_pred
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath='inverse_kinematics_model.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save!")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='inverse_kinematics_model.h5'):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def generate_sample_data(n_samples=10000):
    """
    Generate sample data for demonstration
    In practice, this would come from your robot's forward kinematics
    
    Returns:
        X: Position and orientation data (n_samples x 6)
        y: Joint angles (n_samples x 5)
    """
    # Generate random joint angles within typical robot joint limits
    joint_angles = np.random.uniform(-np.pi, np.pi, (n_samples, 5))
    
    # Simulate forward kinematics (simplified example)
    # In practice, you would use your actual robot's forward kinematics
    positions = np.zeros((n_samples, 6))
    
    for i in range(n_samples):
        theta = joint_angles[i]
        # Simplified forward kinematics (replace with your actual equations)
        positions[i, 0] = np.sum(np.cos(theta))  # Px
        positions[i, 1] = np.sum(np.sin(theta))  # Py
        positions[i, 2] = np.sum(theta) * 0.1    # Pz
        positions[i, 3] = theta[0] + theta[1]    # Rx
        positions[i, 4] = theta[2] + theta[3]    # Ry
        positions[i, 5] = theta[4]               # Rz
    
    return positions, joint_angles


def main():
    """Main function to demonstrate the inverse kinematics neural network"""
    print("Neural Network-Based Inverse Kinematics")
    print("=" * 50)
    
    # Create the neural network
    ik_nn = InverseKinematicsNN()
    
    # Build and display model architecture
    model = ik_nn.build_model()
    print("\nModel Architecture:")
    ik_nn.get_model_summary()
    
    # Generate sample data (replace with your actual data)
    print("\nGenerating sample data...")
    X, y = generate_sample_data(n_samples=10000)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    
    # Prepare data
    print("\nPreparing data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = ik_nn.prepare_data(X, y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train the model
    print("\nTraining the model...")
    history = ik_nn.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,  # Reduced for demo, use 2000 for full training
        batch_size=32,
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating the model...")
    results, predictions = ik_nn.evaluate(X_test, y_test)
    
    # Plot training history
    ik_nn.plot_training_history()
    
    # Save the model
    ik_nn.save_model()
    
    print("\nTraining completed successfully!")
    print("You can now use the trained model for inverse kinematics predictions.")


if __name__ == "__main__":
    main()