"""
Training Script for Inverse Kinematics Neural Network

This script implements the exact hyperparameters from Table 4:
- Learning rate: 0.001
- Epochs: 2000
- Optimizer: Adam
- Loss function: MSE
"""

import numpy as np
import pandas as pd
from inverse_kinematics_nn import InverseKinematicsNN, generate_sample_data
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime


def load_data_from_csv(data_path):
    """
    Load training data from CSV files
    
    Expected format:
    - input_data.csv: columns [Px, Py, Pz, Rx, Ry, Rz]
    - output_data.csv: columns [theta1, theta2, theta3, theta4, theta5]
    """
    try:
        X = pd.read_csv(os.path.join(data_path, 'input_data.csv')).values
        y = pd.read_csv(os.path.join(data_path, 'output_data.csv')).values
        print(f"Loaded data from CSV: X shape {X.shape}, y shape {y.shape}")
        return X, y
    except FileNotFoundError:
        print("CSV files not found. Using generated sample data.")
        return None, None


def train_inverse_kinematics_model(data_path=None, n_samples=50000, epochs=2000, 
                                 save_model=True, plot_results=True):
    """
    Train the inverse kinematics neural network with specified hyperparameters
    
    Args:
        data_path: Path to directory containing input_data.csv and output_data.csv
        n_samples: Number of samples to generate if no data_path provided
        epochs: Number of training epochs (Table 4: 2000)
        save_model: Whether to save the trained model
        plot_results: Whether to plot training results
    """
    
    print("=" * 60)
    print("NEURAL NETWORK-BASED INVERSE KINEMATICS TRAINING")
    print("=" * 60)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize the neural network
    ik_nn = InverseKinematicsNN()
    
    # Build the model with exact architecture from the paper
    print("\nBuilding neural network architecture...")
    model = ik_nn.build_model()
    
    print("Model Architecture Summary:")
    print("-" * 40)
    ik_nn.get_model_summary()
    
    # Load or generate data
    if data_path:
        X, y = load_data_from_csv(data_path)
        if X is None:
            X, y = generate_sample_data(n_samples)
    else:
        print(f"\nGenerating {n_samples} sample data points...")
        X, y = generate_sample_data(n_samples)
    
    print(f"\nDataset Information:")
    print(f"Input features (Px, Py, Pz, Rx, Ry, Rz): {X.shape}")
    print(f"Output targets (θ1, θ2, θ3, θ4, θ5): {y.shape}")
    print(f"Input range - Min: {X.min():.3f}, Max: {X.max():.3f}")
    print(f"Output range - Min: {y.min():.3f}, Max: {y.max():.3f}")
    
    # Prepare data for training
    print("\nPreparing data for training...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = ik_nn.prepare_data(
        X, y, test_size=0.15, validation_size=0.15
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Training with hyperparameters from Table 4
    print("\n" + "=" * 40)
    print("TRAINING CONFIGURATION (Table 4)")
    print("=" * 40)
    print(f"Learning rate: 0.001")
    print(f"Epochs: {epochs}")
    print(f"Optimizer: Adam")
    print(f"Loss function: MSE")
    print(f"Batch size: 32")
    print("=" * 40)
    
    print(f"\nStarting training for {epochs} epochs...")
    history = ik_nn.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=32,
        verbose=1,
        save_best=True
    )
    
    # Evaluate the model
    print("\n" + "=" * 40)
    print("MODEL EVALUATION")
    print("=" * 40)
    
    results, predictions = ik_nn.evaluate(X_test, y_test)
    
    # Calculate detailed statistics
    errors = np.abs(y_test - predictions)
    print(f"\nDetailed Error Analysis:")
    print(f"Mean Absolute Error per joint:")
    for i in range(5):
        print(f"  θ{i+1}: {np.mean(errors[:, i]):.6f} rad ({np.degrees(np.mean(errors[:, i])):.3f}°)")
    
    print(f"\nMax Absolute Error per joint:")
    for i in range(5):
        print(f"  θ{i+1}: {np.max(errors[:, i]):.6f} rad ({np.degrees(np.max(errors[:, i])):.3f}°)")
    
    # Save training results
    results_dir = "training_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
    
    # Save test predictions
    predictions_df = pd.DataFrame(predictions, columns=['theta1_pred', 'theta2_pred', 'theta3_pred', 'theta4_pred', 'theta5_pred'])
    actual_df = pd.DataFrame(y_test, columns=['theta1_actual', 'theta2_actual', 'theta3_actual', 'theta4_actual', 'theta5_actual'])
    results_df = pd.concat([actual_df, predictions_df], axis=1)
    results_df.to_csv(os.path.join(results_dir, 'test_predictions.csv'), index=False)
    
    if plot_results:
        # Plot training history
        print("\nPlotting training history...")
        ik_nn.plot_training_history()
        plt.savefig(os.path.join(results_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        
        # Plot prediction accuracy
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(5):
            axes[i].scatter(y_test[:, i], predictions[:, i], alpha=0.5, s=1)
            axes[i].plot([y_test[:, i].min(), y_test[:, i].max()], 
                        [y_test[:, i].min(), y_test[:, i].max()], 'r--', lw=2)
            axes[i].set_xlabel(f'Actual θ{i+1} (rad)')
            axes[i].set_ylabel(f'Predicted θ{i+1} (rad)')
            axes[i].set_title(f'Joint {i+1} Prediction Accuracy')
            axes[i].grid(True, alpha=0.3)
        
        # Remove the empty subplot
        axes[5].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'prediction_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    if save_model:
        # Save the final model
        model_path = os.path.join(results_dir, 'inverse_kinematics_final_model.h5')
        ik_nn.save_model(model_path)
        
        # Save scalers
        import joblib
        joblib.dump(ik_nn.scaler_input, os.path.join(results_dir, 'scaler_input.pkl'))
        joblib.dump(ik_nn.scaler_output, os.path.join(results_dir, 'scaler_output.pkl'))
    
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {results_dir}/")
    
    return ik_nn, history, results


def main():
    """Main training function with command line arguments"""
    parser = argparse.ArgumentParser(description='Train Inverse Kinematics Neural Network')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to directory containing input_data.csv and output_data.csv')
    parser.add_argument('--n_samples', type=int, default=50000,
                       help='Number of samples to generate if no data provided')
    parser.add_argument('--epochs', type=int, default=2000,
                       help='Number of training epochs (default: 2000 from Table 4)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save the trained model')
    parser.add_argument('--no_plot', action='store_true',
                       help='Do not plot training results')
    
    args = parser.parse_args()
    
    # Train the model
    ik_nn, history, results = train_inverse_kinematics_model(
        data_path=args.data_path,
        n_samples=args.n_samples,
        epochs=args.epochs,
        save_model=not args.no_save,
        plot_results=not args.no_plot
    )
    
    return ik_nn


if __name__ == "__main__":
    main()