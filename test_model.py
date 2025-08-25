"""
Test and Evaluation Script for Inverse Kinematics Neural Network

This script provides comprehensive testing and evaluation capabilities
for the trained inverse kinematics neural network.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from inverse_kinematics_nn import InverseKinematicsNN
import joblib
import os
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class InverseKinematicsEvaluator:
    """Comprehensive evaluation class for inverse kinematics neural network"""
    
    def __init__(self, model_path=None, scaler_input_path=None, scaler_output_path=None):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to the trained model
            scaler_input_path: Path to input scaler
            scaler_output_path: Path to output scaler
        """
        self.ik_nn = InverseKinematicsNN()
        
        if model_path and os.path.exists(model_path):
            self.ik_nn.load_model(model_path)
            
        if scaler_input_path and os.path.exists(scaler_input_path):
            self.ik_nn.scaler_input = joblib.load(scaler_input_path)
            
        if scaler_output_path and os.path.exists(scaler_output_path):
            self.ik_nn.scaler_output = joblib.load(scaler_output_path)
    
    def evaluate_accuracy(self, X_test, y_test, verbose=True):
        """
        Comprehensive accuracy evaluation
        
        Args:
            X_test: Test input data
            y_test: Test output data
            verbose: Whether to print detailed results
        """
        # Get predictions
        y_pred = self.ik_nn.predict(X_test)
        
        # Calculate various metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Per-joint metrics
        joint_mae = [mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(5)]
        joint_rmse = [np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])) for i in range(5)]
        joint_r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(5)]
        
        # Angular errors in degrees
        joint_mae_deg = [np.degrees(mae) for mae in joint_mae]
        joint_rmse_deg = [np.degrees(rmse) for rmse in joint_rmse]
        
        if verbose:
            print("=" * 60)
            print("COMPREHENSIVE ACCURACY EVALUATION")
            print("=" * 60)
            print(f"Overall Metrics:")
            print(f"  Mean Absolute Error (MAE): {mae:.6f} rad ({np.degrees(mae):.3f}°)")
            print(f"  Root Mean Square Error (RMSE): {rmse:.6f} rad ({np.degrees(rmse):.3f}°)")
            print(f"  R-squared Score: {r2:.6f}")
            print(f"  Number of test samples: {len(y_test)}")
            
            print(f"\nPer-Joint Analysis:")
            print("-" * 40)
            for i in range(5):
                print(f"Joint θ{i+1}:")
                print(f"  MAE: {joint_mae[i]:.6f} rad ({joint_mae_deg[i]:.3f}°)")
                print(f"  RMSE: {joint_rmse[i]:.6f} rad ({joint_rmse_deg[i]:.3f}°)")
                print(f"  R²: {joint_r2[i]:.6f}")
                print()
        
        results = {
            'overall': {'mae': mae, 'rmse': rmse, 'r2': r2},
            'per_joint': {
                'mae': joint_mae, 'rmse': joint_rmse, 'r2': joint_r2,
                'mae_deg': joint_mae_deg, 'rmse_deg': joint_rmse_deg
            },
            'predictions': y_pred
        }
        
        return results
    
    def test_convergence_speed(self, X_test, y_test, tolerance_degrees=1.0):
        """
        Test how many predictions fall within a given tolerance
        
        Args:
            X_test: Test input data
            y_test: Test output data
            tolerance_degrees: Tolerance in degrees
        """
        y_pred = self.ik_nn.predict(X_test)
        
        # Calculate absolute errors in degrees
        errors_deg = np.degrees(np.abs(y_test - y_pred))
        
        # Check convergence for each joint
        convergence_rates = []
        for i in range(5):
            within_tolerance = np.sum(errors_deg[:, i] < tolerance_degrees)
            rate = within_tolerance / len(y_test) * 100
            convergence_rates.append(rate)
        
        overall_convergence = np.sum(np.all(errors_deg < tolerance_degrees, axis=1)) / len(y_test) * 100
        
        print(f"\nConvergence Analysis (Tolerance: {tolerance_degrees}°):")
        print("-" * 40)
        for i, rate in enumerate(convergence_rates):
            print(f"Joint θ{i+1}: {rate:.2f}% within tolerance")
        print(f"All joints simultaneously: {overall_convergence:.2f}% within tolerance")
        
        return convergence_rates, overall_convergence
    
    def test_workspace_coverage(self, X_test, y_test, workspace_limits=None):
        """
        Test performance across different workspace regions
        
        Args:
            X_test: Test input data  
            y_test: Test output data
            workspace_limits: Dictionary defining workspace regions
        """
        if workspace_limits is None:
            # Default workspace regions based on position
            workspace_limits = {
                'close': {'Px': (-1, 1), 'Py': (-1, 1), 'Pz': (-1, 1)},
                'medium': {'Px': (-2, 2), 'Py': (-2, 2), 'Pz': (-2, 2)},
                'far': {'Px': (-3, 3), 'Py': (-3, 3), 'Pz': (-3, 3)}
            }
        
        y_pred = self.ik_nn.predict(X_test)
        
        print(f"\nWorkspace Coverage Analysis:")
        print("-" * 40)
        
        for region_name, limits in workspace_limits.items():
            # Find points in this region
            mask = np.ones(len(X_test), dtype=bool)
            for i, (coord, (min_val, max_val)) in enumerate(limits.items()):
                coord_idx = ['Px', 'Py', 'Pz'].index(coord)
                mask &= (X_test[:, coord_idx] >= min_val) & (X_test[:, coord_idx] <= max_val)
            
            if np.sum(mask) > 0:
                region_mae = mean_absolute_error(y_test[mask], y_pred[mask])
                region_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
                
                print(f"{region_name.upper()} region ({np.sum(mask)} samples):")
                print(f"  MAE: {region_mae:.6f} rad ({np.degrees(region_mae):.3f}°)")
                print(f"  RMSE: {region_rmse:.6f} rad ({np.degrees(region_rmse):.3f}°)")
            else:
                print(f"{region_name.upper()} region: No samples found")
    
    def visualize_predictions(self, X_test, y_test, save_path=None):
        """
        Create comprehensive visualization of prediction accuracy
        
        Args:
            X_test: Test input data
            y_test: Test output data
            save_path: Path to save the plots
        """
        y_pred = self.ik_nn.predict(X_test)
        
        # Create subplot layout
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Scatter plots for each joint
        for i in range(5):
            ax = plt.subplot(2, 3, i + 1)
            plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.5, s=1)
            
            # Perfect prediction line
            min_val = min(y_test[:, i].min(), y_pred[:, i].min())
            max_val = max(y_test[:, i].max(), y_pred[:, i].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            plt.xlabel(f'Actual θ{i+1} (rad)')
            plt.ylabel(f'Predicted θ{i+1} (rad)')
            plt.title(f'Joint {i+1} Prediction Accuracy')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # 6. Error distribution
        ax = plt.subplot(2, 3, 6)
        errors = np.degrees(np.abs(y_test - y_pred))
        
        for i in range(5):
            plt.hist(errors[:, i], bins=50, alpha=0.6, label=f'θ{i+1}', density=True)
        
        plt.xlabel('Absolute Error (degrees)')
        plt.ylabel('Density')
        plt.title('Error Distribution by Joint')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def generate_test_report(self, X_test, y_test, save_dir="test_results"):
        """
        Generate a comprehensive test report
        
        Args:
            X_test: Test input data
            y_test: Test output data
            save_dir: Directory to save the report
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("Generating comprehensive test report...")
        
        # 1. Accuracy evaluation
        accuracy_results = self.evaluate_accuracy(X_test, y_test)
        
        # 2. Convergence analysis
        convergence_rates, overall_convergence = self.test_convergence_speed(X_test, y_test)
        
        # 3. Workspace coverage
        self.test_workspace_coverage(X_test, y_test)
        
        # 4. Visualizations
        viz_path = os.path.join(save_dir, 'prediction_analysis.png')
        self.visualize_predictions(X_test, y_test, viz_path)
        
        # 5. Save detailed results
        results_df = pd.DataFrame({
            'Joint': [f'θ{i+1}' for i in range(5)],
            'MAE_rad': accuracy_results['per_joint']['mae'],
            'MAE_deg': accuracy_results['per_joint']['mae_deg'],
            'RMSE_rad': accuracy_results['per_joint']['rmse'],
            'RMSE_deg': accuracy_results['per_joint']['rmse_deg'],
            'R2_Score': accuracy_results['per_joint']['r2'],
            'Convergence_Rate_%': convergence_rates
        })
        
        results_df.to_csv(os.path.join(save_dir, 'joint_analysis.csv'), index=False)
        
        # 6. Save predictions vs actual
        pred_df = pd.DataFrame(
            np.hstack([y_test, accuracy_results['predictions']]),
            columns=[f'θ{i+1}_actual' for i in range(5)] + [f'θ{i+1}_predicted' for i in range(5)]
        )
        pred_df.to_csv(os.path.join(save_dir, 'predictions_vs_actual.csv'), index=False)
        
        print(f"Test report saved to: {save_dir}/")
        
        return accuracy_results


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Test Inverse Kinematics Neural Network')
    parser.add_argument('--model_path', type=str, 
                       default='training_results/inverse_kinematics_final_model.h5',
                       help='Path to trained model')
    parser.add_argument('--scaler_input', type=str,
                       default='training_results/scaler_input.pkl',
                       help='Path to input scaler')
    parser.add_argument('--scaler_output', type=str,
                       default='training_results/scaler_output.pkl',
                       help='Path to output scaler')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data CSV files')
    parser.add_argument('--n_test_samples', type=int, default=5000,
                       help='Number of test samples to generate if no test data provided')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = InverseKinematicsEvaluator(
        model_path=args.model_path,
        scaler_input_path=args.scaler_input,
        scaler_output_path=args.scaler_output
    )
    
    # Load or generate test data
    if args.test_data:
        X_test = pd.read_csv(os.path.join(args.test_data, 'test_input.csv')).values
        y_test = pd.read_csv(os.path.join(args.test_data, 'test_output.csv')).values
    else:
        from inverse_kinematics_nn import generate_sample_data
        print(f"Generating {args.n_test_samples} test samples...")
        X_test, y_test = generate_sample_data(args.n_test_samples)
    
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Generate comprehensive test report
    results = evaluator.generate_test_report(X_test, y_test)
    
    return evaluator, results


if __name__ == "__main__":
    main()