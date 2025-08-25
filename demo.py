"""
Demonstration Script for Neural Network-Based Inverse Kinematics

This script provides a quick demonstration of the inverse kinematics neural network
with a simple example that showcases the complete workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
from inverse_kinematics_nn import InverseKinematicsNN, generate_sample_data
import time


def quick_demo():
    """Quick demonstration of the inverse kinematics neural network"""
    
    print("🤖 NEURAL NETWORK-BASED INVERSE KINEMATICS DEMO")
    print("=" * 55)
    
    # Initialize the neural network
    print("1. Building neural network architecture...")
    ik_nn = InverseKinematicsNN()
    model = ik_nn.build_model()
    
    print("   ✅ Model architecture:")
    print("      - Input: 6 neurons (Px, Py, Pz, Rx, Ry, Rz)")
    print("      - Hidden: 48 → 120 → 60 neurons")
    print("      - Output: 5 neurons (θ1, θ2, θ3, θ4, θ5)")
    print("      - Activation: Sigmoid (hidden), Linear (output)")
    print("      - Dropout: 0.05 in last two hidden layers")
    
    # Generate sample data
    print("\n2. Generating sample training data...")
    X, y = generate_sample_data(n_samples=5000)  # Smaller dataset for demo
    print(f"   ✅ Generated {X.shape[0]} samples")
    print(f"      - Input shape: {X.shape}")
    print(f"      - Output shape: {y.shape}")
    
    # Prepare data
    print("\n3. Preparing data for training...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = ik_nn.prepare_data(X, y)
    print(f"   ✅ Data split:")
    print(f"      - Training: {X_train.shape[0]} samples")
    print(f"      - Validation: {X_val.shape[0]} samples") 
    print(f"      - Test: {X_test.shape[0]} samples")
    
    # Train the model (reduced epochs for demo)
    print("\n4. Training the neural network...")
    print("   ⏳ Training in progress...")
    
    start_time = time.time()
    history = ik_nn.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,  # Reduced for demo
        batch_size=32,
        verbose=0  # Suppress training output for cleaner demo
    )
    training_time = time.time() - start_time
    
    print(f"   ✅ Training completed in {training_time:.1f} seconds")
    print(f"      - Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"      - Final validation loss: {history.history['val_loss'][-1]:.6f}")
    
    # Evaluate the model
    print("\n5. Evaluating model performance...")
    results, predictions = ik_nn.evaluate(X_test, y_test)
    
    # Calculate detailed metrics
    errors_rad = np.abs(y_test - predictions)
    errors_deg = np.degrees(errors_rad)
    
    print("   ✅ Performance metrics:")
    print(f"      - Mean Absolute Error: {np.mean(errors_rad):.6f} rad ({np.mean(errors_deg):.3f}°)")
    print(f"      - Max Absolute Error: {np.max(errors_rad):.6f} rad ({np.max(errors_deg):.3f}°)")
    
    print("\n   📊 Per-joint analysis:")
    for i in range(5):
        mae_joint = np.mean(errors_rad[:, i])
        mae_joint_deg = np.mean(errors_deg[:, i])
        print(f"      - Joint θ{i+1}: {mae_joint:.6f} rad ({mae_joint_deg:.3f}°)")
    
    # Test with a specific example
    print("\n6. Testing with specific examples...")
    
    # Take a few test samples
    n_examples = 3
    test_indices = np.random.choice(len(X_test), n_examples, replace=False)
    
    print("   🎯 Sample predictions:")
    for i, idx in enumerate(test_indices):
        input_pose = X_test[idx:idx+1]  # Keep 2D shape
        actual_angles = y_test[idx]
        predicted_angles = ik_nn.predict(input_pose, denormalize=True)[0]
        
        error = np.abs(actual_angles - predicted_angles)
        error_deg = np.degrees(error)
        
        print(f"\n      Example {i+1}:")
        print(f"      Input pose: [{input_pose[0, 0]:.3f}, {input_pose[0, 1]:.3f}, {input_pose[0, 2]:.3f}, "
              f"{input_pose[0, 3]:.3f}, {input_pose[0, 4]:.3f}, {input_pose[0, 5]:.3f}]")
        print(f"      Actual angles:    [{actual_angles[0]:.3f}, {actual_angles[1]:.3f}, {actual_angles[2]:.3f}, "
              f"{actual_angles[3]:.3f}, {actual_angles[4]:.3f}] rad")
        print(f"      Predicted angles: [{predicted_angles[0]:.3f}, {predicted_angles[1]:.3f}, {predicted_angles[2]:.3f}, "
              f"{predicted_angles[3]:.3f}, {predicted_angles[4]:.3f}] rad")
        print(f"      Errors: [{error[0]:.4f}, {error[1]:.4f}, {error[2]:.4f}, "
              f"{error[3]:.4f}, {error[4]:.4f}] rad")
        print(f"      Errors: [{error_deg[0]:.2f}°, {error_deg[1]:.2f}°, {error_deg[2]:.2f}°, "
              f"{error_deg[3]:.2f}°, {error_deg[4]:.2f}°]")
    
    # Visualization
    print("\n7. Creating visualization...")
    
    # Simple accuracy plot
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training history
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Prediction accuracy for first joint
    plt.subplot(1, 3, 2)
    plt.scatter(y_test[:, 0], predictions[:, 0], alpha=0.6, s=10)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], 
             [y_test[:, 0].min(), y_test[:, 0].max()], 'r--', lw=2)
    plt.xlabel('Actual θ1 (rad)')
    plt.ylabel('Predicted θ1 (rad)')
    plt.title('Joint 1 Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    plt.subplot(1, 3, 3)
    for i in range(5):
        plt.hist(errors_deg[:, i], bins=20, alpha=0.6, label=f'θ{i+1}', density=True)
    plt.xlabel('Absolute Error (degrees)')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✨ Demo completed successfully!")
    print("\nNext steps:")
    print("   • Use 'python train_model.py' for full training (2000 epochs)")
    print("   • Use 'python test_model.py' for comprehensive evaluation")
    print("   • Replace sample data with your robot's actual kinematics")
    print("   • Adjust hyperparameters based on your specific requirements")
    
    return ik_nn, history, results


def architecture_info():
    """Display detailed architecture information"""
    
    print("\n🏗️  DETAILED ARCHITECTURE INFORMATION")
    print("=" * 45)
    
    ik_nn = InverseKinematicsNN()
    model = ik_nn.build_model()
    
    print("Neural Network Specifications:")
    print("-" * 30)
    print("Input Layer:")
    print("  • 6 neurons representing pose: [Px, Py, Pz, Rx, Ry, Rz]")
    print("  • Px, Py, Pz: End-effector position coordinates")
    print("  • Rx, Ry, Rz: End-effector orientation angles")
    
    print("\nHidden Layers:")
    print("  • Layer 1: 48 neurons, sigmoid activation")
    print("  • Layer 2: 120 neurons, sigmoid activation, 0.05 dropout")
    print("  • Layer 3: 60 neurons, sigmoid activation, 0.05 dropout")
    
    print("\nOutput Layer:")
    print("  • 5 neurons representing joint angles: [θ1, θ2, θ3, θ4, θ5]")
    print("  • Linear activation (regression problem)")
    
    print("\nHyperparameters (from Table 4):")
    print("  • Learning rate: 0.001")
    print("  • Epochs: 2000")
    print("  • Optimizer: Adam")
    print("  • Loss function: MSE (Mean Squared Error)")
    
    print("\nModel Summary:")
    print("-" * 20)
    model.summary()
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"\nTotal trainable parameters: {total_params:,}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural Network Inverse Kinematics Demo')
    parser.add_argument('--info', action='store_true', help='Show detailed architecture information')
    parser.add_argument('--quick', action='store_true', help='Run quick demonstration')
    
    args = parser.parse_args()
    
    if args.info:
        architecture_info()
    elif args.quick or len(args.__dict__) == 0:  # Default to quick demo
        quick_demo()
    else:
        print("Use --quick for demonstration or --info for architecture details")