import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model_structure import create_model, load_model
from data_feeder import TestSlidingWindowGenerator
from appliance_data import appliance_data, mains_data
import argparse

def calculate_nilm_metrics(predictions, ground_truth, threshold=10):
    """
    Calculate NILM-specific metrics including classification metrics and SAE

    Parameters:
    predictions (np.array): Model predictions (continuous power values)
    ground_truth (np.array): Ground truth power values
    threshold (float): Power threshold to determine on/off state (default 10W)

    Returns:
    dict: Dictionary containing all calculated metrics
    """

    # Convert to binary classification (on/off)
    pred_binary = (predictions >= threshold).astype(int)
    true_binary = (ground_truth >= threshold).astype(int)

    # Calculate confusion matrix
    cm = confusion_matrix(true_binary, pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Calculate classification metrics
    accuracy = accuracy_score(true_binary, pred_binary)
    precision = precision_score(true_binary, pred_binary, zero_division=0)
    recall = recall_score(true_binary, pred_binary, zero_division=0)
    f1 = f1_score(true_binary, pred_binary, zero_division=0)

    # Calculate regression metrics
    mae = np.mean(np.abs(predictions - ground_truth))
    mse = np.mean((predictions - ground_truth) ** 2)
    rmse = np.sqrt(mse)

    # Calculate SAE (Signal Aggregate Error) - total absolute error
    sae = np.sum(np.abs(predictions - ground_truth))

    # Calculate relative error metrics
    mean_true = np.mean(ground_truth)
    mape = np.mean(np.abs((predictions - ground_truth) / (ground_truth + 1e-6))) * 100  # Avoid division by zero

    return {
        'Confusion_Matrix': cm,
        'True_Negatives': tn,
        'False_Positives': fp,
        'False_Negatives': fn,
        'True_Positives': tp,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'SAE': sae,
        'MAPE': mape,
        'Mean_True_Power': mean_true
    }

def evaluate_model(appliance, network_type, algorithm, test_file, model_dir, threshold=10):
    """
    Evaluate a trained model on test data and calculate comprehensive metrics
    """

    print(f"Evaluating {appliance} model...")
    print(f"Test file: {test_file}")
    print(f"Model directory: {model_dir}")
    print(f"Threshold: {threshold}W")

    # Load test data
    data_frame = pd.read_csv(test_file, header=0)
    test_input = np.array(data_frame.iloc[:, 0], dtype=float)
    test_target = np.array(data_frame.iloc[:, 1], dtype=float)

    print(f"Test data shape: {test_input.shape}")

    # Create and load model
    model = create_model(599)  # Default input window length
    model_filename = f"{appliance}_{network_type}_model.keras"
    model_path = os.path.join(model_dir, model_filename)
    model = load_model(model, network_type, algorithm, appliance, model_path)

    # Create test generator
    window_size = 601  # 599 + 2
    window_offset = int(0.5 * window_size - 1)
    test_generator = TestSlidingWindowGenerator(
        number_of_windows=100,
        inputs=test_input,
        targets=test_target,
        offset=window_offset
    )

    # Generate predictions
    predictions = []
    ground_truths = []

    for input_batch, target_batch in test_generator.load_dataset():
        pred = model.predict(input_batch, verbose=0)
        predictions.extend(pred.flatten())
        ground_truths.extend(target_batch.flatten())

    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    # Denormalize predictions and ground truth
    predictions = (predictions * appliance_data[appliance]["std"]) + appliance_data[appliance]["mean"]
    ground_truths = (ground_truths * appliance_data[appliance]["std"]) + appliance_data[appliance]["mean"]

    # Ensure no negative values
    predictions = np.maximum(predictions, 0)
    ground_truths = np.maximum(ground_truths, 0)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Ground truth shape: {ground_truths.shape}")

    # Calculate metrics
    metrics = calculate_nilm_metrics(predictions, ground_truths, threshold)

    # Print results
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS FOR {appliance.upper()}")
    print("="*50)

    print("CONFUSION MATRIX (Threshold = {threshold}W):")
    print("Predicted | Off (0)    On (1)")
    print("Actual   |-----------------")
    print(f"Off (0)  | {metrics['True_Negatives']:6d}    {metrics['False_Positives']:6d}")
    print(f"On (1)   | {metrics['False_Negatives']:6d}    {metrics['True_Positives']:6d}")
    print()

    print("CLASSIFICATION METRICS (Threshold = {threshold}W):")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")

    print("\nREGRESSION METRICS:")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".2f")
    print(".4f")

    print("\nADDITIONAL METRICS:")
    print(".2f")
    print(".2f")

    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate NILM model with comprehensive metrics')
    parser.add_argument('--appliance', type=str, default='microwave',
                        help='Appliance to evaluate')
    parser.add_argument('--network_type', type=str, default='seq2point',
                        help='Network architecture type')
    parser.add_argument('--algorithm', type=str, default='seq2point',
                        help='Algorithm used for training')
    parser.add_argument('--test_file', type=str, default='datasets/microwave_test_.csv',
                        help='Path to test CSV file')
    parser.add_argument('--model_dir', type=str, default='saved_models/',
                        help='Directory containing saved models')
    parser.add_argument('--threshold', type=float, default=10.0,
                        help='Power threshold for on/off classification (Watts)')

    args = parser.parse_args()

    # Set default test file based on appliance if not specified
    if args.test_file == 'datasets/microwave_test_.csv' and args.appliance != 'microwave':
        args.test_file = f'datasets/{args.appliance}_test_.csv'

    # Run evaluation
    metrics = evaluate_model(
        args.appliance,
        args.network_type,
        args.algorithm,
        args.test_file,
        args.model_dir,
        args.threshold
    )

    print(f"\nEvaluation completed for {args.appliance}")

if __name__ == '__main__':
    main()
