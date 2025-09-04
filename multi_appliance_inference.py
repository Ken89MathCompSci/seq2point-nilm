import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from model_structure import create_model, load_model
from data_feeder import TestSlidingWindowGenerator
from appliance_data import appliance_data, mains_data

class MultiApplianceInference():
    """
    Multi-appliance energy disaggregation inference system.
    Runs multiple seq2point models (one per appliance) on the same aggregate power signal
    to disaggregate individual appliance consumptions.
    
    Parameters:
    appliances (list): List of appliance names to disaggregate
    saved_models_dir (str): Directory containing the saved models
    input_window_length (int): Window size for the models
    batch_size (int): Batch size for inference
    crop (int): Maximum number of data points to process
    """
    
    def __init__(self, appliances, saved_models_dir, input_window_length=599, 
                 batch_size=1000, crop=10000):
        self.appliances = appliances
        self.saved_models_dir = saved_models_dir
        self.input_window_length = input_window_length
        self.batch_size = batch_size
        self.crop = crop
        self.window_size = input_window_length + 2
        self.window_offset = int(0.5 * self.window_size - 1)
        self.number_of_windows = 100
        
        # Load all models
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all appliance models into memory"""
        print("Loading models...")
        for appliance in self.appliances:
            model_path = os.path.join(self.saved_models_dir, f"{appliance}_seq2point_model.h5")
            
            if os.path.exists(model_path):
                model = create_model(self.input_window_length)
                self.models[appliance] = load_model(model, "seq2point", "seq2point", 
                                                  appliance, model_path)
                print(f"Loaded {appliance} model from {model_path}")
            else:
                print(f"Warning: Model for {appliance} not found at {model_path}")
    
    def disaggregate(self, aggregate_data_path):
        """
        Perform multi-appliance disaggregation on aggregate power data
        
        Parameters:
        aggregate_data_path (str): Path to CSV file containing aggregate power data
        
        Returns:
        dict: Dictionary containing predictions for each appliance
        """
        # Load aggregate data
        aggregate_input, _ = self.load_aggregate_data(aggregate_data_path)
        
        results = {}
        predictions = {}
        
        print(f"Running inference for {len(self.models)} appliances...")
        
        # Run each model on the same aggregate data
        for appliance in self.models.keys():
            print(f"Processing {appliance}...")
            
            # Create test generator for this appliance
            # Note: We don't have ground truth targets, so we use zeros as placeholder
            dummy_targets = np.zeros(len(aggregate_input) - 2 * self.window_offset)
            test_generator = TestSlidingWindowGenerator(
                number_of_windows=self.number_of_windows,
                inputs=aggregate_input,
                targets=dummy_targets,
                offset=self.window_offset
            )
            
            # Calculate steps for prediction
            steps_per_epoch = np.round(int(test_generator.total_size / self.batch_size), decimals=0)
            
            # Get predictions from the model
            raw_predictions = self.models[appliance].predict(
                x=test_generator.load_dataset(),
                steps=steps_per_epoch,
                verbose=0
            )
            
            # Denormalize predictions back to original scale
            denormalized_predictions = self.denormalize_predictions(raw_predictions, appliance)
            
            # Store results
            predictions[appliance] = denormalized_predictions
            results[appliance] = {
                'predictions': denormalized_predictions,
                'mean_power': np.mean(denormalized_predictions),
                'max_power': np.max(denormalized_predictions),
                'total_energy': np.sum(denormalized_predictions)  # Approximate total energy
            }
        
        return results, predictions, aggregate_input
    
    def load_aggregate_data(self, data_path):
        """Load aggregate power data from CSV file"""
        print(f"Loading aggregate data from {data_path}")
        
        # Assume CSV has aggregate power in first column
        data_frame = pd.read_csv(data_path, nrows=self.crop, skiprows=0, header=0)
        aggregate_input = np.round(np.array(data_frame.iloc[:, 0], float), 6)
        
        # Create dummy targets (we don't have ground truth for disaggregation)
        dummy_targets = np.zeros(len(aggregate_input) - 2 * self.window_offset)
        
        del data_frame
        return aggregate_input, dummy_targets
    
    def denormalize_predictions(self, predictions, appliance):
        """Convert normalized predictions back to original power scale"""
        appliance_mean = appliance_data[appliance]["mean"]
        appliance_std = appliance_data[appliance]["std"]
        
        denormalized = (predictions * appliance_std) + appliance_mean
        
        # Ensure no negative power values
        denormalized[denormalized < 0] = 0
        
        return denormalized.flatten()
    
    def plot_disaggregation_results(self, results, predictions, aggregate_input, save_path=None):
        """Plot disaggregation results showing aggregate and individual appliances"""
        
        # Denormalize aggregate data
        aggregate_denormalized = (aggregate_input * mains_data["std"]) + mains_data["mean"]
        aggregate_plot_data = aggregate_denormalized[self.window_offset:-self.window_offset]
        
        plt.figure(figsize=(15, 10))
        
        # Plot aggregate power
        plt.subplot(len(predictions) + 1, 1, 1)
        plt.plot(aggregate_plot_data[:len(list(predictions.values())[0])], label='Aggregate Power', color='black', linewidth=2)
        plt.title('Multi-Appliance Energy Disaggregation Results')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot each appliance prediction
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, (appliance, pred) in enumerate(predictions.items()):
            plt.subplot(len(predictions) + 1, 1, i + 2)
            color = colors[i % len(colors)]
            plt.plot(pred[:len(aggregate_plot_data)], label=f'{appliance.title()}', color=color)
            plt.ylabel('Power (W)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            mean_power = np.mean(pred)
            max_power = np.max(pred)
            plt.text(0.02, 0.95, f'Mean: {mean_power:.1f}W, Max: {max_power:.1f}W', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        
        plt.xlabel('Time Steps')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def print_summary_statistics(self, results):
        """Print summary statistics for all appliances"""
        print("\n" + "="*60)
        print("MULTI-APPLIANCE DISAGGREGATION SUMMARY")
        print("="*60)
        
        total_disaggregated = 0
        
        for appliance, stats in results.items():
            print(f"\n{appliance.upper()}:")
            print(f"  Mean Power: {stats['mean_power']:.2f} W")
            print(f"  Max Power: {stats['max_power']:.2f} W")
            print(f"  Total Energy: {stats['total_energy']:.2f} W⋅timesteps")
            
            total_disaggregated += stats['total_energy']
        
        print(f"\nTotal Disaggregated Energy: {total_disaggregated:.2f} W⋅timesteps")
        print("="*60)

def main():
    """Example usage of multi-appliance inference"""
    
    # Define appliances to disaggregate
    appliances = ['kettle', 'fridge', 'dishwasher', 'microwave', 'washingmachine']
    
    # Path to saved models directory
    saved_models_dir = "saved_models/"
    
    # Initialize multi-appliance inference system
    disaggregator = MultiApplianceInference(
        appliances=appliances,
        saved_models_dir=saved_models_dir,
        input_window_length=599,
        batch_size=1000,
        crop=10000
    )
    
    # Path to aggregate power data (CSV with aggregate power in first column)
    aggregate_data_path = "path/to/your/aggregate_power_data.csv"
    
    try:
        # Perform disaggregation
        results, predictions, aggregate_input = disaggregator.disaggregate(aggregate_data_path)
        
        # Print summary statistics
        disaggregator.print_summary_statistics(results)
        
        # Plot results
        disaggregator.plot_disaggregation_results(
            results, predictions, aggregate_input, 
            save_path="multi_appliance_results.png"
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have:")
        print("1. Trained models saved in the specified directory")
        print("2. Aggregate power data CSV file")
        print("3. Updated the paths in the main() function")

if __name__ == "__main__":
    main()
