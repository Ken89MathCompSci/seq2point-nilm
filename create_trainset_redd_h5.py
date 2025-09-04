from dataset_management.redd.redd_parameters import *
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import os
import h5py

DATA_FILE = 'redd.h5'
SAVE_PATH = 'kettle/'
AGG_MEAN = 522
AGG_STD = 814

def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to point learning example for NILM using HDF5 REDD data')
    parser.add_argument('--data_file', type=str, default=DATA_FILE,
                          help='The HDF5 file containing the REDD data')
    parser.add_argument('--appliance_name', type=str, default='kettle',
                          help='which appliance you want to train: kettle,microwave,fridge,dishwasher,washingmachine')
    parser.add_argument('--aggregate_mean',type=int,default=AGG_MEAN,
                        help='Mean value of aggregated reading (mains)')
    parser.add_argument('--aggregate_std',type=int,default=AGG_STD,
                        help='Std value of aggregated reading (mains)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                          help='The directory to store the training data')
    return parser.parse_args()

def load_meter_data_h5(h5_file_path, building, meter):
    """Load data from HDF5 file for specific building and meter using pandas HDFStore"""
    try:
        # Use pandas HDFStore to read the data
        with pd.HDFStore(h5_file_path, 'r') as store:
            key = f'/building{building}/elec/meter{meter}'
            df = store[key]
            
            # The dataframe should have a datetime index and power values
            # Reset index to make timestamp a column, then set it properly
            df = df.reset_index()
            
            # The columns might be named differently - let's be flexible
            if 'power' in df.columns:
                power_col = 'power'
            elif len(df.columns) > 1:
                # Assume the second column is power if no 'power' column
                power_col = df.columns[1]
            else:
                print(f"Warning: Could not identify power column for building{building}/meter{meter}")
                return pd.DataFrame()
            
            # Set up the dataframe with proper column names
            df = df.rename(columns={power_col: 'power'})
            df = df[['index', 'power']].copy()
            df['time'] = pd.to_datetime(df['index'])
            df = df[['time', 'power']].set_index('time')

            # Convert timezone-aware index to timezone-naive for easier comparison
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)

            # Flatten MultiIndex columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            return df
            
    except (KeyError, FileNotFoundError) as e:
        print(f"Warning: Could not find data for building{building}/meter{meter}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data for building{building}/meter{meter}: {e}")
        return pd.DataFrame()

def main():
    start_time = time.time()
    args = get_arguments()
    appliance_name = args.appliance_name
    
    print(f'Processing appliance: {appliance_name}')
    
    sample_seconds = 3  # 3 seconds to get 28,800 points in 24 hours
    nrows = None
    debug = False

    # Define specific time ranges for training, validation and test
    # Use timezone-aware timestamps to match the data
    train_start_str = "2011-04-21 19:41:24"
    train_end_str = "2011-04-22 19:41:21"
    val_start_str = "2011-05-23 10:31:24"
    val_end_str = "2011-05-24 10:31:21"
    test_start_str = "2011-04-18 09:22:12"
    test_end_str = "2011-05-23 09:21:51"

    # Create timezone-naive timestamps (data will be converted to UTC then made naive)
    train_start_datetime = pd.Timestamp(train_start_str)
    train_end_datetime = pd.Timestamp(train_end_str)
    val_start_datetime = pd.Timestamp(val_start_str)
    val_end_datetime = pd.Timestamp(val_end_str)
    test_start_datetime = pd.Timestamp(test_start_str)
    test_end_datetime = pd.Timestamp(test_end_str)
    
    target_train_points = 28800
    target_val_points = 28800  
    target_test_points = 915840
    
    print(f"Data configuration:")
    print(f"Training: Building 3 from {train_start_str} to {train_end_str} ({target_train_points:,} target points)")
    print(f"Validation: Building 3 from {val_start_str} to {val_end_str} ({target_val_points:,} target points)")
    print(f"Testing: Building 1 from {test_start_str} to {test_end_str} ({target_test_points:,} target points)")
    print(f"Sampling interval: {sample_seconds} seconds")

    # Get appliance parameters
    if appliance_name not in params_appliance:
        print(f"Error: Appliance '{appliance_name}' not found in parameters")
        return

    appliance_params = params_appliance[appliance_name]
    
    # Process data using the HDF5 file path (not opening with h5py)
    h5_file_path = args.data_file
    
    # TRAINING AND VALIDATION DATA (Building 3)
    print(f'\nProcessing building 3 for training/validation...')
    
    # Load mains data (meter1 + meter2)
    mains1_df = load_meter_data_h5(h5_file_path, 3, 1)
    mains2_df = load_meter_data_h5(h5_file_path, 3, 2)
    
    if mains1_df.empty or mains2_df.empty:
        print("Error: Could not load mains data from building 3")
        return
        
    # Combine mains
    mains_df = mains1_df.join(mains2_df, how='outer', rsuffix='_2')
    mains_df['aggregate'] = mains_df[['power', 'power_2']].sum(axis=1, skipna=True)
    mains_df = mains_df[['aggregate']]

    # Ensure column names are strings, not tuples
    mains_df.columns = [str(col) for col in mains_df.columns]
    
    # Load appliance data
    # Get the meter number for this appliance in building 3
    house_idx = appliance_params['houses'].index(3)
    appliance_meter = appliance_params['channels'][house_idx]
    
    app_df = load_meter_data_h5(h5_file_path, 3, appliance_meter)
    if app_df.empty:
        print(f"Error: Could not load appliance data for {appliance_name} from building 3")
        return
    app_df = app_df.rename(columns={'power': appliance_name})

    # Ensure column names are strings, not tuples
    app_df.columns = [str(col) for col in app_df.columns]
        
    print(f"    Loaded mains: {len(mains_df)} records, appliance ({appliance_name}): {len(app_df)} records")
    
    # PROCESS TRAINING DATA
    print(f"    Processing TRAINING data from {train_start_str} to {train_end_str}")
    
    # Filter to training time range
    mains_train = mains_df[(mains_df.index >= train_start_datetime) & (mains_df.index <= train_end_datetime)]
    app_train = app_df[(app_df.index >= train_start_datetime) & (app_df.index <= train_end_datetime)]
    
    print(f"    Found {len(mains_train)} mains and {len(app_train)} appliance records for training")
    
    # Join and resample - use merge instead of join to be more robust
    try:
        train_combined = pd.merge(mains_train, app_train, left_index=True, right_index=True, how='outer')
        train_resampled = train_combined.resample(f'{sample_seconds}s').mean().bfill(limit=1)
        train_resampled = train_resampled.dropna()
    except Exception as e:
        print(f"Error in training data merge: {e}")
        print(f"Mains train shape: {mains_train.shape}, columns: {mains_train.columns.tolist()}")
        print(f"App train shape: {app_train.shape}, columns: {app_train.columns.tolist()}")
        return
    
    # Ensure exactly target number of points
    if len(train_resampled) > target_train_points:
        train_resampled = train_resampled.head(target_train_points)
        print(f"    Training: Trimmed to exactly {target_train_points:,} data points")
    elif len(train_resampled) < target_train_points:
        print(f"    Training: Warning - Only {len(train_resampled):,} data points available")
    else:
        print(f"    Training: Perfect - Exactly {len(train_resampled):,} data points")
        
    # Normalize training data
    mean = appliance_params['mean']
    std = appliance_params['std']
    train_resampled['aggregate'] = (train_resampled['aggregate'] - args.aggregate_mean) / args.aggregate_std
    train_resampled[appliance_name] = (train_resampled[appliance_name] - mean) / std
    
    # PROCESS VALIDATION DATA
    print(f"    Processing VALIDATION data from {val_start_str} to {val_end_str}")
    
    # Filter to validation time range
    mains_val = mains_df[(mains_df.index >= val_start_datetime) & (mains_df.index <= val_end_datetime)]
    app_val = app_df[(app_df.index >= val_start_datetime) & (app_df.index <= val_end_datetime)]
    
    print(f"    Found {len(mains_val)} mains and {len(app_val)} appliance records for validation")
    
    # Join and resample - use merge instead of join to be more robust
    try:
        val_combined = pd.merge(mains_val, app_val, left_index=True, right_index=True, how='outer')
        val_resampled = val_combined.resample(f'{sample_seconds}s').mean().bfill(limit=1)
        val_resampled = val_resampled.dropna()
    except Exception as e:
        print(f"Error in validation data merge: {e}")
        print(f"Mains val shape: {mains_val.shape}, columns: {mains_val.columns.tolist()}")
        print(f"App val shape: {app_val.shape}, columns: {app_val.columns.tolist()}")
        return
    
    # Ensure exactly target number of points
    if len(val_resampled) > target_val_points:
        val_resampled = val_resampled.head(target_val_points)
        print(f"    Validation: Trimmed to exactly {target_val_points:,} data points")
    elif len(val_resampled) < target_val_points:
        print(f"    Validation: Warning - Only {len(val_resampled):,} data points available")
    else:
        print(f"    Validation: Perfect - Exactly {len(val_resampled):,} data points")
        
    # Normalize validation data
    val_resampled['aggregate'] = (val_resampled['aggregate'] - args.aggregate_mean) / args.aggregate_std
    val_resampled[appliance_name] = (val_resampled[appliance_name] - mean) / std
    
    # Save training and validation sets
    train_resampled.to_csv(args.save_path + appliance_name + '_training_.csv', index=False, header=False)
    val_resampled.to_csv(args.save_path + appliance_name + '_validation_.csv', index=False, header=False)
    
    print(f"    Size of training set: {len(train_resampled):,} rows (from building 3 training period)")
    print(f"    Size of validation set: {len(val_resampled):,} rows (from building 3 validation period)")
        
    # TEST DATA (Building 1)
    print(f'\nProcessing building 1 for testing...')
    
    # Load mains data from building 1
    mains1_test_df = load_meter_data_h5(h5_file_path, 1, 1)
    mains2_test_df = load_meter_data_h5(h5_file_path, 1, 2)
    
    if mains1_test_df.empty or mains2_test_df.empty:
        print("Error: Could not load mains data from building 1")
        return
        
    # Combine mains
    mains_test_df = mains1_test_df.join(mains2_test_df, how='outer', rsuffix='_2')
    mains_test_df['aggregate'] = mains_test_df[['power', 'power_2']].sum(axis=1, skipna=True)
    mains_test_df = mains_test_df[['aggregate']]

    # Ensure column names are strings, not tuples
    mains_test_df.columns = [str(col) for col in mains_test_df.columns]
    
    # Load appliance data from building 1
    house_idx_test = appliance_params['houses'].index(1)
    appliance_meter_test = appliance_params['channels'][house_idx_test]
    
    app_test_df = load_meter_data_h5(h5_file_path, 1, appliance_meter_test)
    if app_test_df.empty:
        print(f"Error: Could not load appliance data for {appliance_name} from building 1")
        return
    app_test_df = app_test_df.rename(columns={'power': appliance_name})

    # Ensure column names are strings, not tuples
    app_test_df.columns = [str(col) for col in app_test_df.columns]
        
    print(f"    Test period: {test_start_str} to {test_end_str}")
    print(f"    Target: {target_test_points:,} data points")
    
    # Filter to test time range (already defined above)
    
    mains_test_filtered = mains_test_df[(mains_test_df.index >= test_start_datetime) & (mains_test_df.index <= test_end_datetime)]
    app_test_filtered = app_test_df[(app_test_df.index >= test_start_datetime) & (app_test_df.index <= test_end_datetime)]
    
    print(f"    Found {len(mains_test_filtered)} mains and {len(app_test_filtered)} appliance records for testing")
    
    # Join and resample - use merge instead of join to be more robust
    try:
        test_combined = pd.merge(mains_test_filtered, app_test_filtered, left_index=True, right_index=True, how='outer')
        test_resampled = test_combined.resample(f'{sample_seconds}s').mean().bfill(limit=1)
        test_resampled = test_resampled.dropna()
    except Exception as e:
        print(f"Error in test data merge: {e}")
        print(f"Mains test shape: {mains_test_filtered.shape}, columns: {mains_test_filtered.columns.tolist()}")
        print(f"App test shape: {app_test_filtered.shape}, columns: {app_test_filtered.columns.tolist()}")
        return
    
    # Ensure exactly target number of points
    if len(test_resampled) > target_test_points:
        test_resampled = test_resampled.head(target_test_points)
        print(f"    Test: Trimmed to exactly {target_test_points:,} data points")
    elif len(test_resampled) < target_test_points:
        print(f"    Test: Warning - Only {len(test_resampled):,} data points available")
    else:
        print(f"    Test: Perfect - Exactly {len(test_resampled):,} data points")
        
    # Normalize test data
    test_resampled['aggregate'] = (test_resampled['aggregate'] - args.aggregate_mean) / args.aggregate_std
    test_resampled[appliance_name] = (test_resampled[appliance_name] - mean) / std
    
    # Save test set
    test_resampled.to_csv(args.save_path + appliance_name + '_test_.csv', index=False, header=False)
    print(f"    Size of test set: {len(test_resampled):,} rows (from building 1)")

    print(f"\nFiles saved to: {args.save_path}")
    print(f"Total elapsed time: {(time.time() - start_time) / 60:.2f} min")
    
    print(f"\nDataset Summary:")
    print(f"- Training: Building 3 from {train_start_str} to {train_end_str} ({target_train_points:,} target points)")
    print(f"- Validation: Building 3 from {val_start_str} to {val_end_str} ({target_val_points:,} target points)")  
    print(f"- Testing: Building 1 from {test_start_str} to {test_end_str} ({target_test_points:,} target points)")
    print(f"- Sampling interval: {sample_seconds} seconds")
    print(f"- Total data points: {target_train_points + target_val_points + target_test_points:,}")

if __name__ == '__main__':
    main()
