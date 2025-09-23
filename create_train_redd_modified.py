from dataset_management.redd.redd_parameters import *
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import os

import h5py

import tables  # Add this import at the top

DATA_DIRECTORY = './'  # Now expects redd.h5 in current directory
H5_FILE = 'redd.h5'
SAVE_PATH = 'kettle/'
AGG_MEAN = 522
AGG_STD = 814


with h5py.File('redd.h5', 'r') as f:
    def printname(name):
        print(name)
    f.visit(printname)
    obj = f['building3/elec/meter1/table']
    print("Type:", type(obj))
    print("Name:", obj.name)
    print("Shape:", obj.shape)
    print("Dtype:", obj.dtype)
    print("Attrs:", dict(obj.attrs))
    print("Chunks:", obj.chunks)
    print("Maxshape:", obj.maxshape)

def read_h5_data(h5file, house, channel, appliance_name=None):
    key = f'/building{house}/elec/meter{channel}/table'
    with tables.open_file(h5file, 'r') as f:
        if not f.__contains__(key):
            print(f"\nERROR: Dataset {key} not found in HDF5 file.")
            for node in f.walk_nodes():
                print(node._v_pathname)
            raise KeyError(f"Dataset {key} not found in HDF5 file.")
        table = f.get_node(key)
        arr = table.read()
        times = arr['index']
        values = arr['values_block_0'].reshape(-1)
    df = pd.DataFrame({'time': times, 'value': values})
    # Try different units for conversion
    try:
        df['time'] = pd.to_datetime(df['time'], unit='s')
    except Exception as e:
        print("Failed with unit='s':", e)
        try:
            df['time'] = pd.to_datetime(df['time'], unit='ms')
        except Exception as e2:
            print("Failed with unit='ms':", e2)
            try:
                df['time'] = pd.to_datetime(df['time'], unit='us')
            except Exception as e3:
                print("Failed with unit='us':", e3)
                try:
                    df['time'] = pd.to_datetime(df['time'], unit='ns')
                except Exception as e4:
                    print("Failed with unit='ns':", e4)
                    print("Sample index values:", times[:10])
                    raise
    if appliance_name:
        df.rename(columns={'value': appliance_name}, inplace=True)
    else:
        df.rename(columns={'value': f'mains{channel}'}, inplace=True)
    return df

def get_arguments():
    parser = argparse.ArgumentParser(description='sequence to point learning example for NILM')
    parser.add_argument('--appliance_name', type=str, default='kettle',
                        help='which appliance you want to train')
    parser.add_argument('--aggregate_mean', type=int, default=AGG_MEAN,
                        help='Mean value of aggregated reading (mains)')
    parser.add_argument('--aggregate_std', type=int, default=AGG_STD,
                        help='Std value of aggregated reading (mains)')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                        help='The directory to store the training data')
    return parser.parse_args()

args = get_arguments()  # <-- Add this line before main()

start_time = time.time()

def main():
    sample_seconds = 3
    validation_percent = 20
    nrows = None
    debug = False

    appliance_name = args.appliance_name
    print('\n' + appliance_name)
    train = pd.DataFrame(columns=['aggregate', appliance_name])

    training_houses = [3]
    test_house = 1

    train_start_str = "2011-04-21 19:41:24"
    train_end_str = "2011-04-22 19:41:21"
    val_start_str = "2011-05-23 10:31:24"
    val_end_str = "2011-05-24 10:31:21"

    print(f"Using house {training_houses[0]} with separate time periods:")
    print(f"Training: {train_start_str} to {train_end_str}")
    print(f"Validation: {val_start_str} to {val_end_str}")
    print(f"Target: 28,800 data points each with {sample_seconds}s sampling interval")

    # Process training house (house 3)
    for h in training_houses:
        print('Processing house', h, 'for training/validation')
        print(f'    Reading from {H5_FILE} for house {h}')

        # Read data from HDF5
        mains1_df = read_h5_data(H5_FILE, h, 1)
        mains2_df = read_h5_data(H5_FILE, h, 2)
        app_channel = params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]
        app_df = read_h5_data(H5_FILE, h, app_channel, appliance_name=appliance_name)

        mains1_df.set_index('time', inplace=True)
        mains2_df.set_index('time', inplace=True)
        mains_df = mains1_df.join(mains2_df, how='outer')
        mains_df['aggregate'] = mains_df.iloc[:].sum(axis=1)
        mains_df.reset_index(inplace=True)
        del mains_df['mains1'], mains_df['mains2']

        if debug:
            print("    mains_df:")
            print(mains_df.head())
            plt.plot(mains_df['time'], mains_df['aggregate'])
            plt.show()

        app_df.set_index('time', inplace=True)
        mains_df.set_index('time', inplace=True)

        # TRAINING
        print(f"    Processing TRAINING data from {train_start_str} to {train_end_str}")
        train_start_datetime = pd.Timestamp(train_start_str)
        train_end_datetime = pd.Timestamp(train_end_str)
        mains_train_df = mains_df[(mains_df.index >= train_start_datetime) & (mains_df.index <= train_end_datetime)]
        app_train_df = app_df[(app_df.index >= train_start_datetime) & (app_df.index <= train_end_datetime)]
        print(f"    Found {len(mains_train_df)} mains records and {len(app_train_df)} appliance records for training")
        train_align = mains_train_df.join(app_train_df, how='outer').resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)
        train_align = train_align.dropna()
        if len(train_align) > 28800:
            train_align = train_align.head(28800)
            print(f"    Training: Trimmed to exactly 28,800 data points")
        elif len(train_align) < 28800:
            print(f"    Training: Warning - Only {len(train_align)} data points available, less than target 28,800")
        else:
            print(f"    Training: Perfect - Exactly {len(train_align)} data points")
        train_align.reset_index(inplace=True)
        del train_align['time']
        mean = params_appliance[appliance_name]['mean']
        std = params_appliance[appliance_name]['std']
        train_align['aggregate'] = (train_align['aggregate'] - args.aggregate_mean) / args.aggregate_std
        train_align[appliance_name] = (train_align[appliance_name] - mean) / std

        # VALIDATION
        print(f"    Processing VALIDATION data from {val_start_str} to {val_end_str}")
        val_start_datetime = pd.Timestamp(val_start_str)
        val_end_datetime = pd.Timestamp(val_end_str)
        mains_val_df = mains_df[(mains_df.index >= val_start_datetime) & (mains_df.index <= val_end_datetime)]
        app_val_df = app_df[(app_df.index >= val_start_datetime) & (app_df.index <= val_end_datetime)]
        print(f"    Found {len(mains_val_df)} mains records and {len(app_val_df)} appliance records for validation")
        val_align = mains_val_df.join(app_val_df, how='outer').resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)
        val_align = val_align.dropna()
        if len(val_align) > 28800:
            val_align = val_align.head(28800)
            print(f"    Validation: Trimmed to exactly 28,800 data points")
        elif len(val_align) < 28800:
            print(f"    Validation: Warning - Only {len(val_align)} data points available, less than target 28,800")
        else:
            print(f"    Validation: Perfect - Exactly {len(val_align)} data points")
        val_align.reset_index(inplace=True)
        del val_align['time']
        val_align['aggregate'] = (val_align['aggregate'] - args.aggregate_mean) / args.aggregate_std
        val_align[appliance_name] = (val_align[appliance_name] - mean) / std

        del mains1_df, mains2_df, mains_df, app_df
        del mains_train_df, app_train_df, mains_val_df, app_val_df

        if debug:
            print("train_align:")
            print(train_align.head())
            plt.plot(train_align['aggregate'].values)
            plt.plot(train_align[appliance_name].values)
            plt.show()

    val_align.to_csv(args.save_path + appliance_name + '_validation_' + '.csv', mode='a', index=False, header=False)
    train_align.to_csv(args.save_path + appliance_name + '_training_.csv', mode='a', index=False, header=False)

    print("    Size of training set is {:.4f} M rows (from house 3 training period).".format(len(train_align) / 10 ** 6))
    print("    Size of validation set is {:.4f} M rows (from house 3 validation period).".format(len(val_align) / 10 ** 6))

    del train_align, val_align

    # TEST HOUSE
    h = test_house
    test_start_str = "2011-04-18 09:22:12"
    test_end_str = "2011-05-23 09:21:51"
    target_test_points = 915840

    print(f'\nProcessing house {h} for testing')
    print(f'Test period: {test_start_str} to {test_end_str}')
    print(f'Target: {target_test_points:,} data points')
    print(f'    Reading from {H5_FILE} for house {h}')

    mains1_df = read_h5_data(H5_FILE, h, 1)
    mains2_df = read_h5_data(H5_FILE, h, 2)
    app_channel = params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]
    app_df = read_h5_data(H5_FILE, h, app_channel, appliance_name=appliance_name)

    mains1_df.set_index('time', inplace=True)
    mains2_df.set_index('time', inplace=True)
    mains_df = mains1_df.join(mains2_df, how='outer')
    mains_df['aggregate'] = mains_df.iloc[:].sum(axis=1)
    mains_df.reset_index(inplace=True)
    del mains_df['mains1'], mains_df['mains2']

    app_df.set_index('time', inplace=True)
    mains_df.set_index('time', inplace=True)

    test_start_datetime = pd.Timestamp(test_start_str)
    test_end_datetime = pd.Timestamp(test_end_str)
    print(f"    Filtering test data from {test_start_datetime} to {test_end_datetime}")
    mains_df = mains_df[(mains_df.index >= test_start_datetime) & (mains_df.index <= test_end_datetime)]
    app_df = app_df[(app_df.index >= test_start_datetime) & (app_df.index <= test_end_datetime)]
    print(f"    Found {len(mains_df)} mains records and {len(app_df)} appliance records for testing")

    df_align = mains_df.join(app_df, how='outer').resample(str(sample_seconds) + 'S').mean().fillna(method='backfill', limit=1)
    df_align = df_align.dropna()
    if len(df_align) > target_test_points:
        df_align = df_align.head(target_test_points)
        print(f"    Test: Trimmed to exactly {target_test_points:,} data points")
    elif len(df_align) < target_test_points:
        print(f"    Test: Warning - Only {len(df_align):,} data points available, less than target {target_test_points:,}")
    else:
        print(f"    Test: Perfect - Exactly {len(df_align):,} data points")
    df_align.reset_index(inplace=True)
    del mains1_df, mains2_df, mains_df, app_df, df_align['time']

    mean = params_appliance[appliance_name]['mean']
    std = params_appliance[appliance_name]['std']
    df_align['aggregate'] = (df_align['aggregate'] - args.aggregate_mean) / args.aggregate_std
    df_align[appliance_name] = (df_align[appliance_name] - mean) / std

    df_align.to_csv(args.save_path + appliance_name + '_test_.csv', mode='a', index=False, header=False)
    print("    Size of test set is {:.4f} M rows ({:,} points from house 1).".format(len(df_align) / 10 ** 6, len(df_align)))

    del df_align

    print("\nPlease find files in: " + args.save_path)
    print("Total elapsed time: {:.2f} min.".format((time.time() - start_time) / 60))

    print(f"\nDataset Summary:")
    print(f"- Training: House 3 from {train_start_str} to {train_end_str} (28,800 target points)")
    print(f"- Validation: House 3 from {val_start_str} to {val_end_str} (28,800 target points)")
    print(f"- Testing: House 1 from {test_start_str} to {test_end_str} ({target_test_points:,} target points)")
    print(f"- Sampling interval: {sample_seconds} seconds")
    print(f"- Training/Validation: 24 hours each, Testing: ~35 days")
    print(f"- Total data points: 28,800 + 28,800 + {target_test_points:,} = {28800 + 28800 + target_test_points:,}")

if __name__ == '__main__':
    main()