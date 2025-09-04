import h5py
import pandas as pd

def examine_sample_data():
    print("Examining sample data from redd.h5")
    print("=" * 50)
    
    try:
        with h5py.File('redd.h5', 'r') as f:
            # Check building1, meter1 data
            print("Sample data from building1/elec/meter1:")
            meter1_data = f['building1/elec/meter1/table']
            print(f"Shape: {meter1_data.shape}")
            print(f"Columns: {meter1_data.dtype.names}")
            
            # Load first few rows to see the structure
            sample_df = pd.DataFrame(meter1_data[:10])
            print(f"First 10 rows:")
            print(sample_df)
            
            # Check building1, meter2 (mains might be different)
            print(f"\nSample data from building1/elec/meter2:")
            meter2_data = f['building1/elec/meter2/table']
            print(f"Shape: {meter2_data.shape}")
            print(f"Columns: {meter2_data.dtype.names}")
            
            sample_df2 = pd.DataFrame(meter2_data[:10])
            print(f"First 10 rows:")
            print(sample_df2)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    examine_sample_data()
