import h5py
import pandas as pd

def examine_h5_structure(filename):
    print(f"Examining structure of {filename}")
    print("="*50)
    
    try:
        with h5py.File(filename, 'r') as f:
            print("Top-level keys:")
            for key in f.keys():
                print(f"  {key}")
                
            print("\nDetailed structure:")
            def print_structure(name, obj):
                print(name)
            f.visititems(print_structure)
            
    except Exception as e:
        print(f"Error with h5py: {e}")
        
    # Try with pandas HDFStore
    print("\n" + "="*50)
    print("Trying with pandas HDFStore:")
    try:
        store = pd.HDFStore(filename, 'r')
        print("Available keys:")
        for key in store.keys():
            print(f"  {key}")
        store.close()
    except Exception as e:
        print(f"Error with pandas HDFStore: {e}")

if __name__ == "__main__":
    examine_h5_structure("redd.h5")
