import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from data_feeder import TrainSlidingWindowGenerator

print("Testing the data feeder fix...")
print("TensorFlow version:", tf.__version__)

try:
    # Create a generator with small parameters for testing
    gen = TrainSlidingWindowGenerator(
        file_name='datasets/fridge_training_.csv',
        chunk_size=5*10**2,
        batch_size=10,
        crop=1000,  # Small crop for quick testing
        shuffle=True,
        skip_rows=0,
        offset=299,
        ram_threshold=5*10**5
    )
    
    print("Loading dataset...")
    dataset = gen.load_dataset()
    print("✓ Dataset loaded successfully")
    
    print("Getting a batch...")
    batch = next(iter(dataset))
    x_batch, y_batch = batch
    
    print(f"✓ Batch retrieved successfully")
    print(f"  - Input shape: {x_batch.shape}")
    print(f"  - Output shape: {y_batch.shape}")
    print(f"  - Input dtype: {x_batch.dtype}")
    print(f"  - Output dtype: {y_batch.dtype}")
    
    # Verify the shapes are correct
    expected_window_size = 2 * 299 + 1  # 599
    assert x_batch.shape == (10, expected_window_size, 1), f"Input shape mismatch: {x_batch.shape}"
    assert y_batch.shape == (10, 1), f"Output shape mismatch: {y_batch.shape}"
    
    print("\n✅ All tests passed! The fix is working correctly.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
