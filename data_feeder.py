import numpy as np 
import pandas as pd 
import tensorflow as tf

# batch_size: the number of rows fed into the network at once.
# crop: the number of rows in the data set to be used in total.
# chunk_size: the number of lines to read from the file at once.

class TrainSlidingWindowGenerator():

    """Yields features and targets for training a ConvNet.

    Parameters:
    __file_name (string): The path where the training dataset is located.
    __batch_size (int): The size of each batch from the dataset to be processed.
    __chunk_size (int): The size of each chunk of data to be processed.
    __shuffle (bool): Whether the dataset should be shuffled before being returned.
    __offset (int):
    __crop (int): The number of rows of the dataset to return.
    __skip_rows (int): The number of rows of a dataset to skip before reading data.
    __ram_threshold (int): The maximum amount of RAM to utilise at a time.
    total_size (int): The number of rows read from the dataset.

    """

    def __init__(self, 
                file_name, 
                chunk_size, 
                shuffle, 
                offset, 
                batch_size=1000, 
                crop=100000, 
                skip_rows=0, 
                ram_threshold=5 * 10 ** 5):
        self.__file_name = file_name
        self.__batch_size = batch_size
        self.__chunk_size = 10 ** 8
        self.__shuffle = shuffle
        self.__offset = offset
        self.__crop = crop
        self.__skip_rows = skip_rows
        self.__ram_threshold = ram_threshold
        self.total_size = 0
        self.__total_num_samples = crop

    @property
    def total_num_samples(self):
        return self.__total_num_samples
    
    @total_num_samples.setter
    def total_num_samples(self, value):
        self.__total_num_samples = value

    def check_if_chunking(self):

        """Count the number of rows in the dataset and determine whether this is larger than the chunking
        threshold or not. """

        # Loads the file and counts the number of rows it contains.
        print("Importing training file...")

        # First, get the total number of rows in the file without skiprows
        try:
            total_rows_in_file = sum(1 for _ in open(self.__file_name)) - 1  # -1 for header if present, but we use header=None
        except:
            total_rows_in_file = 0

        # Adjust skip_rows if it's larger than available data
        effective_skip_rows = self.__skip_rows if self.__skip_rows < total_rows_in_file else 0
        if self.__skip_rows > 0 and effective_skip_rows == 0:
            print(f"Warning: skip_rows ({self.__skip_rows}) is larger than file size ({total_rows_in_file}), using skip_rows=0")

        # Always use header=None since the CSV files don't have headers
        try:
            chunks = pd.read_csv(self.__file_name,
                                header=None,
                                nrows=self.__crop,
                                skiprows=effective_skip_rows if effective_skip_rows > 0 else None)
            print("Counting number of rows...")
            self.total_size = len(chunks)
            del chunks
            print("Done.")
        except pd.errors.EmptyDataError:
            print("Warning: No data found after skipping rows, using empty dataset")
            self.total_size = 0

        print("The dataset contains ", self.total_size, " rows")

        # Display a warning if there are too many rows to fit in the designated amount RAM.
        if (self.total_size > self.__ram_threshold):
            print("There is too much data to load into memory, so it will be loaded in chunks. Please note that this may result in decreased training times.")
    

    def _get_batch_generator(self):
        """Generator function that yields batches of data."""
        if self.total_size == 0:
            self.check_if_chunking()

        # If the data can be loaded in one go, don't skip any rows.
        if (self.total_size <= self.__ram_threshold):
            # Returns an array of the content from the CSV file.
            # First, get the total number of rows in the file
            try:
                total_rows_in_file = sum(1 for _ in open(self.__file_name))
            except:
                total_rows_in_file = 0

            # Adjust skip_rows if it's larger than available data
            effective_skip_rows = self.__skip_rows if self.__skip_rows < total_rows_in_file else 0

            # Always use header=None since the CSV files don't have headers
            try:
                data_array = np.array(pd.read_csv(self.__file_name,
                                                 nrows=self.__crop,
                                                 header=None,
                                                 skiprows=effective_skip_rows if effective_skip_rows > 0 else None))
            except pd.errors.EmptyDataError:
                # If no data after skipping, create empty array
                data_array = np.empty((0, 2))
            inputs = data_array[:, 0]
            outputs = data_array[:, 1]

            maximum_batch_size = inputs.size - 2 * self.__offset
            self.total_num_samples = maximum_batch_size
            if self.__batch_size < 0:
                self.__batch_size = maximum_batch_size

            indicies = np.arange(maximum_batch_size)
            if self.__shuffle:
                np.random.shuffle(indicies)

            while True:
                for start_index in range(0, maximum_batch_size, self.__batch_size):
                    splice = indicies[start_index : start_index + self.__batch_size]
                    input_data = np.array([inputs[index : index + 2 * self.__offset + 1] for index in splice])
                    output_data = outputs[splice + self.__offset].reshape(-1, 1)
                    
                    # Add channel dimension for CNN input
                    input_data = input_data.reshape(-1, 2 * self.__offset + 1, 1).astype(np.float32)
                    output_data = output_data.astype(np.float32)
                    
                    yield input_data, output_data
                    
        # Skip rows where needed to allow data to be loaded properly when there is not enough memory.
        else:  # Fixed the condition here
            number_of_chunks = np.arange(self.total_size / self.__chunk_size)
            if self.__shuffle:
                np.random.shuffle(number_of_chunks)

            # Yield the data in sections.
            for index in number_of_chunks:
                data_array = np.array(pd.read_csv(self.__file_name, skiprows=int(index) * self.__chunk_size, header=0, nrows=self.__crop))                   
                inputs = data_array[:, 0]
                outputs = data_array[:, 1]

                maximum_batch_size = inputs.size - 2 * self.__offset
                self.total_num_samples = maximum_batch_size
                if self.__batch_size < 0:
                    self.__batch_size = maximum_batch_size

                indicies = np.arange(maximum_batch_size)
                if self.__shuffle:
                    np.random.shuffle(indicies)

                while True:
                    for start_index in range(0, maximum_batch_size, self.__batch_size):
                        splice = indicies[start_index : start_index + self.__batch_size]
                        input_data = np.array([inputs[index : index + 2 * self.__offset + 1] for index in splice])
                        output_data = outputs[splice + self.__offset].reshape(-1, 1)
                        
                        # Add channel dimension for CNN input
                        input_data = input_data.reshape(-1, 2 * self.__offset + 1, 1).astype(np.float32)
                        output_data = output_data.astype(np.float32)
                        
                        yield input_data, output_data
    
    def load_dataset(self):
        """Returns a TensorFlow dataset that generates batches.

        Returns:
        tf.data.Dataset: A dataset object that yields pairs of features and targets.
        """
        # Define the window size
        window_size = 2 * self.__offset + 1

        # Use TensorFlow's from_generator to create a dataset
        # Use None for batch size since it can vary
        dataset = tf.data.Dataset.from_generator(
            self._get_batch_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, window_size, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
            )
        )

        # Add prefetch for better performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
                    
class TestSlidingWindowGenerator(object):

    """Yields features and targets for testing and validating a ConvNet.

    Parameters:
    __number_of_windows (int): The number of sliding windows to produce.
    __offset (int): The offset of the infered value from the sliding window.
    __inputs (numpy.ndarray): The available testing / validation features.
    __targets (numpy.ndarray): The target values corresponding to __inputs.
    __total_size (int): The total number of inputs.

    """

    def __init__(self, number_of_windows, inputs, targets, offset):
        self.__number_of_windows = number_of_windows
        self.__offset = offset
        self.__inputs = inputs
        self.__targets = targets
        self.total_size = len(inputs)

    def load_dataset(self):

        """Yields features and targets for testing and validating a ConvNet.

        Yields:
        input_data (numpy.array): An array of features to test / validate the network with.

        """

        self.__inputs = self.__inputs.flatten()
        max_number_of_windows = self.__inputs.size - 2 * self.__offset

        if self.__number_of_windows < 0:
            self.__number_of_windows = max_number_of_windows

        indicies = np.arange(max_number_of_windows, dtype=int)
        for start_index in range(0, max_number_of_windows, self.__number_of_windows):
            splice = indicies[start_index : start_index + self.__number_of_windows]
            input_data = np.array([self.__inputs[index : index + 2 * self.__offset + 1] for index in splice])
            target_data = self.__targets[splice + self.__offset].reshape(-1, 1)
            yield input_data, target_data
