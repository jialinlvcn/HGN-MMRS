import os

import imageio
import numpy as np
from scipy import io, misc


def split_data(
    gt_array: np.ndarray, train_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits the ground truth data into training and testing sets based on the specified training rate.

    Args:
        gt_array (np.ndarray): The ground truth array containing labeled data.
        train_rate (float): The proportion of data to use for training (e.g., 0.8 for 80% training data).

    Returns:
        tuple: A tuple containing the training matrix and the testing matrix.
    """
    # Find the indices of non-zero elements in the ground truth array
    non_zero_indices = np.argwhere(gt_array != 0)

    # Shuffle the non-zero indices to ensure randomness
    np.random.shuffle(non_zero_indices)

    # Calculate the number of samples and the size of the training set
    num_samples = len(non_zero_indices)
    train_size = int(train_rate * num_samples)

    # Split the indices into training and testing sets
    train_indices = non_zero_indices[:train_size]
    test_indices = non_zero_indices[train_size:]

    # Initialize empty matrices for training and testing data
    train_matrix = np.zeros_like(gt_array)
    test_matrix = np.zeros_like(gt_array)

    # Populate the training matrix with values from the ground truth array
    for idx in train_indices:
        train_matrix[idx[0], idx[1]] = gt_array[idx[0], idx[1]]

    # Populate the testing matrix with values from the ground truth array
    for idx in test_indices:
        test_matrix[idx[0], idx[1]] = gt_array[idx[0], idx[1]]

    return train_matrix, test_matrix


class RSMultiData:
    """_summary_"""

    def __init__(self, **keywords) -> None:
        """
        Initializes the MultiData class to handle multimodal data.

        Args:
            **keywords: Keyword arguments specifying the data files and parameters.
                       Required keys: at least one modal file (e.g., "HSI", "LiDAR").
                       Optional keys: "tr" (train labels), "ts" (test labels), "gt" (ground truth), "train_rate".

        Raises:
            RuntimeError: If no modal files are provided.
            ValueError: If there is a dimensional mismatch in the multimodal data.
        """
        # Ensure at least one modal file is provided
        if len(keywords.keys()) == 0:
            raise RuntimeError("Modal files are at least 1")

        # Extract the dataset name from the directory path
        # self.dataset_name = os.path.split(os.path.normpath(data_dir))[1]
        if "tr" in keywords.keys() & "ts" in keywords.keys():
            raise ValueError(
                "Lost parameter [tr] or [ts], [tr] and [ts] can only be passed as parameters at the same time.\n"
                "If you want to pass in only one label file, use the keyword [gt]."
            )

        # Load training and testing labels if provided
        if "tr" in keywords.keys() and "ts" in keywords.keys():
            self.tr = np.squeeze(
                load_data(keywords["tr"])[0]
            )  # load train labels data file
            self.ts = np.squeeze(
                load_data(keywords["ts"])[0]
            )  # load test labels data file
            self.gt = self.tr + self.ts
        elif "gt" in keywords.keys() and "train_rate" in keywords.keys():
            # Load ground truth and split into train and test sets if "gt" is provided
            self.gt = np.squeeze(load_data(keywords["gt"])[0])
            self.tr, self.ts = split_data(self.gt, keywords["train_rate"])
        elif (
            "tr" in keywords.keys()
            and "ts" in keywords.keys()
            and "train_rate" in keywords.keys()
        ):
            # Handle case where both "tr", "ts", and "train_rate" are provided
            self.tr = np.squeeze(
                load_data(keywords["tr"])[0]
            )  # load train labels data file
            self.ts = np.squeeze(
                load_data(keywords["ts"])[0]
            )  # load test labels data file
            self.gt = self.tr + self.ts
            self.tr, self.ts = split_data(self.gt, self.train_rate)
        elif "gt" in keywords.keys() and "train_rate" not in keywords.keys():
            self.gt = np.squeeze(load_data(keywords["gt"])[0])
            self.tr, self.ts = None, None
        else:
            self.gt, self.tr, self.ts = None, None, None

        # Remove processed keys from the keywords dictionary
        keys_to_pop = ["gt", "train_rate", "tr", "ts"]
        for key in keys_to_pop:
            try:
                keywords.pop(key, None)
            except KeyError:
                pass

        # Load and process modal data
        self.keys, data_shape = [], None
        self.multi_data = {}
        for data_type, filename in keywords.items():
            try:
                data, _ = load_data(filename)  # Load data from file
            except TypeError:
                data = None  # Handle cases where data cannot be loaded
            except RuntimeError:
                raise RuntimeError(
                    f"No data with np.ndarrary format found in {os.path.join(filename)}"
                )

            # Ensure data has the correct shape and normalize if necessary
            if len(data.shape) == 2:
                data = data[:, :, None]
            if len(data.shape) > 2:
                data = normalize_each(data)

            # Store the processed data in the multi_data dictionary
            self.multi_data[data_type] = data
            if not data_shape:
                data_shape = data.shape
            else:
                if data_shape != data.shape:
                    ValueError(
                        f"Mismatch of incoming data file. The shape of data is {data_shape}, \
                                    and the shape of filename is data.shape"
                    )
            self.keys.append(data_type)

        # Check for dimensional consistency across modalities
        if not check_dimensions(self.multi_data):
            raise ValueError("No matching data passed in.")

        if self.gt is None:
            self.gt = np.zeros(data_shape[:2], dtype=np.int64)

        self.tr = (
            self.tr if self.tr is not None else np.zeros(data_shape[:2], dtype=np.int64)
        )
        self.ts = (
            self.ts if self.ts is not None else np.zeros(data_shape[:2], dtype=np.int64)
        )

    def get(self, type_name: str) -> np.ndarray:
        """
        Retrieves the data for a specific modality.

        Args:
            type_name (str): The name of the modality to retrieve.

        Returns:
            np.ndarray: The data for the specified modality.
        """
        return self.multi_data[type_name]

    def __str__(self) -> str:
        """
        Returns a string representation of the MultiData object.

        Returns:
            str: A formatted string describing the dataset and its modalities.
        """
        start_line = f"{self.dataset_name:=^50s}\n"
        logs = (
            f"Multimodal data {self.dataset_name}:\n"
            f"The number of modalities is {len(self.multi_data.keys())}\n"
            f"The image size is {self.tr.shape[0]} x {self.tr.shape[1]}\n"
        )
        end_line = f"\n{'':=^50s}\n"

        return (
            start_line
            + logs
            + "\n".join(
                f"Modality Name: {modality}, Number of Channels: {channels.shape[2]}"
                for modality, channels in self.multi_data.items()
            )
            + end_line
        )


def load_with_path(data_path: str):
    """Loads data from a file based on its extension."""
    ext = os.path.splitext(data_path)[1]
    ext = ext.lower()
    if ext == ".mat":
        return io.loadmat(data_path)
    elif ext == ".tif" or ext == "tiff":
        return misc.imread(data_path)
    elif ext == ".hdr":
        return imageio.imread(data_path)
    else:
        raise ValueError(f"Cannot load {ext} type file")


def normalize_each(nddata: np.ndarray):
    """Normalizes each channel of the input 3D array to the range [0, 1]."""
    nddata = nddata.astype(dtype=np.float32)
    for i in range(nddata.shape[-1]):
        minimal = nddata[:, :, i].min()
        maximal = nddata[:, :, i].max()
        nddata[:, :, i] = (nddata[:, :, i] - minimal) / (maximal - minimal)
    return nddata


def load_from_dict(data: dict) -> np.ndarray:
    """Extracts a numpy array and metadata from a dictionary."""
    header = {}
    for name, item in data.items():
        if isinstance(item, np.ndarray):
            nddata = item
        else:
            header[name] = item
    assert isinstance(nddata, np.ndarray)

    return nddata, header


def check_dimensions(input_dict: dict):
    """Checks if all non-None arrays in the input dictionary have the same height and width."""
    filtered_values = [value for value in input_dict.values() if value is not None]
    if not filtered_values:
        return False
    first_shape = filtered_values[0].shape[:2]
    return all(value.shape[:2] == first_shape for value in filtered_values)


def load_data(data_path: str):
    """Loads data from a file located in the specified directory."""
    data_dict = load_with_path(data_path)
    data, header = load_from_dict(data_dict)
    return data, header
