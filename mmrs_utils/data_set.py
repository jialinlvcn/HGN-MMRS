import numpy as np
from numba import njit, prange
from torch.utils.data import Dataset
from .data_loader import RSMultiData


@njit(parallel=True)  # Use Numba for parallel execution and JIT compilation
def slice_with_padding_numba(value, centers, patch_size, H, W) -> np.ndarray:
    """
    Slices patches from a 3D array (H x W x C) around specified centers with padding.

    Args:
        value (np.ndarray): The input 3D array with shape (H, W, C).
        centers (np.ndarray): An array of center coordinates with shape (N, 2).
        patch_size (int): The size of the patches to extract.
        H (int): The height of the input array.
        W (int): The width of the input array.

    Returns:
        np.ndarray: An array of sliced patches with shape (N, C, patch_size, patch_size).
    """
    # Initialize an array to store the sliced patches
    sliced_images = np.zeros((len(centers), value.shape[-1], patch_size, patch_size))

    # Iterate over each center in parallel
    for i in prange(len(centers)):
        center = centers[i]

        # Calculate the start and end indices for rows and columns
        start_row = center[0] - patch_size // 2
        end_row = center[0] + patch_size // 2 + 1
        start_col = center[1] - patch_size // 2
        end_col = center[1] + patch_size // 2 + 1

        # Ensure the indices are within the bounds of the input array
        valid_start_row = max(start_row, 0)
        valid_end_row = min(end_row, H)
        valid_start_col = max(start_col, 0)
        valid_end_col = min(end_col, W)

        # Extract the valid region and place it in the output array
        sliced_images[
            i,
            :,
            valid_start_row - start_row : valid_end_row - start_row,
            valid_start_col - start_col : valid_end_col - start_col,
        ] = value[
            valid_start_row:valid_end_row, valid_start_col:valid_end_col, :
        ].transpose(2, 0, 1)

    return sliced_images


def patch_cut(value: np.ndarray, centers: np.ndarray, half_size: int, patch_size: int):
    """
    Extracts patches from a 3D array (H x W x C) around specified centers.

    Args:
        value (np.ndarray): The input 3D array with shape (H, W, C).
        centers (np.ndarray): An array of center coordinates with shape (N, 3).
                             Each center is represented as (x, y, _).
        half_size (int): Half of the patch size, used to calculate the patch boundaries.
        patch_size (int): The total size of the patches to extract.

    Returns:
        np.ndarray: An array of extracted patches with shape (N, C, patch_height, patch_width).
    """
    patches = []
    p = half_size

    # Iterate over each center
    for i in prange(len(centers)):
        x, y, _ = centers[i]

        # Calculate the start and end indices for rows (x-axis)
        x_start = max(x - p, 0)
        x_end = min(x + (patch_size - p), value.shape[0])

        # Calculate the start and end indices for columns (y-axis)
        y_start = max(y - p, 0)
        y_end = min(y + (patch_size - p), value.shape[1])

        # Extract the patch and transpose it to (C, H, W) format
        patch = value[x_start:x_end, y_start:y_end].transpose(2, 0, 1)
        patches.append(patch)

    return np.array(patches)  # Convert the list of patches to a numpy array


def slice_with_padding(data: dict, centers: list, patch_size: int) -> tuple[dict, list]:
    """
    Slices patches from multiple 3D arrays (H x W x C) around specified centers with padding.

    Args:
        data (dict): A dictionary where keys are modality names and values are 3D arrays (H x W x C).
        centers (list): An list of center coordinates.
        patch_size (int): The size of the patches to extract.

    Returns:
        tuple: A tuple containing:
            - sliced_data (dict): A dictionary of sliced patches for each modality.
            - centers (list): The original centers list.
    """
    H, W = list(data.values())[0].shape[0], list(data.values())[0].shape[1]
    sliced_data = {}
    for key, value in data.items():
        sliced_images = slice_with_padding_numba(value, centers, patch_size, H, W)
        sliced_data[key] = sliced_images

    return sliced_data, centers


def slice_with_ignore(data: dict, centers: list, patch_size: int) -> tuple[dict, list]:
    """
    Slices patches from multiple 3D arrays (H x W x C) around specified centers, ignoring centers
    that are too close to the edges of the array.

    Args:
        data (dict): A dictionary where keys are modality names and values are 3D arrays (H x W x C).
        centers (list): A list of center coordinates, where each center is represented as (x, y, gt).
        patch_size (int): The size of the patches to extract.

    Returns:
        tuple: A tuple containing:
            - sliced_data (dict): A dictionary of sliced patches for each modality.
            - indices (list): A list of valid center coordinates after filtering.
    """
    H, W = list(data.values())[0].shape[0], list(data.values())[0].shape[1]

    # Filter out centers that are too close to the edges of the array
    indices = [
        (x, y, gt)
        for x, y, gt in centers
        if x > patch_size // 2
        and x < H - patch_size // 2 - 1
        and y > patch_size
        and y < W - patch_size // 2 - 1
    ]

    # Slice patches from the data using the filtered centers
    sliced_data, indices = slice_with_padding(data, indices, patch_size)

    return sliced_data, indices  # Return the sliced data and the filtered centers


def get_nonzero(array: np.ndarray, ignore: list):
    """Retrieves the coordinates and values of non-zero elements in an array, excluding specified values."""
    coordinates = np.argwhere(array != 0)
    ignore_set = set(ignore)
    result = [
        (
            coord[0],
            coord[1],
            array[coord[0], coord[1]] - (array[coord[0], coord[1]] > ignore).sum(),
        )
        for coord in coordinates
        if array[coord[0], coord[1]] not in ignore_set
    ]
    return result


class RSMultiDataset(Dataset):
    def __init__(
        self,
        data: RSMultiData,
        is_train: bool,
        patch_size: int = 1,
        slice_method: str = "ignore",
        ignore: list = [0],
        is_flip: bool = False,
    ):
        """
        Initializes a dataset for multimodal data.

        Args:
            data (RSMultiData): An instance of the MultiData class containing multimodal data.
            is_train (bool): Whether the dataset is for training (True) or testing (False).
            patch_size (int, optional): The size of the patches to extract. Defaults to 1.
            slice_method (str, optional): The method for slicing patches. \
                Can be "ignore" or "padding". Defaults to "ignore".
            ignore (list, optional): A list of labels to ignore. Defaults to [0].
            is_flip (bool, optional): Whether to apply random flipping during training. Defaults to False.
        """
        self.multi_data = data.multi_data
        self.patch_size = patch_size
        self.is_train = is_train
        self.is_flip = is_flip
        if is_train:
            self.gt = data.tr
        else:
            self.gt = data.ts

        # Slice patches based on the specified method
        nonzero = get_nonzero(self.gt, ignore)
        if slice_method == "ignore":
            self.patchs, self.indices = slice_with_ignore(
                self.multi_data, nonzero, patch_size
            )
        elif slice_method == "padding":
            self.patchs, self.indices = slice_with_padding(
                self.multi_data, nonzero, patch_size
            )
        else:
            # Raise an error for unsupported slice methods
            RuntimeError(
                f'No such slice method {slice_method} \n \
                         You can choose "ignore" or "padding "'
            )
        self.modal_list = list(self.patchs.keys())

    def __len__(self) -> int:
        """
        Returns the number of patches in the dataset.

        Returns:
            int: The number of patches.
        """
        return len(self.indices)

    def flip(self, arrays):
        """
        Applies random flipping to the input arrays.

        Args:
            arrays (np.ndarray): The input array to flip.

        Returns:
            np.ndarray: The flipped array.
        """
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        axes = []
        if horizontal:
            axes.append(-1)
        if vertical:
            axes.append(-2)
        if axes:
            arrays = np.flip(arrays, axis=axes).copy()
        return arrays

    def __getitem__(self, idx: int):
        """
        Retrieves the i-th patch and its corresponding label and position.

        Args:
            i (int): The index of the patch to retrieve.

        Returns:
            tuple: A tuple containing:
                - data_zip (list): A list of patches from each modality.
                - gt (int): The label of the patch.
                - (x_pos, y_pos): The position of the patch in the original image.
        """
        x_pos, y_pos, gt = self.indices[idx]
        if self.is_train and self.is_flip:
            data_zip = [
                self.flip(d[idx].astype(np.float32)) for d in self.patchs.values()
            ]
        else:
            data_zip = [d[idx].astype(np.float32) for d in self.patchs.values()]
        if len(data_zip) == 1:
            data_zip = data_zip[0]
        return (data_zip, gt, (x_pos, y_pos))
