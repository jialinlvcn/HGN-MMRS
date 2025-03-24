import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import enlighten
from torch.utils.data import DataLoader, Dataset
from tabulate import tabulate
import random

def generate_unique_colors(N):
    if N > 256**3:
        raise ValueError("N cannot be greater than 16777216 (256^3), as there are only 16777216 unique RGB colors.")

    unique_colors = set()
    while len(unique_colors) < N:
        # 随机生成一个 RGB 颜色
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        unique_colors.add(color)

    return list(unique_colors)

def generate_cmap(length, cmap_name="viridis"):
    colors = generate_unique_colors(int(length))
    colors[0] = (0, 0, 0)
    cmap_dict = {i: colors[i] for i in range(int(length))}
    return cmap_dict

def hsi2rgb(
    hsi_image: np.ndarray, rgb_channels: list, save_path: str, percentile_low=2, percentile_high=98, gamma=0.6
) -> np.ndarray:
    """
    Converts a hyperspectral image (HSI) to an RGB image by extracting specified channels,
    enhancing contrast, and applying gamma correction.

    Args:
        hsi_image (np.ndarray): The input hyperspectral image with shape (H, W, C).
        rgb_channels (list): A list of three indices specifying the channels to use for R, G, and B.
        save_path (str): The path to save the resulting RGB image.
        percentile_low (float, optional): The lower percentile for contrast stretching. Defaults to 2.
        percentile_high (float, optional): The upper percentile for contrast stretching. Defaults to 98.
        gamma (float, optional): The gamma correction value. Defaults to 0.6 (gamma < 1 brightens the image).

    Returns:
        np.ndarray: The generated RGB image.
    """
    # Extract the specified channels for R, G, and B
    red_channel = hsi_image[:, :, rgb_channels[0]]
    green_channel = hsi_image[:, :, rgb_channels[1]]
    blue_channel = hsi_image[:, :, rgb_channels[2]]

    # Combine the channels into an RGB image
    rgb_image = np.stack((red_channel, green_channel, blue_channel), axis=-1)

    # Apply percentile-based contrast stretching
    low = np.percentile(rgb_image, percentile_low)  # Calculate the lower percentile value
    high = np.percentile(rgb_image, percentile_high)  # Calculate the upper percentile value
    rgb_image = (rgb_image - low) / (high - low + 1e-8)  # Stretch the intensity range (avoid division by zero)
    rgb_image = np.clip(rgb_image, 0, 1)  # Clip values to the range [0, 1]

    # Apply gamma correction to adjust brightness
    rgb_image = rgb_image**gamma  # Gamma < 1 brightens the image

    # Save the resulting RGB image
    plt.imsave(save_path, rgb_image)

    return rgb_image


def norm_(image: np.ndarray) -> np.ndarray:
    """Normalizes each channel of a 3D image array to the range [0, 1]."""
    norm_image = image.copy()
    for i in range(image.shape[-1]):
        img = image[:, :, i]
        img = img - img.min()
        img = img / img.max()
        norm_image[:, :, i] = img
    return norm_image


def sar2grey(sar_image: np.ndarray, save_path: str) -> np.ndarray:
    """
    Converts a SAR (Synthetic Aperture Radar) image to a grayscale image by combining specific channels,
    normalizing the result, and saving it.

    Args:
        sar_image (np.ndarray): The input SAR image with shape (H, W, C).
        save_path (str): The path to save the resulting grayscale image.

    Returns:
        np.ndarray: The generated grayscale image.
    """
    # Normalize each channel of the SAR image to the range [0, 1]
    sar_image = norm_(sar_image)

    # Extract specific channels (VH, HV, VV)
    VH = sar_image[:, :, 2]
    HV = sar_image[:, :, 1]
    VV = sar_image[:, :, 3]

    # Combine the channels into a single grayscale image
    sar_image = VV + VH + HV

    # Normalize the combined image to the range [0, 1]
    sar_image = sar_image - sar_image.min()
    sar_image = sar_image / sar_image.max()
    sar_image = sar_image.astype(np.float32)

    # Create a figure with the appropriate size and resolution
    plt.figure(figsize=(sar_image.shape[1] / 100, sar_image.shape[0] / 100), dpi=100)

    # Display the grayscale image
    plt.imshow(sar_image, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")  # Hide the axes

    # Save the grayscale image
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    return sar_image


def lidar2grey(lidar_image: np.ndarray, save_path: str):
    """
    Converts a LiDAR image to a grayscale image and saves it.

    Args:
        lidar_image (np.ndarray): The input LiDAR image with shape (H, W) or (H, W, C).
        save_path (str): The path to save the resulting grayscale image.

    Returns:
        np.ndarray: The input LiDAR image, converted to grayscale if necessary.
    """
    # If the LiDAR image has multiple channels, use only the first channel
    if lidar_image.shape[-1] != 1:
        lidar_image = lidar_image[:, :, 0]

    # Convert the LiDAR image to float32 for consistency
    lidar_image = lidar_image.astype(np.float32)

    # Create a figure with the appropriate size and resolution
    plt.figure(figsize=(lidar_image.shape[1] / 100, lidar_image.shape[0] / 100), dpi=100)

    # Display the LiDAR image in grayscale
    plt.imshow(lidar_image, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")  # Hide the axes

    # Save the grayscale image
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    return lidar_image


def drawing_categories(categories_points, cmap: dict, save_path: str):
    """
    Converts a categorical label map into an RGB image using a provided color map and saves it.

    Args:
        categories_points (np.ndarray): A 2D or 3D array containing categorical labels.
        cmap (dict): A dictionary mapping category labels to RGB values.
        save_path (str): The path to save the resulting RGB image.

    Returns:
        np.ndarray: The generated RGB image.
    """
    # If the input is 3D, squeeze it to 2D
    if len(categories_points.shape) == 3:
        categories_points = np.squeeze(categories_points)

    # Get the height (H) and width (W) of the input array
    H, W = categories_points.shape

    # Initialize an empty RGB image with shape (H, W, 3)
    rgb_image = np.zeros((H, W, 3), dtype=np.float64)

    # Map each category to its corresponding RGB color
    for key, rgb_values in cmap.items():
        rgb_image[:, :, 0][categories_points == key] = rgb_values[0]  # Red channel
        rgb_image[:, :, 1][categories_points == key] = rgb_values[1]  # Green channel
        rgb_image[:, :, 2][categories_points == key] = rgb_values[2]  # Blue channel

    # Save the resulting RGB image
    plt.imsave(save_path, rgb_image)

    return rgb_image


def get_corrd(array, ignore):
    """
    Retrieves the coordinates and adjusted values of elements in an array that are not in the ignore set.

    Args:
        array (np.ndarray): The input 2D array.
        ignore (list): A list of values to ignore.

    Returns:
        list: A list of tuples, where each tuple contains:
            - The row index of the element.
            - The column index of the element.
            - The adjusted value of the element (original value minus the number of ignored values).
    """
    coordinates = np.argwhere(array >= 0)
    ignore_set = set(ignore)
    ignore = np.array(ignore)

    # Filter out coordinates with values in the ignore set and adjust the values
    result = [
        (coord[0], coord[1], array[coord[0], coord[1]] - (ignore < array[coord[0], coord[1]]).sum())
        for coord in coordinates
        if array[coord[0], coord[1]] not in ignore_set
    ]

    return result


class FullPatchDataset(Dataset):
    def __init__(self, multi_data, ignore=[], patch_size=7):
        """
        Initializes a dataset for extracting full patches from multimodal data.

        Args:
            multi_data (RSMultiData): An instance of the MultiData class containing multimodal data.
            ignore (list, optional): A list of values to ignore in the ground truth. Defaults to [].
            patch_size (int, optional): The size of the patches to extract. Defaults to 7.
        """
        self.multi_data = multi_data
        self.full_idx = get_corrd(multi_data.gt, ignore)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.full_idx)

    def __getitem__(self, i):
        # Get the position and label of the i-th patch
        x_pos, y_pos, gt = self.full_idx[i]

        # Calculate the start and end indices for rows and columns
        start_row = x_pos - self.patch_size // 2
        end_row = x_pos + self.patch_size // 2 + 1
        start_col = y_pos - self.patch_size // 2
        end_col = y_pos + self.patch_size // 2 + 1

        valid_start_row = max(start_row, 0)
        valid_end_row = min(end_row, self.multi_data.gt.shape[0])
        valid_start_col = max(start_col, 0)
        valid_end_col = min(end_col, self.multi_data.gt.shape[1])
        data_zip = []
        for data_type in self.multi_data.keys:
            value = self.multi_data.get(data_type)
            sliced_image = np.zeros((value.shape[-1], self.patch_size, self.patch_size), dtype=np.float32)
            sliced_image[
                :,
                valid_start_row - start_row : valid_end_row - start_row,
                valid_start_col - start_col : valid_end_col - start_col,
            ] = value[valid_start_row:valid_end_row, valid_start_col:valid_end_col, :].transpose(2, 0, 1)
            data_zip.append(sliced_image)
        return (data_zip, gt, x_pos, y_pos)


def print_color_preview(label, color):
    """Print RGB colors with color preview"""
    r, g, b = color
    # ANSI 转义序列设置背景颜色
    color_block = f"\033[48;2;{r};{g};{b}m      \033[0m"
    print(f"Class {label}: RGB({r}, {g}, {b}) {color_block}")


def draw_result(
    multi_data,
    save_path: str = None,
    manager: enlighten.Manager = None,
    model: torch.nn.Module = None,
    device: str = "cpu",
    patch_size: int = 7,
    ignore: list = [],
    draw_gt: bool = False,
    batch_size: int = 64,
    cmap: dict = None,
):
    """
    Generates a classification map visualization from model predictions or ground truth.

    Args:
        multi_data (RSMultiData): MultiData object containing multimodal data.
        save_path (str, optional): Path to save the generated visualization. Defaults to None.
        manager (enlighten.Manager, optional): Progress tracking manager. Defaults to None.
        model (torch.nn.Module, optional): The trained model for making predictions. Defaults to None.
        device (str, optional): Device for computation ("cpu" or "cuda"). Defaults to "cpu".
        patch_size (int, optional): Size of patches to process. Defaults to 7.
        ignore (list, optional): List of class indices to ignore. Defaults to [].
        draw_gt (bool, optional): Flag to draw ground truth instead of predictions. Defaults to False.
        batch_size (int, optional): Batch size to iter
        cmap (dict, optional): Cmap to draw result

    Returns:
        np.ndarray: Generated RGB classification map.
    """
    # Initialize empty RGB image with same dimensions as ground truth
    H, W = multi_data.gt.shape
    rgb_image = np.zeros((H, W, 3), dtype=np.float64)

    if not cmap:
        cmap = generate_cmap(multi_data.gt.max() + 1)
        # for label, color in Augsburg_cmap.items():
        #     if label > 0:
        #         print_color_preview(label-1, color)
        table_data = [
            [
                label,
                f"RGB({color[0]}, {color[1]}, {color[2]})",
                f"\033[48;2;{color[0]};{color[1]};{color[2]}m      \033[0m",
            ]
            for label, color in cmap.items()
        ]
        print(
            tabulate(
                table_data,
                headers=["Class", "RGB Color", "Preview"],
                tablefmt="pretty",
                stralign="left",  # 设置文字左对齐
                numalign="left",  # 设置数字左对齐（如果有数字）
            )
        )
    elif multi_data.gt.max() > len(cmap):
        ValueError(
            f"The length of the cmap is too small, the length of \
                   the ground truth is {len(multi_data.gt.max())}, but the length of the Cmap is {len(cmap)}."
        )

    # Create dataset and dataloader for full image processing
    # if draw_gt or not model:
    #     ignore = ignore.remove(0) if 0 in ignore else ignore
    #     ignore = [itr - 1 for itr in ignore] if ignore else []
    data_set = FullPatchDataset(multi_data, ignore, patch_size)
    loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

    # Initialize progress bar if a manager is provided
    manager = manager if manager else enlighten.get_manager()
    ticks = manager.counter(
        total=len(loader), desc="Drawing classification maps", unit="pixel", color="red", leave=False
    )

    ignore_tensor = torch.tensor(ignore)

    # Process data in batches
    for _, (data, gt, x_pos, y_pos) in enumerate(loader):
        if draw_gt or not model:  # Visualize ground truth
            # Get colors from ground truth labels
            
            rgb_values = np.array([cmap[p.item() + (p >= ignore_tensor).sum().item()] for p in gt])
            rgb_image[x_pos, y_pos] = rgb_values[:, :3]
        else:  # Generate model predictions
            # Set model and data to evaluation mode and move to target device
            model.eval()
            model = model.to(device)
            data, gt = [d.to(device) for d in data], gt.to(device)

            # Get model predictions
            output = model(data)
            try:  # Get model predictions
                predict = torch.argmax(output, dim=1)
            except TypeError:
                predict = torch.argmax(output[0], dim=1)

            # Map predictions to colors (offset by 1 to match colormap)
            rgb_values = np.array([cmap[p.item() + 1] for p in predict])
            rgb_image[x_pos, y_pos] = rgb_values[:, :3]

        # Update progress bar if a manager is provided
        ticks.update()

    # Close the progress bar if a manager is provided
    ticks.close()

    # Normalize the RGB image to the range [0, 1]
    if rgb_image.max() <= 255 and rgb_image.min() >= 1:
        rgb_image = rgb_image / 255
    elif rgb_image.min() < 1 and rgb_image.max() == rgb_image.min():
        rgb_image = np.zeros_like(rgb_image)
    else:
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    # Save the RGB image if a save path is provided
    if save_path:
        # Create the directory for saving the output if it doesn't exist
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])

        plt.imsave(save_path, rgb_image)
        print(f"The result is successfully saved to the {save_path}")

    return rgb_image
