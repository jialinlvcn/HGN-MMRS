import os
import torch.nn as nn
import mm_model
import mmrs_utils

param_dict = {"Houston2013": [144, 1, 15], "Trento": [63, 1, 6], "Augsburg": [180, 4, 1, 7]}

Houston2013_cmap = {
    0: [0, 0, 0],
    1: [0, 205, 0],
    2: [127, 255, 0],
    3: [46, 139, 87],
    4: [0, 139, 0],
    5: [160, 82, 45],
    6: [0, 255, 255],
    7: [255, 255, 255],
    8: [216, 191, 216],
    9: [255, 0, 0],
    10: [139, 0, 0],
    11: [254, 19, 153],
    12: [255, 255, 0],
    13: [238, 154, 0],
    14: [85, 26, 139],
    15: [255, 127, 80],
}

Augsburg_cmap = {
    0: [0, 0, 0],
    1: [0, 205, 0],
    2: [233, 232, 79],
    3: [125, 197, 105],
    4: [241, 101, 106],
    5: [166, 91, 164],
    6: [105, 92, 168],
    7: [0, 255, 255],
}


def get_model(model_name: str, data_type: str, patch_size: int = 7) -> nn.Module:
    """
    Retrieves and initializes a specific model

    Args:
        model_name (str): The name of the fusion model to be initialized. \
        (S2E, Early, Late, Cross, Middle, End, FusAt, Decision, HGN)
        data_type (str): The type of data used to determine model parameters. (Houston2013, Trento or Augsburg)
        patch_size (int, optional): The size of the patch used. Defaults to 7.

    Returns:
        torch.nn.Module: An instance of the requested model initialized with the specified parameters.
    """
    # Extract parameters and number of classes based on the data type
    param = param_dict[data_type][:-1]
    class_num = param_dict[data_type][-1]

    # Initialize the appropriate model based on the model_name
    if model_name == "S2E":
        model = mm_model.S2ENet(param, n_classes=class_num, patch_size=patch_size)
    elif model_name == "Early":
        model = mm_model.Early_fusion_CNN(param, n_classes=class_num)
    elif model_name == "Late":
        model = mm_model.Late_fusion_CNN(param, n_classes=class_num)
    elif model_name == "Cross":
        model = mm_model.Cross_fusion_CNN(param, n_classes=class_num)
    elif model_name == "Middle":
        model = mm_model.Middle_fusion_CNN(param, n_classes=class_num)
    elif model_name == "End":
        model = mm_model.EndNet(param, n_classes=class_num)
    elif model_name == "FusAt":
        model = mm_model.FusAtNet(param, num_classes=class_num)
    elif model_name == "Decision":
        model = mm_model.Decision_fusion_CNN(param, n_classes=class_num)
    elif model_name == "HGN":
        model = mm_model.HGN(param, n_classes=class_num)

    return model


def get_dataset(
    data_dir: str, data_type: str, patch_size=7, p=1.0, slice_method="ignore", ignore=[0]
) -> tuple[mmrs_utils.RSMultiDataset, mmrs_utils.RSMultiDataset, mmrs_utils.RSMultiData]:
    """
    Loads and prepares the dataset for training and testing based on the provided parameters.

    Args:
        data_dir (str): The directory where the dataset is stored.
        data_type (str): The type of dataset to load (e.g., "Houston2013", "Trento", "Augsburg").
        patch_size (int, optional): The size of the patch to extract from the data. Defaults to 7.
        p (float, optional): The proportion of training data to use. Defaults to 1.0 (full dataset).
        slice_method (str, optional): The method for handling edge slices in the dataset. Defaults to "ignore".
        ignore (list, optional): List of labels to ignore in the dataset. Defaults to [0].

    Returns:
        tuple (MultiDataset, MultiDataset, MultiData): \
        A tuple containing the training dataset, testing dataset, and the raw data object.
    """
    # Determine the training label file based on the proportion `p`
    if p == 1.0:
        tr = "TRLabel.mat"
    else:
        tr = f"TRLabel-{int(p * 100)}.mat"

    # Load the dataset based on the specified data type
    if data_type == "Houston2013":
        data = mmrs_utils.RSMultiData(
            tr=os.path.join(data_dir, data_type, tr),
            ts=os.path.join(data_dir, data_type, "TSLabel.mat"),
            HSI=os.path.join(data_dir, data_type, "HSI.mat"),
            LiDAR=os.path.join(data_dir, data_type, "LiDAR.mat"),
        )
    elif data_type == "Trento":
        data = mmrs_utils.RSMultiData(
            tr=os.path.join(data_dir, data_type, tr),
            ts=os.path.join(data_dir, data_type, "TSLabel.mat"),
            HSI=os.path.join(data_dir, data_type, "HSI.mat"),
            LiDAR=os.path.join(data_dir, data_type, "LiDAR.mat"),
        )
    elif data_type == "Augsburg":
        data = mmrs_utils.RSMultiData(
            tr=os.path.join(data_dir, data_type, tr),
            ts=os.path.join(data_dir, data_type, "TSLabel.mat"),
            HSI=os.path.join(data_dir, data_type, "HSI.mat"),
            SAR=os.path.join(data_dir, data_type, "SAR.mat"),
            DSM=os.path.join(data_dir, data_type, "DSM.mat"),
        )

    # Create training and testing datasets using the loaded data
    trainset = mmrs_utils.RSMultiDataset(
        data, is_train=True, patch_size=patch_size, slice_method=slice_method, ignore=ignore
    )
    testset = mmrs_utils.RSMultiDataset(
        data, is_train=False, patch_size=patch_size, slice_method=slice_method, ignore=ignore
    )

    return trainset, testset, data
