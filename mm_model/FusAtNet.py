"""
Re-implementation for paper "FusAtNet: Dual Attention based SpectroSpatial Multimodal Fusion Network for
Hyperspectral and LiDAR Classification"
The official keras implementation is in https://github.com/ShivamP1993/FusAtNet
"""

import torch
import torch.nn as nn


class ConvUnit(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        return x


class ConvUnit_NP(nn.Module):
    # No Padding
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, bias=True)
        self.bn = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        return x


class Residual_Unit1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        identity0 = self.activation(self.bn1(self.conv1(x)))
        identity1 = self.activation(self.bn2(self.conv2(identity0)))
        # x = self.activation(self.bn2(self.conv2(x))) + self.activation(self.bn1(self.conv1(x)))
        x = identity0 + identity1
        x = self.max_pool(x)
        return x


class Residual_Unit2(nn.Module):
    # without pooling
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        identity0 = self.activation(self.bn1(self.conv1(x)))
        identity1 = x = self.activation(self.bn2(self.conv2(identity0)))
        # identity = x
        # x = self.activation(self.bn2(self.conv2(x)))
        x = identity0 + identity1
        return x


class Hyper_Feature_Extractor(nn.Module):
    def __init__(self, input_channels, output_channels=1024):
        super().__init__()
        self.conv1 = ConvUnit(input_channels, 256)
        self.conv2 = ConvUnit(256, 256)
        self.conv3 = ConvUnit(256, 256)

        self.conv4 = ConvUnit(256, 256)
        self.conv5 = ConvUnit(256, 256)
        self.conv6 = ConvUnit(256, output_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        return out


class Spectral_Attention_Module(nn.Module):
    def __init__(self, input_channels, output_channels=1024):
        super().__init__()
        self.res1 = Residual_Unit1(input_channels, 256)
        self.res2 = Residual_Unit1(256, 256)
        self.conv1 = ConvUnit(256, 256)
        self.conv2 = ConvUnit(256, output_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.conv1(out)
        out = self.conv2(out)
        # out = self.max_pool(out)
        out = self.avg_pool(out)
        return out


class Spatial_Attention_module(nn.Module):
    def __init__(self, input_channels, output_channels=1024):
        super().__init__()
        self.res1 = Residual_Unit2(input_channels, 128)
        self.res2 = Residual_Unit2(128, 256)
        self.conv1 = ConvUnit(256, 256)
        self.conv2 = ConvUnit(256, output_channels)

    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


class Modality_Feature_Extractor(nn.Module):
    def __init__(self, input_channels, output_channels=1024):
        super().__init__()
        self.conv1 = ConvUnit(input_channels, 256)
        self.conv2 = ConvUnit(256, 256)
        self.conv3 = ConvUnit(256, 256)

        self.conv4 = ConvUnit(256, 256)
        self.conv5 = ConvUnit(256, 256)
        self.conv6 = ConvUnit(256, output_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        return out


class Modality_Attention_Module(nn.Module):
    def __init__(self, input_channels, output_channels=1024):
        super().__init__()
        self.res1 = Residual_Unit2(input_channels, 128)
        self.res2 = Residual_Unit2(128, 256)
        self.conv1 = ConvUnit(256, 256)
        self.conv2 = ConvUnit(256, output_channels)

    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


class Classification_Module(nn.Module):
    # Re-implemented early_fusion_CNN for paper "More Diverse Means Better:
    # Multimodal Deep Learning Meets Remote-Sensing Imagery Classiﬁcation"
    # But not use APs to convert 1-band LiDAR data to 21-band.
    def __init__(self, input_channels, n_classes):
        super().__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 'SAME' mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # For concatenated image x (7×7×d)
        self.conv1 = nn.Conv2d(input_channels, filters[0], kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2 = nn.BatchNorm2d(filters[1])
        # Max pooling ('SAME' mode) --> 4×4×32
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(filters[2])
        self.conv4 = nn.Conv2d(filters[2], filters[3], (1, 1))
        self.bn4 = nn.BatchNorm2d(filters[3])
        # Max pooling ('SAME' mode) --> 2×2×128
        self.conv5 = nn.Conv2d(filters[3], filters[3], (1, 1))
        self.bn5 = nn.BatchNorm2d(filters[3])
        self.conv6 = nn.Conv2d(filters[3], filters[2], (1, 1))
        self.bn6 = nn.BatchNorm2d(filters[2])
        # Average Pooling --> 1×1×64    # Use AdaptiveAvgPool2d() for more robust

        self.conv7 = nn.Conv2d(filters[2], n_classes, (1, 1))

        # weight_init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        # for image x1, x2
        x = self.activation(self.bn1(self.conv1(data)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.max_pool(x)
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.max_pool(x)

        x = self.activation(self.bn5(self.conv5(x)))
        x = self.activation(self.bn6(self.conv6(x)))
        x = self.avg_pool(x)
        x = self.conv7(x)

        x = torch.squeeze(x)  # For fully convolutional NN
        return x


class FusAtNet(nn.Module):
    def __init__(self, param, num_classes):
        super().__init__()
        self.n_classes = num_classes
        if len(param) == 2:
            self.hfe = Hyper_Feature_Extractor(param[0], 1024)
            self.spectral_am = Spectral_Attention_Module(param[0], 1024)
            self.spatial_am = Spatial_Attention_module(param[1], 1024)
            self.mfe = Modality_Feature_Extractor(1024 * 2 + param[0] + param[1], 1024)
            self.mam = Modality_Attention_Module(1024 * 2 + param[0] + param[1], 1024)
            self.cm = Classification_Module(1024, num_classes)
        elif len(param) == 3:
            self.hfe = Hyper_Feature_Extractor(param[0], 1024)
            self.spectral_am = Spectral_Attention_Module(param[0], 1024)
            self.spatial_am = Spatial_Attention_module(param[1], 1024)
            self.spat_am = Spatial_Attention_module(param[2], 1024)
            self.mfe = Modality_Feature_Extractor(1024 * 3 + param[0] + param[1] + param[2], 1024)
            self.mam = Modality_Attention_Module(1024 * 3 + param[0] + param[1] + param[2], 1024)
            self.cm = Classification_Module(1024, num_classes)

        self.init_wight()

    def init_wight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        if len(data) == 2:
            Fhs = self.hfe(data[0])
            Ms = self.spectral_am(data[0]) * Fhs
            Mt = self.spatial_am(data[1]) * Fhs
            Fm = self.mfe(torch.cat([data[0], data[1], Ms, Mt], 1))
            Am = self.mam(torch.cat([data[0], data[1], Ms, Mt], 1))
            Fss = Fm * Am
            out = self.cm(Fss)
        elif len(data) == 3:
            Fhs = self.hfe(data[0])
            Ms = self.spectral_am(data[0]) * Fhs
            Mt = self.spatial_am(data[1]) * Fhs
            Mt2 = self.spat_am(data[2]) * Fhs
            Fm = self.mfe(torch.cat([data[0], data[1], data[2], Ms, Mt, Mt2], 1))
            Am = self.mam(torch.cat([data[0], data[1], data[2], Ms, Mt, Mt2], 1))
            Fss = Fm * Am
            out = self.cm(Fss)
        return out.view(-1, self.n_classes)
