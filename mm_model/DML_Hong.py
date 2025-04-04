"""
Re-implementation for paper "More Diverse Means Better:
Multimodal Deep Learning Meets Remote-Sensing Imagery Classification"
The official tensorflow implementation is in https://github.com/danfenghong/IEEE_TGRS_MDL-RS
"""

import torch
import torch.nn as nn


class Early_fusion_CNN(nn.Module):
    # Re-implemented early_fusion_CNN for paper "More Diverse Means Better:
    # Multimodal Deep Learning Meets Remote-Sensing Imagery Classiﬁcation"
    # But not use APs to convert 1-band LiDAR data to 21-band.
    def __init__(self, param, n_classes):
        super().__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.n_classes = n_classes

        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 'SAME' mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # For concatenated image x (7×7×d)
        self.conv1 = nn.Conv2d(sum(param), filters[0], kernel_size=3, padding=1, bias=True)
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
        x = self.activation(self.bn1(self.conv1(torch.cat(data, 1))))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.max_pool(x)
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.max_pool(x)

        x = self.activation(self.bn5(self.conv5(x)))
        x = self.activation(self.bn6(self.conv6(x)))
        x = self.avg_pool(x)
        x = self.conv7(x)

        x = x.view(-1, self.n_classes)  # For fully convolutional NN
        return x


class Middle_fusion_CNN(nn.Module):
    # Re-implemented middle_fusion_CNN for paper "More Diverse Means Better:
    # Multimodal Deep Learning Meets Remote-Sensing Imagery Classiﬁcation"
    # But not use APs to convert 1-band LiDAR data to 21-band.
    def __init__(self, param, n_classes):
        super().__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.n_classes = n_classes

        self.data_num = len(param)

        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 'SAME' mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.conv4, self.bn4 = (
            nn.ModuleList() for _ in range(8)
        )
        for p in param:
            # For image a (7×7×d)
            self.conv1.append(nn.Conv2d(p, filters[0], kernel_size=3, padding=1, bias=True))
            self.bn1.append(nn.BatchNorm2d(filters[0]))
            self.conv2.append(nn.Conv2d(filters[0], filters[1], (1, 1)))
            self.bn2.append(nn.BatchNorm2d(filters[1]))
            # Max pooling ('SAME' mode) --> 4×4×32
            self.conv3.append(nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True))
            self.bn3.append(nn.BatchNorm2d(filters[2]))
            self.conv4.append(nn.Conv2d(filters[2], filters[3], (1, 1)))
            self.bn4.append(nn.BatchNorm2d(filters[3]))

        self.conv5 = nn.Conv2d(filters[3] * len(param), filters[3], (1, 1))
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
        x_list = []
        for i, x in enumerate(data):
            # for image a
            x = self.activation(self.bn1[i](self.conv1[i](x)))
            x = self.activation(self.bn2[i](self.conv2[i](x)))
            x = self.max_pool(x)
            x = self.activation(self.bn3[i](self.conv3[i](x)))
            x = self.activation(self.bn4[i](self.conv4[i](x)))
            x = self.max_pool(x)
            x_list.append(x)

        x = self.activation(self.bn5(self.conv5(torch.cat(x_list, 1))))
        x = self.activation(self.bn6(self.conv6(x)))
        x = self.avg_pool(x)
        x = self.conv7(x)

        x = x.view(-1, self.n_classes)  # For fully convolutional NN
        return x


class Late_fusion_CNN(nn.Module):
    # Re-implemented late_fusion_CNN for paper "More Diverse Means Better:
    # Multimodal Deep Learning Meets Remote-Sensing Imagery Classiﬁcation"
    # But not use APs to convert 1-band LiDAR data to 21-band.
    def __init__(self, param, n_classes):
        super().__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.n_classes = n_classes

        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 'SAME' mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3 = (nn.ModuleList() for _ in range(6))
        self.conv4, self.bn4, self.conv5, self.bn5, self.conv6, self.bn6 = (nn.ModuleList() for _ in range(6))
        for p in param:
            # For image a (7×7×d)
            self.conv1.append(nn.Conv2d(p, filters[0], kernel_size=3, padding=1, bias=True))
            self.bn1.append(nn.BatchNorm2d(filters[0]))
            self.conv2.append(nn.Conv2d(filters[0], filters[1], (1, 1)))
            self.bn2.append(nn.BatchNorm2d(filters[1]))
            # Max pooling ('SAME' mode) --> 4×4×32
            self.conv3.append(nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True))
            self.bn3.append(nn.BatchNorm2d(filters[2]))
            self.conv4.append(nn.Conv2d(filters[2], filters[3], (1, 1)))
            self.bn4.append(nn.BatchNorm2d(filters[3]))
            # Max pooling ('SAME' mode) --> 2×2×128
            self.conv5.append(nn.Conv2d(filters[3], filters[3], (1, 1)))
            self.bn5.append(nn.BatchNorm2d(filters[3]))
            self.conv6.append(nn.Conv2d(filters[3], filters[2], (1, 1)))
            self.bn6.append(nn.BatchNorm2d(filters[2]))
            # Average Pooling --> 1×1×64    # Use AdaptiveAvgPool2d() for more robust

        self.conv7 = nn.Conv2d(filters[2] * len(param), n_classes, (1, 1))

        # weight_init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x_list = []
        for i, x in enumerate(data):
            # for image a
            x = self.activation(self.bn1[i](self.conv1[i](x)))
            x = self.activation(self.bn2[i](self.conv2[i](x)))
            x = self.max_pool(x)
            x = self.activation(self.bn3[i](self.conv3[i](x)))
            x = self.activation(self.bn4[i](self.conv4[i](x)))
            x = self.max_pool(x)
            x = self.activation(self.bn5[i](self.conv5[i](x)))
            x = self.activation(self.bn6[i](self.conv6[i](x)))
            x = self.avg_pool(x)
            x_list.append(x)
        x = self.conv7(torch.cat(x_list, 1))
        x = x.view(-1, self.n_classes)  # For fully convolutional NN
        return x


class Decision_fusion_CNN(nn.Module):
    # Re-implemented late_fusion_CNN for paper "More Diverse Means Better:
    # Multimodal Deep Learning Meets Remote-Sensing Imagery Classiﬁcation"
    # But not use APs to convert 1-band LiDAR data to 21-band.
    def __init__(self, param, n_classes):
        super().__init__()
        self.n_classes = n_classes
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.n_classes = n_classes

        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 'SAME' mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3 = (nn.ModuleList() for _ in range(6))
        self.conv4, self.bn4, self.conv5, self.bn5, self.conv6, self.bn6 = (nn.ModuleList() for _ in range(6))
        self.conv7 = nn.ModuleList()
        for p in param:
            # For image a (7×7×d)
            self.conv1.append(nn.Conv2d(p, filters[0], kernel_size=3, padding=1, bias=True))
            self.bn1.append(nn.BatchNorm2d(filters[0]))
            self.conv2.append(nn.Conv2d(filters[0], filters[1], (1, 1)))
            self.bn2.append(nn.BatchNorm2d(filters[1]))
            # Max pooling ('SAME' mode) --> 4×4×32
            self.conv3.append(nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True))
            self.bn3.append(nn.BatchNorm2d(filters[2]))
            self.conv4.append(nn.Conv2d(filters[2], filters[3], (1, 1)))
            self.bn4.append(nn.BatchNorm2d(filters[3]))
            # Max pooling ('SAME' mode) --> 2×2×128
            self.conv5.append(nn.Conv2d(filters[3], filters[3], (1, 1)))
            self.bn5.append(nn.BatchNorm2d(filters[3]))
            self.conv6.append(nn.Conv2d(filters[3], filters[2], (1, 1)))
            self.bn6.append(nn.BatchNorm2d(filters[2]))
            # Average Pooling --> 1×1×64    # Use AdaptiveAvgPool2d() for more robust

            self.conv7.append(nn.Conv2d(filters[2], n_classes, (1, 1)))

    def forward(self, data):
        x_list = []
        for i, x in enumerate(data):
            # for image a
            x = self.activation(self.bn1[i](self.conv1[i](x)))
            x = self.activation(self.bn2[i](self.conv2[i](x)))
            x = self.max_pool(x)
            x = self.activation(self.bn3[i](self.conv3[i](x)))
            x = self.activation(self.bn4[i](self.conv4[i](x)))
            x = self.max_pool(x)
            x = self.activation(self.bn5[i](self.conv5[i](x)))
            x = self.activation(self.bn6[i](self.conv6[i](x)))
            x = self.avg_pool(x)
            x = self.conv7[i](x)
            x_list.append(x)
        x = sum(x_list) / len(x_list)
        x = x.view(-1, self.n_classes) # For fully convolutional NN
        return x


class Cross_fusion_CNN(nn.Module):
    # But not use APs to convert 1-band LiDAR data to 21-band.
    def __init__(self, param, n_classes):
        super().__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.n_classes = n_classes

        input_channels, input_channels2 = param
        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 'SAME' mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # For image a (7×7×d)
        self.conv1_a = nn.Conv2d(input_channels, filters[0], kernel_size=3, padding=1, bias=True)
        self.bn1_a = nn.BatchNorm2d(filters[0])
        self.conv2_a = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2_a = nn.BatchNorm2d(filters[1])
        # Max pooling ('SAME' mode) --> 4×4×32
        self.conv3_a = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
        self.bn3_a = nn.BatchNorm2d(filters[2])
        self.conv4_a = nn.Conv2d(filters[2], filters[3], (1, 1))
        self.bn4_a = nn.BatchNorm2d(filters[3])
        # Max pooling ('SAME' mode) --> 2×2×128

        # For image b (7×7×d)
        self.conv1_b = nn.Conv2d(input_channels2, filters[0], kernel_size=3, padding=1, bias=True)
        self.bn1_b = nn.BatchNorm2d(filters[0])
        self.conv2_b = nn.Conv2d(filters[0], filters[1], (1, 1))
        self.bn2_b = nn.BatchNorm2d(filters[1])
        # Max pooling ('SAME' mode) --> 4×4×32
        self.conv3_b = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
        self.bn3_b = nn.BatchNorm2d(filters[2])
        self.conv4_b = nn.Conv2d(filters[2], filters[3], (1, 1))
        self.bn4_b = nn.BatchNorm2d(filters[3])
        # Max pooling ('SAME' mode) --> 2×2×128

        self.conv5 = nn.Conv2d(filters[3] + filters[3], filters[3], (1, 1))
        self.bn5 = nn.BatchNorm2d(filters[3])
        self.conv6 = nn.Conv2d(filters[3], filters[2], (1, 1))
        self.bn6 = nn.BatchNorm2d(filters[2])
        # Average Pooling --> 1×1×64    # Use AdaptiveAvgPool2d() for more robust

        self.conv7 = nn.Conv2d(filters[2], n_classes, (1, 1))

    def forward(self, data):
        x_all = []
        for i, x in enumerate(data):
            x_all.append(x)
        x1, x2 = x_all
        # for image a
        x1 = self.activation(self.bn1_a(self.conv1_a(x1)))
        x1 = self.activation(self.bn2_a(self.conv2_a(x1)))
        x1 = self.max_pool(x1)
        x1 = self.activation(self.bn3_a(self.conv3_a(x1)))

        # for image b
        x2 = self.activation(self.bn1_b(self.conv1_b(x2)))
        x2 = self.activation(self.bn2_b(self.conv2_b(x2)))
        x2 = self.max_pool(x2)
        x2 = self.activation(self.bn3_b(self.conv3_b(x2)))

        x11 = self.activation(self.bn4_a(self.conv4_a(x1)))
        x11 = self.max_pool(x11)
        x22 = self.activation(self.bn4_b(self.conv4_b(x2)))
        x22 = self.max_pool(x22)
        x12 = self.activation(self.bn4_b(self.conv4_b(x1)))
        x12 = self.max_pool(x12)
        x21 = self.activation(self.bn4_a(self.conv4_a(x2)))
        x21 = self.max_pool(x21)

        joint_encoder_layer1 = torch.cat([x11 + x21, x22 + x12], 1)
        joint_encoder_layer2 = torch.cat([x11, x12], 1)
        joint_encoder_layer3 = torch.cat([x22, x21], 1)

        fusion1 = self.activation(self.bn5(self.conv5(joint_encoder_layer1)))
        fusion1 = self.activation(self.bn6(self.conv6(fusion1)))
        fusion1 = self.avg_pool(fusion1)
        fusion1 = self.conv7(fusion1)

        fusion2 = self.activation(self.bn5(self.conv5(joint_encoder_layer2)))
        fusion2 = self.activation(self.bn6(self.conv6(fusion2)))
        fusion2 = self.avg_pool(fusion2)
        fusion2 = self.conv7(fusion2)

        fusion3 = self.activation(self.bn5(self.conv5(joint_encoder_layer3)))
        fusion3 = self.activation(self.bn6(self.conv6(fusion3)))
        fusion3 = self.avg_pool(fusion3)
        fusion3 = self.conv7(fusion3)

        fusion1 = fusion1.view(-1, self.n_classes)  # For fully convolutional NN
        fusion2 = fusion2.view(-1, self.n_classes)  # For fully convolutional NN
        fusion3 = fusion3.view(-1, self.n_classes)  # For fully convolutional NN
        return sum([fusion1, fusion2, fusion3]) / 3
