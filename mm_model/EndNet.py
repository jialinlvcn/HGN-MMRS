"""
Re-implementation for paper "Deep Encoder-Decoder Networks for Classification of Hyperspectral and LiDAR Data"
The official tensorflow implementation is in https://github.com/danfenghong/IEEE_GRSL_EndNet
"""

import torch
import torch.nn as nn


class EndNet(nn.Module):
    # Re-implemented middle_fusion_CNN for paper "More Diverse Means Better:
    # Multimodal Deep Learning Meets Remote-Sensing Imagery Classiﬁcation"
    # But not use APs to convert 1-band LiDAR data to 21-band.
    def __init__(self, param, n_classes):
        super().__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        self.n_classes = n_classes

        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.encoder_fc1, self.encoder_bn1, self.encoder_fc2, self.encoder_bn2 = (
            nn.ModuleList() for _ in range(4)
        )
        self.encoder_fc3, self.encoder_bn3, self.encoder_fc4, self.encoder_bn4 = (
            nn.ModuleList() for _ in range(4)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 'SAME' mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        for p in param:
            # Encoder
            # For image a (1×1×d)
            self.encoder_fc1.append(
                nn.Conv2d(p, filters[0], kernel_size=3, padding=1, bias=True)
            )
            self.encoder_bn1.append(nn.BatchNorm2d(filters[0]))
            self.encoder_fc2.append(nn.Conv2d(filters[0], filters[1], (1, 1)))
            self.encoder_bn2.append(nn.BatchNorm2d(filters[1]))
            self.encoder_fc3.append(
                nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1, bias=True)
            )
            self.encoder_bn3.append(nn.BatchNorm2d(filters[2]))
            self.encoder_fc4.append(nn.Conv2d(filters[2], filters[3], (1, 1)))
            self.encoder_bn4.append(nn.BatchNorm2d(filters[3]))
        self.joint_encoder_fc5 = nn.Conv2d(filters[3] * len(param), filters[3], (1, 1))
        self.joint_encoder_bn5 = nn.BatchNorm2d(filters[3])
        self.joint_encoder_fc6 = nn.Conv2d(filters[3], filters[2], (1, 1))
        self.joint_encoder_bn6 = nn.BatchNorm2d(filters[2])
        self.joint_encoder_fc7 = nn.Conv2d(filters[2], n_classes, (1, 1))
        self.joint_encoder_bn7 = nn.BatchNorm2d(n_classes)

        self.decoder_fc1, self.decoder_fc2 = (nn.ModuleList() for _ in range(2))
        self.decoder_fc3, self.decoder_fc4 = (nn.ModuleList() for _ in range(2))

        for p in param:
            self.decoder_fc1.append(
                nn.ConvTranspose2d(filters[3], filters[2], kernel_size=(1, 1))
            )
            self.decoder_fc2.append(
                nn.ConvTranspose2d(filters[2], filters[1], kernel_size=(1, 1))
            )
            self.decoder_fc3.append(
                nn.ConvTranspose2d(filters[1], filters[0], kernel_size=(1, 1))
            )
            self.decoder_fc4.append(
                nn.ConvTranspose2d(filters[0], p, kernel_size=(1, 1))
            )

        self.init_wight()

    def init_wight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x_list = []
        for i, x in enumerate(data):
            x = self.activation(self.encoder_bn1[i](self.encoder_fc1[i](x)))
            x = self.activation(self.encoder_bn2[i](self.encoder_fc2[i](x)))
            # x = self.max_pool(x)
            x = self.activation(self.encoder_bn3[i](self.encoder_fc3[i](x)))
            x = self.activation(self.encoder_bn4[i](self.encoder_fc4[i](x)))
            # x = self.max_pool(x)
            x_list.append(x)

        joint_x = torch.cat(x_list, 1)
        joint_x = self.activation(
            self.joint_encoder_bn5(self.joint_encoder_fc5(joint_x))
        )
        out = self.activation(self.joint_encoder_bn6(self.joint_encoder_fc6(joint_x)))
        out = self.avg_pool(out)
        out = self.joint_encoder_fc7(out)
        out = out.view(-1, self.n_classes)
        xx_list = []
        for i in range(len(self.decoder_fc1)):
            x1 = self.sigmoid(self.decoder_fc1[i](joint_x))
            x1 = self.sigmoid(self.decoder_fc2[i](x1))
            x1 = self.sigmoid(self.decoder_fc3[i](x1))
            x1 = self.sigmoid(self.decoder_fc4[i](x1))
            xx_list.append(x1)

        return (out, xx_list, data)
