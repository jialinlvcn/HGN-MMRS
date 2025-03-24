import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, channels, kernels, paddings):
        super().__init__()
        if len(channels) != 3:
            raise RuntimeError("the lenghts of list channels and output_chs must be 3")
        self.layers_ = nn.Sequential()
        for i in range(len(channels) - 1):
            self.layers_.add_module(
                f"conv{i}",
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernels[i], padding=paddings[i], bias=True),
            )
            self.layers_.add_module(f"bn{i}", nn.BatchNorm2d(channels[i + 1]))
            self.layers_.add_module(f"relu{i}", nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.layers_(x)
        return out


class UnitGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers_ = nn.Sequential()
        for i in range(len(channels) - 2):
            self.layers_.add_module(
                f"conv{i}",
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            )
            self.layers_.add_module(f"bn{i}", nn.BatchNorm2d(channels[i + 1]))
            self.layers_.add_module(f"relu{i}", nn.PReLU(num_parameters=1))
        self.layers_.add_module("conv", nn.Conv2d(channels[-2], channels[-1], kernel_size=(3, 3), padding=(1, 1), bias=True))

    def forward(self, x):
        out = self.layers_(x)
        out = F.adaptive_avg_pool2d(F.sigmoid(out), 1)
        return out


class IndependentExtractor(nn.Module):
    def __init__(self, block_params, n_classes: int):
        super().__init__()
        self.blocks_list = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 'SAME' mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        for input_channels, kernels_list, paddings_list in zip(*block_params):
            self.blocks_list.append(Block(input_channels, kernels_list, paddings_list))

        last_channel = block_params[0][-1][-1]
        self.conv7 = nn.Conv2d(last_channel, n_classes, (1, 1))

    def forward(self, x):
        forward_features = []
        for i, block in enumerate(self.blocks_list):
            x = block(x)
            forward_features.append(x)
            if i == len(self.blocks_list) - 1:
                x = self.avg_pool(x)
            else:
                x = self.max_pool(x)

        x = self.conv7(x)
        return x, forward_features


class HGN(nn.Module):
    def __init__(self, params, n_classes):
        super().__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.parms = params
        self.n_classes = n_classes

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # 'SAME' mode
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.signal_extractor = nn.ModuleList()
        for param in self.parms:
            input_chs = [
                [param, filters[0], filters[1]],
                [filters[1], filters[2], filters[3]],
                [filters[3], filters[3], filters[2]],
            ]
            kernels = [[(3, 3), (1, 1)], [(3, 3), (1, 1)], [(1, 1), (1, 1)]]
            paddings = [[1, 0], [1, 0], [0, 0]]
            block_params = (input_chs, kernels, paddings)
            self.signal_extractor.append(IndependentExtractor(block_params, n_classes))

        self.low_gate = UnitGate([filters[1] * (len(params) + 1), int(filters[1] / 2), int(filters[1] / 4), len(params) + 1])
        self.mid_gate = UnitGate([filters[3] * (len(params) + 1), int(filters[3] / 2), int(filters[3] / 4), len(params) + 1])
        self.high_gate = UnitGate([filters[2] * (len(params) + 1), int(filters[2] / 2), int(filters[2] / 4), len(params) + 1])
        self.decision_gate = UnitGate([n_classes * (len(params) + 1), int(filters[2] / 2), int(filters[2] / 4), len(params) + 1])

        self.fusion_block1 = Block([sum(params), filters[0], filters[1]], [(3, 3), (1, 1)], [1, 0])
        self.fusion_block2 = Block([filters[1] * (len(params) + 1), filters[2], filters[3]], [(3, 3), (1, 1)], [1, 0])
        self.fusion_block3 = Block([filters[3] * (len(params) + 1), filters[3], filters[2]], [(1, 1), (1, 1)], [0, 0])

        self.conv = nn.Conv2d(filters[2] * (len(params) + 1), n_classes, (1, 1))

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data: list):
        logits, features = [], []
        for _, (x, extractor) in enumerate(zip(data, self.signal_extractor)):
            x, forward_features = extractor(x)
            logits.append(x)
            features.append(forward_features)
        out = self.fusion_forward(data, features)
        logits.append(out)
        x_fusion = torch.cat(logits, 1)

        gate = self.decision_gate(x_fusion)
        fusion_out = (gate.unsqueeze(-1) * torch.stack(logits, 1)).sum(dim=1)
        return fusion_out.view(-1, self.n_classes)

    def fusion_forward(self, data, features):
        early_feats = data
        x_fusion = torch.cat(early_feats, 1)
        fusion_out = self.fusion_block1(x_fusion)

        low_feats = [f[0] for f in features]
        low_feats.append(fusion_out)
        fusion_out = self.gate_aggregation(low_feats, gate_name="low")
        fusion_out = self.max_pool(fusion_out)
        fusion_out = self.fusion_block2(fusion_out)

        mid_feats = [f[1] for f in features]
        mid_feats.append(fusion_out)
        fusion_out = self.gate_aggregation(mid_feats, gate_name="mid")
        fusion_out = self.max_pool(fusion_out)
        fusion_out = self.fusion_block3(fusion_out)

        high_feats = [f[2] for f in features]
        high_feats.append(fusion_out)
        fusion_out = self.gate_aggregation(high_feats, gate_name="high")
        fusion_out = self.avg_pool(fusion_out)
        fusion_out = self.conv(fusion_out)

        return fusion_out

    def gate_aggregation(self, features:list[torch.tensor], gate_name="low"):
        if gate_name == "low":
            gate_weight = self.low_gate(torch.cat(features, dim=1))
        elif gate_name == "mid":
            gate_weight = self.mid_gate(torch.cat(features, dim=1))
        elif gate_name == "high":
            gate_weight = self.high_gate(torch.cat(features, dim=1))
        elif gate_name == "decision":
            gate_weight = self.decision_gate(torch.cat(features, dim=1))

        fusion_out = gate_weight.unsqueeze(-1) * torch.stack(features, dim=1)
        fusion_out = torch.cat([fusion_out[:, i] for i in range(fusion_out.shape[1])], 1)
        return fusion_out



if __name__ == "__main__":
    for rate in [10, 20, 50, 100]:
        for dataset in ["Houston2013", "Augsburg"]:
            save_path = f"checkpoint/{dataset}-IndNet-{rate}.pt"
            model_dict = torch.load(save_path, weights_only=True)

            new_state_dict = {}
            for key, value in model_dict.items():
                if key.startswith("gate1"):
                    new_key = key.replace("gate1", "low_gate")
                    new_state_dict[new_key] = value
                elif key.startswith("gate2"):
                    new_key = key.replace("gate2", "mid_gate")
                    new_state_dict[new_key] = value
                elif key.startswith("gate3"):
                    new_key = key.replace("gate3", "high_gate")
                    new_state_dict[new_key] = value
                elif key.startswith("gate4"):
                    new_key = key.replace("gate4", "decision_gate")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            if dataset == "Houston2013":
                model = HGN([144, 1], 15)
                input1 = torch.autograd.Variable(torch.randn(64, 144, 7, 7))
                input2 = torch.autograd.Variable(torch.randn(64, 1, 7, 7))
                logit = model([input1, input2])
                print(logit.shape)
            elif dataset == "Augsburg":
                model = HGN([180, 4, 1], 7)
                input1 = torch.autograd.Variable(torch.randn(64, 180, 7, 7))
                input2 = torch.autograd.Variable(torch.randn(64, 4, 7, 7))
                input3 = torch.autograd.Variable(torch.randn(64, 1, 7, 7))
                logit = model([input1, input2, input3])
                print(logit.shape)
            unex, ex = model.load_state_dict(new_state_dict, strict=False)
            torch.save(model.state_dict(), save_path.replace("IndNet", "HGN"))

