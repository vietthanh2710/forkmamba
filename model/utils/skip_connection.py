import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionBlock(nn.Module):
    def __init__(self, in_c, mid_c):
        super(FusionBlock, self).__init__()
        self.mid_c = mid_c or in_c
        self.in_c = in_c

        self.W_x = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_c)
        )

        self.W_skip = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_c)
        )

        # Spatial Attention
        self.psi = nn.Sequential(
            nn.Conv2d(mid_c, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=False)

        # Channel Fusion
        self.gap1 = nn.AdaptiveAvgPool2d((1,1))
        self.gap2 = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Conv2d(mid_c*2, mid_c, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace = False)
        )

        self.fcs = nn.ModuleList([])

        for i in range(2):
            self.fcs.append(
                nn.Conv2d(mid_c, in_c, kernel_size=1, stride=1)
            )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, f1, f2):

        assert f1.shape == f2.shape, "Input features must have the same shape"

        batch_size, _, H, W = f1.shape
        ori_feats = torch.cat((f1, f2), dim=1)
        ori_feats = ori_feats.view(batch_size, 2, self.in_c, H, W)

        f1_mid = self.W_x(f1)
        f2_mid = self.W_skip(f2)
        feats = torch.cat((f1_mid, f2_mid), dim=1)
        feats = feats.view(batch_size, 2, self.mid_c, feats.shape[2], feats.shape[3])
        feats_U = torch.sum(feats, dim=1)

        #Channel Fusion

        feats_S1 = self.gap1(feats_U)
        feats_S2 = self.gap2(feats_U)
        feats_S = torch.cat((feats_S1, feats_S2), dim = 1)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, 2, self.in_c, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(ori_feats*attention_vectors, dim=1)

        #Spatial Enhancement
        psi = self.relu(feats_U)
        psi = self.psi(psi)
        out = feats_V * psi

        return out