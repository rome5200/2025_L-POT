import torch
import torch.nn as nn
import torch.nn.functional as F


class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, spiral_len: int):
        super().__init__()
        self.spiral_len = spiral_len
        self.conv = nn.Conv1d(in_channels * spiral_len, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, spiral_idx):
        B, C, V = x.size()
        L = spiral_idx.size(1)
        idx = spiral_idx.unsqueeze(0).unsqueeze(0).expand(B, C, V, L)
        x_expand = x.unsqueeze(-1).expand(B, C, V, L)
        neigh = torch.gather(x_expand, 2, idx)
        neigh = neigh.reshape(B, C * L, V)
        out = self.conv(neigh)
        out = self.bn(out)
        return self.relu(out)


class SpiralBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spiral_index, dropout=0.0):
        super().__init__()
        spiral_len = spiral_index.shape[1]
        self.spiral = SpiralConv(in_channels, out_channels, spiral_len)
        self.use_res = (in_channels == out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, spiral_idx):
        out = self.spiral(x, spiral_idx)
        out = self.dropout(out)
        if self.use_res:
            return out + x
        else:
            return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class ImprovedSpiralNet(nn.Module):
    """개선된 SpiralNet with Multi-scale features"""
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_blocks, spiral_indices=None, dropout=0.0, use_se=True):
        super().__init__()
        self.spiral_indices = spiral_indices
        self.in_channels = in_channels
        num_blocks = len(spiral_indices)
        self.use_se = use_se

        self.encoder = nn.ModuleList()
        for i in range(num_blocks):
            in_c = in_channels if i == 0 else hidden_channels
            out_c = hidden_channels
            spiral_index = spiral_indices[i]
            self.encoder.append(
                SpiralBlock(in_c, out_c, spiral_index, dropout=dropout)
            )

        if use_se:
            self.se_blocks = nn.ModuleList([
                SEBlock(hidden_channels) for _ in range(num_blocks)
            ])
        else:
            self.se_blocks = [nn.Identity()] * num_blocks

        self.skip_conns = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(num_blocks - 1):
            self.skip_conns.append(nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=1))
            spiral_index_dec = spiral_indices[num_blocks - 2 - i]
            self.decoder.append(
                SpiralBlock(hidden_channels, hidden_channels, spiral_index_dec, dropout=dropout)
            )

        # Multi-scale classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels // 2, hidden_channels // 4, kernel_size=1),
            nn.BatchNorm1d(hidden_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels // 4, out_channels, kernel_size=1)
        )

    def forward(self, verts, indices):
        x = verts.permute(0, 2, 1)
        features = []

        for block, idx, se in zip(self.encoder, self.spiral_indices, self.se_blocks):
            x = block(x, idx)
            x = se(x)
            features.append(x)

        for skip_conv, dec_block, idx, feat in zip(
            self.skip_conns,
            self.decoder,
            reversed(self.spiral_indices[:-1]),
            reversed(features[:-1])
        ):
            x = torch.cat([x, feat], dim=1)
            x = skip_conv(x)
            x = dec_block(x, idx)

        out = self.classifier(x)
        return out.squeeze(1)


class SimplePointTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_hidden=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        h2 = self.norm2(x)
        x = x + self.ff(h2)
        return x


class HybridSpiralNetPointNetTransformer(nn.Module):
    def __init__(self, spiralnet, in_channels, hidden_dim=256, out_dim=1, dropout=0.3,
                 transformer_heads=4, transformer_blocks=1):
        super().__init__()
        self.spiralnet = spiralnet

        # PointNet branch
        self.pointnet_mlp1 = nn.Linear(in_channels, hidden_dim)
        self.pointnet_bn1 = nn.BatchNorm1d(hidden_dim)
        self.pointnet_mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.pointnet_bn2 = nn.BatchNorm1d(hidden_dim)
        self.pointnet_dropout = nn.Dropout(dropout)

        # Transformer branch
        self.transformer_blocks = nn.ModuleList([
            SimplePointTransformerBlock(hidden_dim, num_heads=transformer_heads, dropout=dropout)
            for _ in range(transformer_blocks)
        ])

        self.pointnet_head = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x, spiral_idx):
        spiral_out = self.spiralnet(x, spiral_idx)

        B, V, C = x.size()
        
        # PointNet branch
        feat1 = self.pointnet_mlp1(x)
        feat1 = feat1.view(B * V, -1)
        feat1 = self.pointnet_bn1(feat1)
        feat1 = F.relu(feat1)
        feat1 = feat1.view(B, V, -1)

        feat2 = self.pointnet_mlp2(feat1)
        feat2 = feat2.view(B * V, -1)
        feat2 = self.pointnet_bn2(feat2)
        feat2 = F.relu(feat2)
        feat2 = feat2.view(B, V, -1)

        point_feat = self.pointnet_dropout(feat2)

        # Transformer Attention
        for block in self.transformer_blocks:
            point_feat = block(point_feat)

        # Global pooling
        global_feat, _ = torch.max(point_feat, dim=1, keepdim=True)
        global_feat = global_feat.expand(-1, V, -1)

        # Concat
        pn_feat = torch.cat([point_feat, global_feat], dim=2)
        pointnet_out = self.pointnet_head(pn_feat).squeeze(-1)

        # Ensemble
        out = (spiral_out + pointnet_out) / 2
        return out