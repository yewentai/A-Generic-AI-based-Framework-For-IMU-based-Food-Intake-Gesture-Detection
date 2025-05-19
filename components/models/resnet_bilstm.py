import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Downsample(nn.Module):
    """
    Anti-alias downsampling layer using a fixed FIR filter.
    Factor >1 applies a convolution with a triangular/cubic/etc. kernel.
    """

    def __init__(self, channels, factor=2, order=1):
        super().__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        # Compute kernel (box filter convolved order times)
        box = np.ones(factor)
        kernel = box.copy()
        for _ in range(order - 1):
            kernel = np.convolve(kernel, box)
        kernel = kernel / kernel.sum()
        # Buffer the filter for depthwise conv
        filt = torch.tensor(kernel, dtype=torch.float32)
        filt = filt.unsqueeze(0).unsqueeze(0)
        filt = filt.repeat(channels, 1, 1)
        self.register_buffer("filt", filt)
        self.stride = factor
        self.pad = int((kernel.size - 1) / 2)

    def forward(self, x):
        # x: [B, C, L]
        return F.conv1d(x, self.filt, stride=self.stride, padding=self.pad, groups=x.size(1))


class ResBlock(nn.Module):
    """
    Basic 1D residual block: BN-ReLU-Conv -> BN-ReLU-Conv -> + identity
    """

    def __init__(self, channels, kernel_size=5, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False, padding_mode="circular")
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False, padding_mode="circular")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        return out + identity


class ResNetFeatureExtractor(nn.Module):
    """
    1D ResNetEncoder feature extractor composed of repeated Conv-ResBlocks-Downsample stages.
    """

    def __init__(self, in_channels=3, config=None):
        super().__init__()
        # Default config: list of tuples (out_ch, conv_k, n_resblocks, res_k, downfactor, downorder)
        if config is None:
            config = [
                (64, 5, 2, 5, 2, 2),
                (128, 5, 2, 5, 2, 2),
                (256, 5, 2, 5, 5, 1),
                (512, 5, 2, 5, 5, 1),
                (1024, 5, 0, 5, 3, 1),
            ]
        layers = []
        current_channels = in_channels
        for out_ch, conv_k, n_blocks, res_k, down_f, down_o in config:
            # initial conv
            pad = conv_k // 2
            layers.append(nn.Conv1d(current_channels, out_ch, conv_k, padding=pad, bias=False, padding_mode="circular"))
            # residual blocks
            for _ in range(n_blocks):
                layers.append(ResBlock(out_ch, kernel_size=res_k, padding=res_k // 2))
            # BN+ReLU
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            # downsample
            layers.append(Downsample(out_ch, factor=down_f, order=down_o))
            current_channels = out_ch
        self.layers = nn.Sequential(*layers)
        # final feature dim is current_channels
        self.out_features = current_channels

    def forward(self, x):
        # x: [B, C, L]
        return self.layers(x)


class ResNetCopy(nn.Module):
    """
    Wraps a ResNetFeatureExtractor and returns flattened features.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.extractor = ResNetFeatureExtractor(in_channels=in_channels)
        self.out_features = self.extractor.out_features

    def forward(self, x):
        # x: [B, C, L]
        feats = self.extractor(x)  # [B, out_ch, L']
        B = feats.size(0)
        return feats.view(B, -1)  # [B, out_ch * L']


class BiLSTMHead(nn.Module):
    """
    Sequence labeling head: Linear -> BiLSTM -> per-frame classifier.
    """

    def __init__(self, feature_dim, seq_length, num_classes, hidden_dim=128, lstm_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(feature_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.seq_length = seq_length

    def forward(self, x):
        # x: [B, feature_dim]
        B = x.size(0)
        h = self.proj(x)  # [B, hidden_dim]
        seq = h.unsqueeze(1).repeat(1, self.seq_length, 1)
        # [B, seq_length, hidden_dim]
        out, _ = self.lstm(seq)  # [B, seq_length, 2*hidden_dim]
        logits = self.classifier(out)  # [B, seq_length, num_classes]
        return logits


class ResNet_BiLSTM(nn.Module):
    """
    Combines ResNetEncoder encoder with BiLSTM head for end-to-end frame-level classification.
    """

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        feats = self.encoder(x)  # [B, feature_dim]
        logits = self.head(feats)  # [B, seq_len, num_classes]
        # swap to channel-first:
        return logits.permute(0, 2, 1).contiguous()  # [B, num_classes, seq_len]
