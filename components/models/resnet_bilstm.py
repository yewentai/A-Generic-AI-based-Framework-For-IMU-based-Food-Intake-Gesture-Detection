#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
ResNet-BiLSTM Model Definition
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-24
Description : This module implements the ResNet-BiLSTM architecture for IMU-based
              segmentation tasks. It features a 1D ResNet encoder for feature extraction,
              optional downsampling, and a BiLSTM sequence labeling head for frame-level
              classification. The model is designed for end-to-end training and supports
              flexible integration with different encoder and head configurations.
===============================================================================
"""

import torch.nn as nn


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
