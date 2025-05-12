import torch.nn as nn


class SequenceLabelingHead(nn.Module):
    """
    A more flexible sequence labeling head that upsamples features to match the original
    sequence length, then applies per-frame classification.

    This approach uses a series of bi-directional LSTM layers to capture temporal dependencies,
    followed by a linear projection to the number of classes per frame.
    """

    def __init__(self, feature_dim, seq_length, num_classes, hidden_dim=128):
        super(SequenceLabelingHead, self).__init__()
        self.feature_dim = feature_dim
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Project features to a sequence of hidden states
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)

        # Bi-directional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # Final projection for each time step
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        # x shape: [B, feature_dim]
        batch_size = x.shape[0]

        # Project to hidden dimension
        hidden = self.feature_projection(x)

        # Repeat the hidden state seq_length times
        # This broadcasts the global features to each time step
        sequence = hidden.unsqueeze(1).repeat(1, self.seq_length, 1)
        # sequence shape: [B, seq_length, hidden_dim]

        # Apply LSTM
        sequence, _ = self.lstm(sequence)
        # sequence shape: [B, seq_length, 2*hidden_dim]

        # Apply classifier to each time step
        logits = self.classifier(sequence)
        # logits shape: [B, seq_length, num_classes]

        return logits


class ResNetSeqLabeler(nn.Module):
    """
    Combines a ResNet encoder with a sequence labeling head for frame-level prediction.
    """

    def __init__(self, encoder, seq_labeler):
        super(ResNetSeqLabeler, self).__init__()
        self.encoder = encoder
        self.seq_labeler = seq_labeler

    def forward(self, x):
        # x shape: [B, channels, seq_len]
        features = self.encoder(x)
        # features shape: [B, feature_dim]

        logits = self.seq_labeler(features)
        # logits shape: [B, seq_len, num_classes]

        return logits
