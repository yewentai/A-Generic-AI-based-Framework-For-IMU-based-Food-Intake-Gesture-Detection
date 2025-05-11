import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        """
        A simple MLP head with one hidden layer.

        Parameters:
            feature_dim (int): Dimensionality of the input feature vector.
            num_classes (int): Number of target classes.
        """
        super(MLPClassifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        logits = self.fc(x)
        return logits


class DeepMLPClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=3):
        """
        A deeper MLP classifier head.

        Parameters:
            input_dim (int): Dimensionality of the input feature vector.
            num_classes (int): Number of target classes.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class BNMLPClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=3):
        """
        MLP classifier head with batch normalization.

        Parameters:
            input_dim (int): Dimensionality of the input feature vector.
            num_classes (int): Number of target classes.
        """
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.net(x)


class ResNetMLP(nn.Module):
    def __init__(self, encoder, classifier):
        """
        Parameters:
            encoder (nn.Module): Pre-trained ResNet encoder.
            classifier (nn.Module): MLP classifier head.
        """
        super(ResNetMLP, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


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
