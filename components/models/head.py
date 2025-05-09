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
