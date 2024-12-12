import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


# Chomp1d removes excess padding from the output of Conv1d layers
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        """
        Removes the last `chomp_size` elements along the time dimension
        to ensure the output size matches the expected size.
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Slice to remove extra padding added during convolution
        return x[:, :, : -self.chomp_size].contiguous()


# TemporalBlock implements a single block of the Temporal Convolutional Network
class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        """
        Defines a temporal block with two dilated convolution layers, ReLU activations,
        and residual connections.

        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Size of the convolution kernel
            stride: Convolution stride
            dilation: Dilation factor for convolution
            padding: Amount of padding added to input
            dropout: Dropout rate
        """
        super(TemporalBlock, self).__init__()

        # First convolution layer with weight normalization
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)  # Remove extra padding
        self.relu1 = nn.ReLU()  # First ReLU activation
        self.dropout1 = nn.Dropout(dropout)  # Apply dropout for regularization

        # Second convolution layer with weight normalization
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)  # Remove extra padding
        self.relu2 = nn.ReLU()  # Second ReLU activation
        self.dropout2 = nn.Dropout(dropout)  # Apply dropout for regularization

        # Sequential layer combining convolution, chomp, activation, and dropout
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        # Downsample the input if necessary for residual connection
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()  # Final ReLU activation after combining residual
        self.init_weights()  # Initialize weights

    def init_weights(self):
        """
        Initialize weights of the convolution layers with a normal distribution.
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Forward pass through the temporal block.

        Args:
            x: Input tensor of shape [batch_size, num_channels, seq_length]
        Returns:
            Output tensor after applying the block and residual connection.
        """
        out = self.net(x)
        # Add residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# TemporalConvNet is a multi-layer temporal convolutional network
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Defines a Temporal Convolutional Network (TCN) with multiple levels of
        TemporalBlock.

        Args:
            num_inputs: Number of input channels
            num_channels: List of output channels for each layer
            kernel_size: Size of the convolution kernel
            dropout: Dropout rate
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)  # Number of temporal blocks

        for i in range(num_levels):
            dilation_size = 2**i  # Exponentially increasing dilation factor
            in_channels = (
                num_inputs if i == 0 else num_channels[i - 1]
            )  # Input channels for this block
            out_channels = num_channels[i]  # Output channels for this block

            # Add a temporal block to the layers
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        # Combine all blocks into a sequential network
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the TCN.

        Args:
            x: Input tensor of shape [batch_size, num_inputs, seq_length]
        Returns:
            Output tensor after passing through the TCN.
        """
        return self.network(x)


class SimpleTCN(nn.Module):
    def __init__(self, input_channels, output_channels, num_channels, kernel_size=3):
        super(SimpleTCN, self).__init__()

        layers = []
        num_levels = len(num_channels)

        # Input layer
        layers.append(
            nn.Conv1d(
                input_channels, num_channels[0], kernel_size=kernel_size, padding="same"
            )
        )
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(num_channels[0]))

        # Hidden layers with increasing dilation
        for i in range(num_levels - 1):
            dilation_size = 2**i
            in_channels = num_channels[i]
            out_channels = num_channels[i + 1]

            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding="same",
                        dilation=dilation_size,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels),
                    nn.Dropout(0.2),
                ]
            )

        # Output layer
        layers.append(
            nn.Conv1d(num_channels[-1], output_channels, kernel_size=1, padding="same")
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiStageTCN(nn.Module):
    def __init__(
        self, input_channels, num_channels_per_stage, kernel_size=3, dropout=0.2
    ):
        """
        A multi-stage Temporal Convolutional Network (TCN).

        Args:
            input_channels: Number of input channels to the first stage.
            num_channels_per_stage: List of lists, where each sublist contains the number of channels
                                    for each layer in a specific stage.
            kernel_size: Size of the convolution kernel for all stages.
            dropout: Dropout rate for all stages.
        """
        super(MultiStageTCN, self).__init__()

        stages = []
        for stage_idx, num_channels in enumerate(num_channels_per_stage):
            # For the first stage, input channels come from the dataset
            # For subsequent stages, input channels come from the last stage's output
            in_channels = (
                input_channels
                if stage_idx == 0
                else num_channels_per_stage[stage_idx - 1][-1]
            )

            # Create a TemporalConvNet for the current stage
            stages.append(
                TemporalConvNet(
                    in_channels, num_channels, kernel_size=kernel_size, dropout=dropout
                )
            )

        # Combine all stages into a sequential module
        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        """
        Forward pass through the Multi-Stage TCN.

        Args:
            x: Input tensor of shape [batch_size, num_channels, seq_length].
        Returns:
            Output tensor after passing through all stages.
        """
        return self.stages(x)


# Define function to create random input data
def generate_input_data(batch_size, input_channels, seq_length):
    return torch.rand(batch_size, input_channels, seq_length)


# Define test function to compare SimpleTCN and TemporalConvNet
def test_models():

    # Configuration
    batch_size = 8
    input_channels = 3
    seq_length = 10
    kernel_size = 3
    dropout = 0.2

    # Define the number of channels for each layer in both models
    num_channels = [16, 32, 64]

    # Define the output channels for SimpleTCN (for example purposes, making it equal to the last num_channels to simplify comparison)
    output_channels = 64

    # Generate synthetic input data
    x = generate_input_data(batch_size, input_channels, seq_length)

    # Initialize both models
    simple_tcn = SimpleTCN(
        input_channels, output_channels, num_channels, kernel_size=kernel_size
    )
    temporal_conv_net = TemporalConvNet(
        input_channels, num_channels, kernel_size=kernel_size, dropout=dropout
    )

    # Forward pass through the models
    simple_tcn_output = simple_tcn(x)
    temporal_conv_net_output = temporal_conv_net(x)

    # Print results
    print("SimpleTCN output shape:", simple_tcn_output.shape)
    print("TemporalConvNet output shape:", temporal_conv_net_output.shape)

    # Ensure the dimensions agree
    assert (
        simple_tcn_output.shape == temporal_conv_net_output.shape
    ), "Output shapes do not match!"


# Test the Multi-Stage TCN
def test_multi_stage_tcn():
    batch_size = 8
    input_channels = 3
    seq_length = 50
    kernel_size = 3
    dropout = 0.2

    # Define the number of channels for each stage
    num_channels_per_stage = [
        [16, 32],  # Stage 1
        [32, 64],  # Stage 2
        [64, 128],  # Stage 3
    ]

    # Generate synthetic input data
    x = generate_input_data(batch_size, input_channels, seq_length)

    # Initialize the Multi-Stage TCN
    multi_stage_tcn = MultiStageTCN(
        input_channels, num_channels_per_stage, kernel_size=kernel_size, dropout=dropout
    )

    # Forward pass through the Multi-Stage TCN
    output = multi_stage_tcn(x)

    # Print the output shape
    print("Multi-Stage TCN output shape:", output.shape)


# Run the test
test_models()
test_multi_stage_tcn()
