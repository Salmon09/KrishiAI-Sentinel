import torch
import torch.nn as nn

# =====================================================================
# 1. SPATIAL PERCEPTION LAYER (Drone Vision Matrix Processor)
# =====================================================================
class SpatialVisionBlock(nn.Module):
    """
    A Deep Learning CNN block designed to extract spatial anomaly vectors
    from high-dimensional aerial drone camera captures.
    """
    def __init__(self):
        super(SpatialVisionBlock, self).__init__()
        # Ingests a 3-channel image (RGB), outputs 16 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # Reduces spatial dimensions by half to extract dominant features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected projection layer to create a dense embedding vector
        # Assuming an input crop resolution of 64x64 scaled down to 32x32 by pooling
        self.fc = nn.Linear(16 * 32 * 32, 64) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape expected: [Batch_Size, 3, 64, 64]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten spatial feature maps into a 1D vector
        spatial_embedding = self.relu(self.fc(x))
        return spatial_embedding  # Outputs a clean 64-dimensional spatial context vector


# =====================================================================
# 2. TEMPORAL PERCEPTION LAYER (IoT Sequential Data Processor)
# =====================================================================
class TemporalTelemetryBlock(nn.Module):
    """
    An LSTM recurrent block engineered to parse sequential time-series 
    arrays from ground IoT sensors to predict structural degradation trends.
    """
    def __init__(self, input_dim=4, hidden_dim=32, num_layers=1):
        super(TemporalTelemetryBlock, self).__init__()
        # input_dim=4 represents features: temperature, soil moisture, humidity, NPK metrics
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape expected: [Batch_Size, Sequence_Length, Input_Dim]
        lstm_out, (hn, cn) = self.lstm(x)
        # Gather the final hidden state vector at the end of the sequence timeline
        last_time_step = lstm_out[:, -1, :]
        temporal_embedding = self.relu(self.fc(last_time_step))
        return temporal_embedding  # Outputs a clean 32-dimensional temporal context vector