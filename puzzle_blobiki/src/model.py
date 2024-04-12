import torch
import torch.nn as nn
import torch.nn.functional as f


class FeatureExtractor(nn.Module):
    def __init__(self, input_size: int = 7, hidden_size: int = 128, num_lstms: int = 64, batch_size: int = 32,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.lstm_0 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_lstms, dropout=dropout,
                              batch_first=True, bidirectional=True)
        self.h0_0 = torch.randn(2 * num_lstms, batch_size, hidden_size)
        self.c0_0 = torch.randn(2 * num_lstms, batch_size, hidden_size)

        self.linear_0 = nn.Linear(2 * hidden_size, 2 * hidden_size)

        self.lstm_1 = nn.LSTM(input_size=2 * hidden_size, hidden_size=hidden_size, num_layers=num_lstms, dropout=dropout,
                              batch_first=True, bidirectional=True)
        self.h0_1 = torch.randn(num_lstms, batch_size, hidden_size)
        self.c0_1 = torch.randn(num_lstms, batch_size, hidden_size)

        self.linear_1 = nn.Linear(2 * hidden_size, 2 * hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.h0_0, self.c0_0 = self.h0_0.to(x.device), self.c0_0.to(x.device)
        x, (self.h0_0, self.c0_0) = self.lstm_0(x)
        x = self.linear_0(x)
        x = f.relu(x)

        self.h0_1, self.c0_1 = self.h0_1.to(x.device), self.c0_1.to(x.device)
        x1, (self.h0_0, self.c0_0) = self.lstm_1(x)
        x1 = self.linear_1(x1)
        x1 = f.relu(x1)

        x = x1
        x = x.max(dim=1).values

        return x


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=2 * hidden_size, num_heads=8, dropout=dropout)
        self.layer_norm_1 = nn.LayerNorm(2 * hidden_size)
        self.linear = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.layer_norm_2 = nn.LayerNorm(2 * hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, _ = self.attention(x, x, x)
        x = self.layer_norm_1(x1 + x)

        x1 = self.linear(x)
        x = self.layer_norm_2(x1 + x)

        return x


class Transformer(nn.Module):
    def __init__(self, num_transformers: int = 8, hidden_size: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.transformer_layers = nn.Sequential(
            *[TransformerLayer(hidden_size=hidden_size, dropout=dropout) for _ in range(num_transformers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer_layers(x)
        return x


class RegressionHead(nn.Module):
    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 1)
        self.linear_3 = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = f.relu(x)
        x = self.linear_2(x)
        x = f.relu(x)
        x = self.linear_3(x)
        return x


class BlobNet(nn.Module):
    def __init__(self, input_size: int = 7, hidden_size: int = 128, num_lstms: int = 64, num_transformers: int = 8,
                 batch_size: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_size=input_size, hidden_size=hidden_size, num_lstms=num_lstms,
                                                  batch_size=batch_size, dropout=dropout)
        self.transformer = Transformer(num_transformers=num_transformers, hidden_size=hidden_size, dropout=dropout)
        self.head = RegressionHead(hidden_size=hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= 180.0  # Normalise to [-1, 1], as the angles are in degrees
        x = self.feature_extractor(x)
        x = self.transformer(x)
        x = self.head(x)
        return x
