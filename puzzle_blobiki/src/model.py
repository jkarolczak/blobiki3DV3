import torch
import torch.nn as nn
import torch.nn.functional as f


class FeatureExtractorLayer(nn.Module):
    def __init__(self, input_size: int = 7, output_size: int = 128) -> None:
        super().__init__()

        self.weight_1 = nn.Parameter(torch.empty(input_size, output_size))
        self.bias_1 = nn.Parameter(torch.zeros(output_size))
        nn.init.normal_(self.weight_1)

        self.weight_2 = nn.Parameter(torch.empty(output_size, output_size))
        self.bias_2 = nn.Parameter(torch.zeros(output_size))
        nn.init.normal_(self.weight_2)

        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x, self.weight_1) + self.bias_1
        x = f.relu(x)
        x1 = torch.matmul(x, self.weight_2) + self.bias_2
        x1 = f.relu(x1)
        x = self.layer_norm(x1 + x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, input_size: int = 7, hidden_size: int = 128, num_extractors: int = 4) -> None:
        super().__init__()
        self.extractor_1 = FeatureExtractorLayer(input_size=input_size, output_size=hidden_size)
        self.deep_extractors = nn.Sequential(
            *[FeatureExtractorLayer(input_size=hidden_size, output_size=hidden_size) for _ in
              range(num_extractors - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor_1(x)
        x = self.deep_extractors(x)
        x = x.max(dim=1).values
        return x


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=dropout)
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)

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
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = f.relu(x)
        x = self.linear_2(x)
        return x


class BlobNet(nn.Module):
    def __init__(self, input_size: int = 7, hidden_size: int = 128, num_extractors: int = 64, num_transformers: int = 8,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_size=input_size, hidden_size=hidden_size,
                                                  num_extractors=num_extractors)
        self.transformer = Transformer(num_transformers=num_transformers, hidden_size=hidden_size, dropout=dropout)
        self.head = RegressionHead(hidden_size=hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= 180.0  # Normalise to [-1, 1], as the angles are in degrees
        x = self.feature_extractor(x)
        x = self.transformer(x)
        x = self.head(x)
        return x
