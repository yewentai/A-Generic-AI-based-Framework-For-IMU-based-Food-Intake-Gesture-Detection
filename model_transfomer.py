import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, num_classes)
    
    def forward(self, x):
        # x has shape [batch_size, seq_len, input_dim]
        x = self.embedding(x)
        x = x * np.sqrt(self.model_dim)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # required shape for transformer [seq_len, batch_size, model_dim]
        
        transformer_out = self.transformer_encoder(x)  # [seq_len, batch_size, model_dim]
        transformer_out = transformer_out.mean(dim=0)  # Pool along seq_len
        out = self.output_layer(transformer_out)  # [batch_size, num_classes]
        return out