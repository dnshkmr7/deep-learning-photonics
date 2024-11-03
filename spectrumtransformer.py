import torch
import torch.nn as nn
import math

class SpectrumTransformer(nn.Module):
    def __init__(self, input_dim, param_dim, output_dim, nhead, num_encoder_layers, dropout = 0.1):
        super(SpectrumTransformer, self).__init__()
        self.embed_dim = ((input_dim - 1) // nhead + 1) * nhead

        self.embedding = nn.Linear(input_dim, self.embed_dim)
        self.param_embedding = nn.Linear(param_dim, self.embed_dim)

        self.pos_encoder = PositionalEncoding(self.embed_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model = self.embed_dim, 
                                                   nhead = nhead, 
                                                   dropout = dropout, 
                                                   batch_first = True
                                                   )
        self.activation = nn.Tanh()
        self.encoder = nn.TransformerEncoder(encoder_layer, 
                                             num_layers = num_encoder_layers
                                             )
        self.decoder = nn.Linear(self.embed_dim, output_dim)

    def forward(self, src, params):
        src_embedded = self.embedding(src)
        params_embedded = self.param_embedding(params).unsqueeze(1)

        combined_input = src_embedded + params_embedded

        combined_input = self.pos_encoder(combined_input)
        combined_input = self.activation(combined_input)

        output = self.encoder(combined_input)
        return self.decoder(output.mean(dim = 1))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)