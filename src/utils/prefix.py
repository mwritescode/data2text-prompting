import torch
from torch import nn

class PrefixEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.is_flat = config.is_flat
        if not self.is_flat:
            # Use a two-layer MLP to encode the prefix
            self.embedding = nn.Embedding(config.prefix_len, config.hidden_size)
            self.trans = nn.Sequential(
                nn.Linear(config.hidden_size, config.prefix_hidden_size),
                nn.Tanh(),
                nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = nn.Embedding(config.prefix_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if not self.is_flat:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values