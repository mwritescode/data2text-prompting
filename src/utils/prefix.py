import torch
from torch import nn

class PrefixEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.is_flat = config.is_flat
        self.prefix_len = config.prefix_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.prefix_len).long()
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
        
        self.dropout = torch.nn.Dropout(config.prefix_dropout_prob)

    def forward(self, batch_size):
        prefix = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.embedding.weight.device)

        if not self.is_flat:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        
        past_key_values = past_key_values.view(
            batch_size,
            self.prefix_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return past_key_values