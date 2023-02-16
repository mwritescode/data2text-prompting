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

        self.use_layer_dep =config.use_layer_dep

        self.prefix_tokens = torch.arange(self.prefix_len).long()
        if not self.is_flat:
            # Use a two-layer MLP to encode the prefix
            self.embedding = nn.Embedding(config.prefix_len, config.hidden_size)
            self.trans = nn.Sequential(
                nn.Linear(config.hidden_size, config.prefix_hidden_size),
                nn.Tanh(),
                nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
            if self.use_layer_dep:
                self.weights = nn.Parameter(torch.zeros(config.num_hidden_layers - 1))
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

        if self.use_layer_dep:
            new_vals = ()
            for i, past_key_value in enumerate(past_key_values):
                if i > 0:
                    past_key_value = past_key_value + self.weights[i-1] * past_key_values[i-1]                
                new_vals += (past_key_value, )
        else:
            new_vals = past_key_values

        return new_vals

class PrefixEncoderForSeq2SeqModels(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prefix_dec = PrefixEncoder(config=config)
        if config.use_encoder_prefix:
            self.prefix_enc = PrefixEncoder(config=config)
        if config.use_cross_prefix:
            self.prefix_cross = PrefixEncoder(config=config)

        self.use_encoder_prefix = config.use_encoder_prefix
        self.use_cross_prefix = config.use_cross_prefix
        self.prefix_len = config.prefix_len
    
    def forward(self, batch_size, sample_size=1):
        batch_size_dec = batch_size * sample_size
        batch_size_enc = batch_size
        decoder_past_key_values = self.prefix_dec(batch_size_dec)

        if self.use_encoder_prefix:
            encoder_past_key_values = self.prefix_enc(batch_size_enc)
        if self.use_cross_prefix:
            cross_past_key_values = self.prefix_cross(batch_size_dec)

        results = []
        for i, key_value in enumerate(decoder_past_key_values):
            past_dict = {'decoder': key_value}
            if self.use_encoder_prefix:
                encoder_key_value = encoder_past_key_values[i]
                past_dict['encoder'] = encoder_key_value
            if self.use_cross_prefix:
                cross_key_value = cross_past_key_values[i]
                past_dict['cross'] = cross_key_value
            results.append(past_dict)
        
        return tuple(results)