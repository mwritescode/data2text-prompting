import torch
from torch import nn

#TODO: I don't think we are training a way that deals with unseen categories, re-read the paper to see what they implement in that case

class ControlPrefixEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.is_flat = config.is_flat
        self.prefix_len = config.prefix_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        self.input_dep_prefixes = config.input_dep_prefixes # Should be a dict {'category_name': {'seen': num_seen_classes_in_category, 'unseen': num_unseen_classes_in_category}}
        self.control_prefix_len = config.control_prefix_len

        self.prefix_tokens = torch.arange(self.prefix_len).long()

        if not self.is_flat:
            # Use a two-layer MLP to encode the prefix
            self.embedding = nn.Embedding(config.prefix_len, config.hidden_size)
            self.trans = nn.Sequential(
                nn.Linear(config.hidden_size, config.prefix_hidden_size),
                nn.Tanh(),
                nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
            # Encode the control prefixes
            self.control_prefixes = nn.ModuleDict()
            for name, num_classes_dict in self.input_dep_prefixes.items():
                num_unseen_classes = num_classes_dict.get('unseen', 0)
                num_classes = num_classes_dict.get('seen', 1) + num_unseen_classes
                self.control_prefixes[f'embed_{name}'] = nn.Embedding(
                    self.control_prefix_len * num_classes, config.hidden_size
                )
                if num_unseen_classes > 0:
                    print('Maxing prefixes for unseen classes zero')
                    with torch.no_grad():
                        self.control_prefixes[f'embed_{name}'].weight[-num_unseen_classes:] = torch.zeros(
                            self.control_prefixes[f'embed_{name}'].weight[-num_unseen_classes:].shape
                        ).to(self.embedding.weight.device)

        else:
            self.embedding = nn.Embedding(config.prefix_len, config.num_hidden_layers * 2 * config.hidden_size)
        
        self.dropout = torch.nn.Dropout(config.prefix_dropout_prob)

    def forward(self, conditional_info, batch_size):
        prefix = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.embedding.weight.device)

        if not self.is_flat:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
            print(past_key_values.shape)

            for name, num_classes_dict in self.input_dep_prefixes.items():
                num_classes = num_classes_dict.get('seen', 1) + num_classes_dict.get('unseen', 0)
                input_dep_tokens = torch.arange(
                    self.control_prefix_len * num_classes
                    ).long().unsqueeze(0).expand(batch_size, -1).to(self.embedding.weight.device)
                
                input_dep_embeddings = self.control_prefixes[f'embed_{name}'](input_dep_tokens)
                input_dep_past_key_values = self.trans(input_dep_embeddings)

                idxmap = {
                    i: ((i) * self.control_prefix_len, ((i + 1) * self.control_prefix_len)) 
                    for i in range(num_classes)
                    }
                cond = list(map(idxmap.get, conditional_info[name].tolist()))
        
                input_dep_past_key_values = torch.stack([input_dep_past_key_values[i, j[0] : j[1], :] for i,j in enumerate(cond)])
                if input_dep_past_key_values.shape[0] < past_key_values.shape[0]:
                    # Repeat control prefix n times if we are in contrastive/beam search decoding
                    input_dep_past_key_values = input_dep_past_key_values.repeat_interleave(
                        past_key_values.shape[0] // input_dep_past_key_values.shape[0], dim=0)

                past_key_values = torch.cat([past_key_values, input_dep_past_key_values], dim=1)
        
        _, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            seqlen,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return past_key_values

class ControlPrefixEncoderForSeq2SeqModels(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prefix_dec = ControlPrefixEncoder(config=config)
        if config.use_encoder_prefix:
            self.prefix_enc = ControlPrefixEncoder(config=config)
        if config.use_cross_prefix:
            self.prefix_cross = ControlPrefixEncoder(config=config)

        self.use_encoder_prefix = config.use_encoder_prefix
        self.use_cross_prefix = config.use_cross_prefix
        self.prefix_len = config.prefix_len
    
    def forward(self, conditional_info, batch_size, sample_size=1):
        batch_size_dec = batch_size * sample_size
        batch_size_enc = batch_size
        decoder_past_key_values = self.prefix_dec(conditional_info, batch_size_dec)

        if self.use_encoder_prefix:
            encoder_past_key_values = self.prefix_enc(conditional_info, batch_size_enc)
        if self.use_cross_prefix:
            cross_past_key_values = self.prefix_cross(conditional_info, batch_size_dec)

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