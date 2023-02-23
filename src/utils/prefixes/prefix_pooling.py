import torch
import random
from torch import nn

class PromptPool(nn.Module):
    def __init__(self, 
                prompt_length=5, 
                pool_size=20, 
                top_k=3, 
                embed_dim=768, 
                embedding_key='mean_max', # or 'max', or 'mean_max' or 'features'
                prompt_init='normal', # or 'zero' or 'normal'
                use_learnable_key=False, 
                prompt_key_init='normal',
                random_idxs_prob=0.3):
        super().__init__()

        self.prompt_len = prompt_length
        self.pool_size = pool_size
        self.top_k = top_k
        self.embed_dim = embed_dim
        self.embedding_key = embedding_key
        self.use_learnable_key = use_learnable_key
        self.random_idxs_prob=random_idxs_prob

        pool_shape = (pool_size, prompt_length, embed_dim)
        self.prompt = self.__initialize_parameter(shape=pool_shape, init_method=prompt_init)
       
        if use_learnable_key:
            key_shape = (pool_size, embed_dim)
            self.keys = self.__initialize_parameter(shape=key_shape, init_method=prompt_key_init)
        else:
            self.keys = self.prompt.mean(dim=1)

    def __initialize_parameter(self, shape, init_method):
        if init_method == 'zero':
            param = nn.Parameter(torch.zeros(shape))
        elif init_method == 'uniform':
            param = nn.Parameter(torch.randn(shape))
            nn.init.uniform_(param, -1, 1)
        elif init_method == 'normal':
            param = nn.Parameter(torch.randn(shape))
            nn.init.normal_(param)
        else:
            raise ValueError(f"Unsupported initialization method {init_method}. Choose between 'zero', 'uniform' and 'normal'.")
        return param
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, input_embed, features=None):
        batch_size = input_embed.shape[0]
        if self.embedding_key == 'mean':
            embed_keys = input_embed.mean(dim=1)
        elif self.embedding_key == 'max':
            embed_keys = input_embed.max(dim=1).values
        elif self.embedding_key == 'mean_max':
            embed_keys = input_embed.max(dim=1).values + 2 * input_embed.mean(dim=1)
        elif self.embedding_key == 'features' and features is not None:
            embed_keys = features
        else:
            raise ValueError(f"Unsupported way {self.embedding_key} of computing the embedding keys!. Choose between 'mea', 'max', 'mean_max' and 'features'.")
        
        if not self.use_learnable_key:
            self.keys = self.prompt.mean(dim=1)
            
        prompt_keys_l2 = self.l2_normalize(self.keys, dim=1)
        embed_keys_l2 = self.l2_normalize(embed_keys, dim=1)

        similarity = torch.matmul(embed_keys_l2, prompt_keys_l2.T)
        ids = similarity.topk(k=self.top_k, dim=1).indices
        if random.random() < self.random_idxs_prob and self.training:
            ids = torch.randint(low=0, high=self.pool_size, size=ids.shape)

        selected_prompts = self.prompt[ids].reshape(batch_size, self.top_k * self.prompt_len, -1)
        return selected_prompts

class PrefixEncoderWithPromptPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.is_flat = config.is_flat
        self.prefix_len = config.prefix_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        self.all_prefix_len = config.prefix_len + config.top_k * config.input_dep_prompt_len

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

        self.prompt_pool = PromptPool(
            prompt_length=config.input_dep_prompt_len, 
            pool_size=config.pool_size,
            top_k=config.top_k,
            embed_dim=config.hidden_size,
            use_learnable_key=config.use_learnable_key,
            random_idxs_prob=config.random_idxs_prob)
        
        self.dropout = torch.nn.Dropout(config.prefix_dropout_prob)
        self.pool_dropout = torch.nn.Dropout(config.pool_dropout_prob)

    def forward(self, inputs_embeds, batch_size):
        prefix = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.embedding.weight.device)

        if not self.is_flat:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        
        input_dep_prefix = self.prompt_pool(inputs_embeds)
        input_dep_prefix = self.pool_dropout(self.trans(input_dep_prefix))
        if inputs_embeds.shape[0] < past_key_values.shape[0]:
            input_dep_prefix = input_dep_prefix.repeat(past_key_values.shape[0]//inputs_embeds.shape[0], 1, 1)

        past_key_values = torch.cat([past_key_values, input_dep_prefix], dim=1)
        
        past_key_values = past_key_values.view(
            batch_size,
            self.all_prefix_len,
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

class PrefixEncoderForSeq2SeqModelsWithPromptPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prefix_dec = PrefixEncoderWithPromptPool(config=config)
        if config.use_encoder_prefix:
            self.prefix_enc = PrefixEncoderWithPromptPool(config=config)
        if config.use_cross_prefix:
            self.prefix_cross = PrefixEncoderWithPromptPool(config=config)

        self.use_encoder_prefix = config.use_encoder_prefix
        self.use_cross_prefix = config.use_cross_prefix
        self.prefix_len = config.prefix_len
    
    def forward(self, inputs_embeds, batch_size, sample_size=1):
        batch_size_dec = batch_size * sample_size
        batch_size_enc = batch_size
        decoder_past_key_values = self.prefix_dec(inputs_embeds, batch_size_dec)

        if self.use_encoder_prefix:
            encoder_past_key_values = self.prefix_enc(inputs_embeds, batch_size_enc)
        if self.use_cross_prefix:
            cross_past_key_values = self.prefix_cross(inputs_embeds, batch_size_dec)

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



