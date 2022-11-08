import torch
import random
from torch import nn
from typing import List, Optional, Tuple, Dict, Union
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartModel, BartForConditionalGeneration, _expand_mask, shift_tokens_right
from transformers.models.bart.modeling_bart import BartAttention, BartEncoderLayer, BartDecoderLayer, BartEncoder, BartDecoder
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPast, Seq2SeqModelOutput, BaseModelOutput, Seq2SeqLMOutput

class BartAttentionWithCacheKey(BartAttention):
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0, 
        is_decoder: bool = False, 
        bias: bool = True,
        cache_key: str = 'self'):
        super().__init__(embed_dim, num_heads, dropout, is_decoder, bias)
        self.cache_key = cache_key
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        key_value_states: Optional[torch.Tensor] = None, 
        past_key_value: Optional[Dict[str, Tuple[torch.Tensor]]] = None, # Chnaged this to be of the new type
        attention_mask: Optional[torch.Tensor] = None, 
        layer_head_mask: Optional[torch.Tensor] = None, 
        output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, Tuple[torch.Tensor]]]]:

        in_past_key_value = past_key_value[self.cache_key]
        attn_output, attn_weights_reshaped, out_past_key_value = super().forward(hidden_states, key_value_states, in_past_key_value, attention_mask, layer_head_mask, output_attentions)
        past_key_value[self.cache_key] = out_past_key_value

        return attn_output, attn_weights_reshaped, out_past_key_value

class BartEncoderLayerWithCacheKey(BartEncoderLayer):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        new_attn = BartAttentionWithCacheKey(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            cache_key='encoder')
        setattr(self, 'self_attn', new_attn)
    
    def forward(
        self, 
        hidden_states: torch.FloatTensor, 
        attention_mask: torch.FloatTensor, 
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:

        # TODO: should we return the past_key_values in the encoder?
        
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class BartDecoderLayerWithCacheKey(BartDecoderLayer):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        new_decoder_attn = BartAttentionWithCacheKey(            
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            cache_key='self')
        new_cross_attn = BartAttentionWithCacheKey(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            cache_key='encoder_decoder'
        )
        setattr(self, 'self_attn', new_decoder_attn)
        setattr(self, 'encoder_attn', new_cross_attn)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        encoder_hidden_states: Optional[torch.Tensor] = None, 
        encoder_attention_mask: Optional[torch.Tensor] = None, 
        layer_head_mask: Optional[torch.Tensor] = None, 
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None, 
        past_key_value: Optional[Tuple[torch.Tensor]] = None, 
        output_attentions: Optional[bool] = False, 
        use_cache: Optional[bool] = True) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        return super().forward(
            hidden_states, 
            attention_mask, 
            encoder_hidden_states, 
            encoder_attention_mask, 
            layer_head_mask, 
            cross_attn_layer_head_mask, 
            past_key_value, 
            output_attentions, 
            use_cache)
    
class BartEncoderWithCacheKey(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
    
        new_layers = nn.ModuleList([BartEncoderLayerWithCacheKey(config) for _ in range(config.encoder_layers)])
        setattr(self, 'layers', new_layers)
    
    def forward(
        self, 
        input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        head_mask: Optional[torch.Tensor] = None, 
        inputs_embeds: Optional[torch.FloatTensor] = None, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None, 
        return_dict: Optional[bool] = None,
        past_key_values: Optional[Tuple[Dict[str, Tuple[torch.Tensor]]]] = None) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        encoder_cache = ()

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                past_key_value = past_key_values[idx] if past_key_values is not None else {}
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions, past_key_value)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        past_key_value=past_key_value
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            
            encoder_cache += (past_key_value.copy(), )

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions, encoder_cache] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions, past_key_values=encoder_cache
        )

class BartDecoderWithCacheKey(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)

        new_layers = nn.ModuleList([BartDecoderLayerWithCacheKey(config) for _ in range(config.encoder_layers)])
        setattr(self, 'layers', new_layers)
    
    def forward(
        self, 
        input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        encoder_hidden_states: Optional[torch.FloatTensor] = None, 
        encoder_attention_mask: Optional[torch.LongTensor] = None, 
        head_mask: Optional[torch.Tensor] = None, 
        cross_attn_head_mask: Optional[torch.Tensor] = None, 
        past_key_values: Optional[Tuple[Dict[str, Tuple[torch.FloatTensor]]]] = None, 
        inputs_embeds: Optional[torch.FloatTensor] = None, 
        use_cache: Optional[bool] = None, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None, 
        return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        
        return super().forward(
            input_ids, 
            attention_mask, 
            encoder_hidden_states, 
            encoder_attention_mask, 
            head_mask, 
            cross_attn_head_mask, 
            past_key_values, 
            inputs_embeds, 
            use_cache, 
            output_attentions, 
            output_hidden_states, 
            return_dict)

class BartModelWithCachekey(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        new_enc = BartEncoderWithCacheKey(config, self.shared)
        new_dec = BartDecoderWithCacheKey(config, self.shared)
        setattr(self, 'encoder', new_enc)
        setattr(self, 'decoder', new_dec)

        self.post_init()
    
    def forward(
        self, 
        input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        decoder_input_ids: Optional[torch.LongTensor] = None, 
        decoder_attention_mask: Optional[torch.LongTensor] = None, 
        head_mask: Optional[torch.Tensor] = None, 
        decoder_head_mask: Optional[torch.Tensor] = None, 
        cross_attn_head_mask: Optional[torch.Tensor] = None, 
        encoder_outputs: Optional[List[torch.FloatTensor]] = None, 
        past_key_values: Optional[Tuple[Dict[str, Tuple[torch.FloatTensor]]]] = None, 
        inputs_embeds: Optional[torch.FloatTensor] = None, 
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None, 
        use_cache: Optional[bool] = None, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None, 
        return_dict: Optional[bool] = None) -> Union[Tuple, Seq2SeqModelOutput]:

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                past_key_values=past_key_values
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class BartForConditionalGenerationWithCacheKey(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        new_model = BartModelWithCachekey(config)
        setattr(self, 'model', new_model)
    
    def forward(
        self, 
        input_ids: torch.LongTensor = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        decoder_input_ids: Optional[torch.LongTensor] = None, 
        decoder_attention_mask: Optional[torch.LongTensor] = None, 
        head_mask: Optional[torch.Tensor] = None,         
        decoder_head_mask: Optional[torch.Tensor] = None, 
        cross_attn_head_mask: Optional[torch.Tensor] = None, 
        encoder_outputs: Optional[List[torch.FloatTensor]] = None, 
        past_key_values: Optional[Tuple[Dict[str, Tuple[torch.FloatTensor]]]] = None, 
        inputs_embeds: Optional[torch.FloatTensor] = None, 
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None, 
        labels: Optional[torch.LongTensor] = None, 
        use_cache: Optional[bool] = None, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None, 
        return_dict: Optional[bool] = None) -> Union[Tuple, Seq2SeqLMOutput]:

        return super().forward(
            input_ids, 
            attention_mask, 
            decoder_input_ids, 
            decoder_attention_mask, 
            head_mask, 
            decoder_head_mask, 
            cross_attn_head_mask, 
            encoder_outputs, 
            past_key_values, 
            inputs_embeds, 
            decoder_inputs_embeds, 
            labels, 
            use_cache, 
            output_attentions, 
            output_hidden_states, 
            return_dict)
