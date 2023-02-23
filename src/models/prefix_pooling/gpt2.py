import torch
from torch import nn
from transformers import PretrainedConfig, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel, GPT2LMHeadModel

from src.utils.prefixes.prefix_pooling import PrefixEncoderWithPromptPool
from src.models.custom_save import CustomSavePreTrainedModel

class GPT2PrefixPoolConfig(PretrainedConfig):
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, 
        plm_name_or_path='gpt2-medium',
        prefix_len=5,
        prefix_dropout_prob=0.0,
        prefix_hidden_size=512,
        is_flat=False,
        pad_token_id=50257,
        objective_type='sentence',
        use_layer_dep=False,
        pool_size=10,
        input_dep_prompt_len=2,
        top_k=2,
        use_learnable_key=False,
        pool_dropout_prob=0.2,
        random_idxs_prob=0.3,
        **kwargs):
        super().__init__(**kwargs)
        self.plm_name_or_path = plm_name_or_path
        self.prefix_len = prefix_len
        self.prefix_dropout_prob = prefix_dropout_prob
        self.prefix_hidden_size = prefix_hidden_size
        self.is_flat = is_flat
        plm_config = AutoConfig.from_pretrained(plm_name_or_path).to_dict()
        del plm_config['_name_or_path']
        self.update(plm_config)
        self.pad_token_id = pad_token_id
        self.vocab_size = self.pad_token_id + 1 
        self.objective_type = objective_type # or 'sentence' or 'token' which is the classical objective
        self.use_layer_dep = use_layer_dep
        self.pool_size = pool_size
        self.input_dep_prompt_len = input_dep_prompt_len
        self.top_k = top_k
        self.use_learnable_key = use_learnable_key
        self.pool_dropout_prob = pool_dropout_prob
        self.random_idx_prob = random_idxs_prob

class GPT2PrefixPoolWithLMHeadModel(GPT2PreTrainedModel, CustomSavePreTrainedModel):
    _keys_to_ignore_on_load_missing = [r'\b(pretrained_model.)']
    def __init__(self, config, pretrained_model=None):
        super().__init__(config)
        print(config)
        if pretrained_model is None:
            self.pretrained_model = GPT2LMHeadModel.from_pretrained(config.plm_name_or_path, pad_token_id=config.pad_token_id)
            self.pretrained_model.resize_token_embeddings(config.vocab_size)
        else:
            self.pretrained_model = pretrained_model

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.prefix_len = config.prefix_len + config.top_k * config.input_dep_prompt_len
        self.prefix_encoder = PrefixEncoderWithPromptPool(config)

    def train(self, mode=True):
        super().train(mode)
        self.pretrained_model.eval()
    
    def get_input_embeddings(self) -> nn.Module:
        return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.pretrained_model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.pretrained_model.set_output_embeddings(new_embeddings=new_embeddings)
    
    def get_input_embeddings(self):
        return self.pretrained_model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.pretrained_model.set_input_embeddings(new_embeddings=new_embeddings)

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        inputs_embeds = kwargs.get('inputs_embeds', None)

        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        
        batch_size = input_ids.shape[0]
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, self.prefix_len).to(input_ids.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        
        if past is None:
            inputs_embeds = self.pretrained_model.get_input_embeddings()(input_ids)
            past = self.prefix_encoder(inputs_embeds=inputs_embeds, batch_size=batch_size)
            if position_ids is not None:
                position_ids = position_ids[:, self.prefix_len:]
                
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "inputs_embeds": inputs_embeds
        }

    def forward(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):  
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is not None and self.training:
            raise ValueError("past_key_value is dedicated to prefix tokens in this implementation. Please don't use it for anything else.")
        
        if past_key_values is None:
            batch_size = input_ids.shape[0]
            inputs_embeds = self.pretrained_model.get_input_embeddings()(input_ids)
            past_key_values = self.prefix_encoder(inputs_embeds=inputs_embeds, batch_size=batch_size)
            if attention_mask is not None:
                prefix_attention_mask = torch.ones(batch_size, self.prefix_len).to(input_ids.device)
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        labels_for_plm = None if self.config.objective_type == 'sentence' else labels
        input_ids = None if inputs_embeds is not None else input_ids

        transformer_outputs = self.pretrained_model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels_for_plm,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if labels_for_plm is None:
            lm_logits = transformer_outputs.logits if return_dict else transformer_outputs[0]

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                batch_size, seqlen, _ = shift_logits.shape
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(batch_size, seqlen).sum(dim=-1)
                loss = loss.mean()

            if not return_dict:
                output = (lm_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
            )
        else:
            return transformer_outputs

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )