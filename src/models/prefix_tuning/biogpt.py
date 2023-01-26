import torch
from torch import nn
from transformers import PretrainedConfig, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.biogpt.modeling_biogpt import BioGptForCausalLM, BioGptPreTrainedModel

from src.utils.prefix import PrefixEncoder

class BioGPTPrefixTuningConfig(PretrainedConfig):
    model_type = "biogpt"

    def __init__(self, 
        plm_name_or_path='microsoft/biogpt',
        prefix_len=5,
        prefix_dropout_prob=0.0,
        prefix_hidden_size=512,
        is_flat=False,
        objective_type='sentence',
        use_layer_dep=False,
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
        self.objective_type = objective_type # or 'sentence' or 'token' which is the classical objective
        self.use_layer_dep = use_layer_dep

class BioGPTPrefixTuningWithLMHeadModel(BioGptPreTrainedModel):
    def __init__(self, config, pretrained_model=None):
        super().__init__(config)
        print(config)
        if pretrained_model is None:
            self.pretrained_model = BioGptForCausalLM.from_pretrained(config.plm_name_or_path)
        else:
            self.pretrained_model = pretrained_model

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.prefix_len = config.prefix_len
        self.prefix_encoder = PrefixEncoder(config)

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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        
        batch_size = input_ids.shape[0]
        attention_mask = kwargs.get("attention_mask", None)

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, self.prefix_len).to(input_ids.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        if past_key_values is None:
            past_key_values = self.prefix_encoder(batch_size=batch_size)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask
        }

    def forward(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
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
            past_key_values = self.prefix_encoder(batch_size=batch_size)
            if attention_mask is not None:
                prefix_attention_mask = torch.ones(batch_size, self.prefix_len).to(input_ids.device)
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        labels_for_plm = None if self.config.objective_type == 'sentence' else labels

        transformer_outputs = self.pretrained_model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
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