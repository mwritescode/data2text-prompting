import torch
from torch import nn
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.generation_beam_constraints import Constraint
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import StoppingCriteriaList
from typing import List, Optional, Tuple, Dict, Union, Iterable, Callable
from transformers import PretrainedConfig, AutoConfig, BartPretrainedModel
from transformers.generation_utils import GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput

from ..utils.prefix import PrefixEncoderForSeq2SeqModels
from ..utils.modeling_bart import BartForConditionalGeneration

class BartForConditionalGenerationWithPrefix(BartPretrainedModel):
    def __init__(self, config, pretrained_model=None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        print(config)
        if pretrained_model is None:
            print('instantiating model')
            self.pretrained_model = BartForConditionalGeneration.from_pretrained(config.plm_name_or_path)
        else:
            self.pretrained_model = pretrained_model

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.prefix_len = config.prefix_len
        self.prefix_encoder = PrefixEncoderForSeq2SeqModels(config)
    
    def train(self, mode=True):
        super().train(mode)
        self.pretrained_model.eval()
    
    def get_encoder(self):
        return self.pretrained_model.get_encoder()

    def get_decoder(self):
        return self.pretrained_model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        return self.pretrained_model.resize_token_embeddings(new_num_tokens=new_num_tokens)

    def get_output_embeddings(self):
        return self.pretrained_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.pretrained_model.set_output_embeddings(new_embeddings=new_embeddings)
    
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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        batch_size = input_ids.shape[0]
        prefix_key_values = self.prefix_encoder(batch_size=batch_size)
        
        return self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            prefix_key_values=prefix_key_values
        )
    
    def generate(
        self, 
        input_ids,
        attention_mask=None,
        **generation_kwargs
        ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
        
        batch_size = input_ids.shape[0]
        prefix_key_values = self.prefix_encoder(batch_size, sample_size=generation_kwargs['num_beams'])

        return self.pretrained_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_key_values=prefix_key_values,
            **generation_kwargs)


class BartPrefixTuningConfig(PretrainedConfig):
    model_type = "bart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(self, 
        plm_name_or_path='facebook/bart-base',
        prefix_len=5,
        prefix_dropout_prob=0.0,
        prefix_hidden_size=512,
        is_flat=False,
        objective_type='sentence',
        use_encoder_prefix=True,
        use_cross_prefix=True,
        **kwargs):
        super().__init__(**kwargs)
        self.plm_name_or_path = plm_name_or_path
        self.prefix_len = prefix_len
        self.prefix_dropout_prob = prefix_dropout_prob
        self.prefix_hidden_size = prefix_hidden_size
        self.is_flat = is_flat
        self.use_encoder_prefix = use_encoder_prefix
        self.use_cross_prefix = use_cross_prefix
        plm_config = AutoConfig.from_pretrained(plm_name_or_path).to_dict()
        del plm_config['_name_or_path']
        self.update(plm_config)
        self.objective_type = objective_type # or 'sentence' or 'token' which is the classical objective