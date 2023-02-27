from collections import OrderedDict

from src.models.prefix_tuning import (
    BartPrefixTuningConfig, 
    BartForConditionalGenerationWithPrefix,
    T5PrefixTuningConfig,
    T5ForConditionalGenerationWithPrefix,
    BioGPTPrefixTuningConfig, 
    BioGPTPrefixTuningWithLMHeadModel,
    GPT2PrefixTuningConfig,
    GPT2PrefixTuningWithLMHeadModel
)

from src.models.control_prefixes import (
    BartControlPrefixesConfig,
    BartForConditionalGenerationWithControlPrefixes,
    T5ControlPrefixesConfig,
    T5ForConditionalGenerationWithControlPrefixes,
    BioGPTControlPrefixesConfig,
    BioGPTControlPrefixesWithLMHeadModel,
    GPT2ControlPrefixesConfig,
    GPT2ControlPrefixesWithLMHeadModel
)

from src.models.prefix_pooling import (
    BartPrefixPoolConfig, 
    BartForConditionalGenerationWithPrefixPool,
    T5PrefixPoolConfig,
    T5ForConditionalGenerationWithPrefixPool,
    BioGPTPrefixPoolConfig, 
    BioGPTPrefixPoolWithLMHeadModel,
    GPT2PrefixPoolConfig,
    GPT2PrefixPoolWithLMHeadModel
)


AUTO_PREFIX_TUNING = OrderedDict([
    ('gpt2', [GPT2PrefixTuningConfig, GPT2PrefixTuningWithLMHeadModel]),
    ('biogpt', [BioGPTPrefixTuningConfig, BioGPTPrefixTuningWithLMHeadModel]),
    ('bart', [BartPrefixTuningConfig, BartForConditionalGenerationWithPrefix]),
    ('t5', [T5PrefixTuningConfig, T5ForConditionalGenerationWithPrefix]),
    ('SciFive', [T5PrefixTuningConfig, T5ForConditionalGenerationWithPrefix])
])

AUTO_CONTROL_PREFIXES = OrderedDict([
    ('gpt2', [GPT2ControlPrefixesConfig, GPT2ControlPrefixesWithLMHeadModel]),
    ('biogpt', [BioGPTControlPrefixesConfig, BioGPTControlPrefixesWithLMHeadModel]),
    ('bart', [BartControlPrefixesConfig, BartForConditionalGenerationWithControlPrefixes]),
    ('t5', [T5ControlPrefixesConfig, T5ForConditionalGenerationWithControlPrefixes]),
    ('SciFive', [T5ControlPrefixesConfig, T5ForConditionalGenerationWithControlPrefixes]),
])

AUTO_PREFIX_POOLING = OrderedDict([
    ('gpt2', [GPT2PrefixPoolConfig, GPT2PrefixPoolWithLMHeadModel]),
    ('biogpt', [BioGPTPrefixPoolConfig, BioGPTPrefixPoolWithLMHeadModel]),
    ('bart', [BartPrefixPoolConfig, BartForConditionalGenerationWithPrefixPool]),
    ('t5', [T5PrefixPoolConfig, T5ForConditionalGenerationWithPrefixPool]),
    ('SciFive', [T5PrefixPoolConfig, T5ForConditionalGenerationWithPrefixPool])
])

class AutoModelCustom:
    class_mappings = {}
    @classmethod
    def from_config(cls, **kwargs) -> None:
        model_name = kwargs.get('plm_name_or_path', None)
        if model_name is None:
            raise Exception(
                'You should specify which model you want to instanstiate using the argument plm_name_or_path')
        model = None
        for key, value in cls.class_mappings.items():
            if key in model_name:
                config = value[0](**kwargs)
                model = value[1](config)
                break
        return model


class AutoModelForPrefixTuning(AutoModelCustom):
    class_mappings = AUTO_PREFIX_TUNING
    
class AutoModelForControlPrefixes(AutoModelCustom):
    class_mappings = AUTO_CONTROL_PREFIXES

class AutoModelForPrefixPooling(AutoModelCustom):
    class_mappings = AUTO_PREFIX_POOLING