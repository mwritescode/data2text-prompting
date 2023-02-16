import torch
from transformers import PreTrainedModel

class ForceRequiredAttributeDefinitionMeta(type):
    def __call__(cls, *args, **kwargs):
        class_object = type.__call__(cls, *args, **kwargs)
        class_object.check_required_attributes()
        return class_object

class CustomSavePreTrainedModel(PreTrainedModel, metaclass=ForceRequiredAttributeDefinitionMeta):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.prefix_encoder = None

    def check_required_attributes(self):
        if self.prefix_encoder is None:
            raise NotImplementedError('Subclass must define self.prefix_encoder attribute.')
    
    def save_pretrained(
            self, 
            save_directory, 
            is_main_process=True, 
            state_dict=None, 
            save_function=torch.save, 
            push_to_hub=False, 
            max_shard_size="10GB", 
            safe_serialization=False, 
            **kwargs):
        if state_dict is None:
            state_dict = self.prefix_encoder.state_dict(prefix='prefix_encoder.')
            
        return super().save_pretrained(
            save_directory, 
            is_main_process, 
            state_dict, 
            save_function, 
            push_to_hub, 
            max_shard_size, 
            safe_serialization, 
            **kwargs)