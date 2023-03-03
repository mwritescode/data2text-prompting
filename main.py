import os
import torch
import random
import torchinfo
import numpy as np
from argparse import ArgumentParser
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.models.auto import (
    AutoModelForPrefixTuning, 
    AutoModelForControlPrefixes,
    AutoModelForPrefixPooling
)
from src.data.webNLG import webNLG
from src.data.USMLESymp import USMLESymp
from src.config.config import get_cfg_defaults
from src.utils.training.trainer import Trainer
from src.utils.training.collators import (
    DataColatorForEncoderDecoderModel, 
    DataCollatorForDecoderOnlyModel
)

def get_model_from_cfg(cfg):
    model_cfg = {
        'plm_name_or_path': cfg.MODEL.PLM,
        'prefix_len': cfg.MODEL.TASK_PREFIX_LEN,
        'prefix_dropout_prob': cfg.MODEL.PREFIX_DROPOUT,
        'prefix_hidden_size':cfg.MODEL.PREFIX_HIDDEN_SIZE,
        'is_flat': cfg.MODEL.FLAT_PREFIX,
        'objective_type': cfg.MODEL.OBJECTIVE_TYPE,
        'use_layer_dep': cfg.MODEL.USE_LAYER_DEP
    }
    if cfg.MODEL.TYPE == 'prefix':
        model = AutoModelForPrefixTuning.from_config(**model_cfg)
    elif cfg.MODEL.TYPE == 'control':
        input_dep_prefixes = {cat_tuple[0]: cat_tuple[1] for cat_tuple in cfg.MODEL.INPUT_DEP_PREFIXES}
        print(input_dep_prefixes)
        model_cfg.update({
            'control_prefix_len': cfg.MODEL.C_PREFIX_LEN,
            'input_dep_prefixes': input_dep_prefixes,
        })
        model = AutoModelForControlPrefixes.from_config(**model_cfg)
    else:
        model_cfg.update({
            'pool_size': cfg.MODEL.POOL_SIZE,
            'input_dep_prompt_len': cfg.MODEL.POOL_PREFIX_LEN,
            'top_k': cfg.MODEL.POOL_TOP_K,
            'use_learnable_key': cfg.MODEL.POOL_LEARNABLE_KEY,
            'pool_dropout_prob': cfg.MODEL.POOL_DROPOUT_PROB,
        })
        model = AutoModelForPrefixPooling.from_config(**model_cfg)
    return model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # safe to call even when the GPU is not availabe

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('config_path', help='Path of the model\'s configuration file')
    args = args.parse_args()
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    set_seed(seed=cfg.SYSTEM.SEED)


    input_dep_prefixes = {cat_tuple[0]: cat_tuple[1] for cat_tuple in cfg.MODEL.INPUT_DEP_PREFIXES}

    has_category = cfg.MODEL.TYPE == 'control' and 'cats' in input_dep_prefixes.keys()
    has_polarity = cfg.MODEL.TYPE == 'control' and 'polarity' in input_dep_prefixes.keys()
    model_name = cfg.MODEL.PLM
    model = get_model_from_cfg(cfg)
    torchinfo.summary(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'gpt2' in model_name:
        print('Adapting the size of the model embedding to include <|pad|>:')
        print('len(tokenizer) = ', len(tokenizer))
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        print('len(tokenizer) = ', len(tokenizer))
    
    dataset_class = webNLG
    dataset_kwargs = {
        'include_category': has_category
    }
    if cfg.TRAIN.DATASET.lower().strip() != 'webnlg':
        dataset_class = USMLESymp
        dataset_kwargs['include_polarity'] = has_polarity

    data_train = dataset_class('train', **dataset_kwargs)
    data_val = dataset_class('dev', **dataset_kwargs)
    data_val_expl = dataset_class('dev', explode_dev=True, **dataset_kwargs)
    data_test = dataset_class('test', **dataset_kwargs)

    separator = tokenizer.decode(model.config.eos_token_id)
    if 'gpt' in cfg.MODEL.PLM:
        collator = DataCollatorForDecoderOnlyModel(
            has_category=has_category, has_polarity=has_polarity, 
            tokenizer=tokenizer, separator=separator)
    else:
        collator = DataColatorForEncoderDecoderModel(
            has_category=has_category, has_polarity=has_polarity,
            tokenizer=tokenizer, t5_preamble=cfg.TRAIN.T5_PREAMBLE)

    batch_size = cfg.TRAIN.BATCH_SIZE
    num_workers = cfg.SYSTEM.NUM_WORKERS
    train_loader = DataLoader(
        data_train, batch_size=batch_size, shuffle=True, 
        pin_memory=True, num_workers=num_workers, collate_fn=collator(train=True))
    val_loader = DataLoader(
        data_val_expl, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers, collate_fn=collator(train=True))
    val_loader_gen = DataLoader(
        data_val, batch_size=1, shuffle=False,
        pin_memory=True, num_workers=num_workers, collate_fn=collator(train=False))
    test_loader = DataLoader(
        data_test, batch_size=1, shuffle=False,
        pin_memory=True, num_workers=num_workers, collate_fn=collator(train=False)
    )

    trainer = Trainer(
        model=model, tokenizer=tokenizer, 
        train_loader=train_loader, val_loader=val_loader, 
        val_loader_gen=val_loader_gen, test_loader=test_loader,
        device=cfg.SYSTEM.DEVICE, cfg=cfg)
    trainer.fit()
