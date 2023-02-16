from yacs.config import CfgNode as CN

_C = CN()

# System configuration parameters

_C.SYSTEM = CN()
_C.SYSTEM.NUM_WORKERS = 0
_C.SYSTEM.DEVICE = 'cuda'
_C.SYSTEM.SEED = 222

# Training configuration parameters

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 5
_C.TRAIN.NUM_EPOCHS = 5
_C.TRAIN.LR = 1e-4 # 7e-5 for t5
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.EVAL_INTERVAL = 1
_C.TRAIN.EVAL_GEN_INTERVAL = 2
_C.TRAIN.EVAL_GEN_MODE = 'beam' # or 'contrastive'
_C.TRAIN.EVAL_BASE_FOLDER = 'generation-outputs'
_C.TRAIN.DATASET = 'webnlg' #or 'usmle-symp'
_C.TRAIN.T5_PREAMBLE = ''

_C.MODEL = CN()
_C.MODEL.PLM = 'facebook/bart-base' # but also BART, BIOGPT and T5
_C.MODEL.TYPE = 'control' # or 'control' or 'prefix-pooling'
_C.MODEL.TASK_PREFIX_LEN = 5
_C.MODEL.C_PREFIX_LEN = 3
_C.MODEL.POOL_SIZE = 10
_C.MODEL.POOL_PREFIX_LEN = 2
_C.MODEL.POOL_TOP_K = 3
_C.MODEL.POOL_DROPOUT_PROB = 0.2
_C.MODEL.POOL_LEARNABLE_KEY = False
_C.MODEL.PREFIX_HIDDEN_SIZE = 512
_C.MODEL.PREFIX_DROPOUT = 0.0
_C.MODEL.USE_LAYER_DEP = False
_C.MODEL.FLAT_PREFIX = False
_C.MODEL.OBJECTIVE_TYPE = 'sentence'
_C.MODEL.INPUT_DEP_PREFIXES = [('cats', 10)]

# Wandb logging options

_C.LOG = CN()
_C.LOG.WANDB_PROJECT = 'data2text-prompting'
_C.LOG.RUN_NAME_POSTFIX = ''
# the run name is better when it's generated on the fly 
# mixing the model's PLM and type with the name postfix we provide here

# Checkpointing options during training

_C.CHECKPOINT = CN()
_C.CHECKPOINT.RESTORE = False
_C.CHECKPOINT.RESTORE_FROM = 'checkpoints/model_epoch_5.pt'
_C.CHECKPOINT.SAVE_TO_FOLDER = 'checkpoints'
_C.CHECKPOINT.INTERVAL = 100


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

def save_cfg_default():
    """
    Save in a YAML file the default version of the configuration file, 
    in order to provide a template to be modified.
    """
    with open('src/config/experiments/default.yaml', 'w') as f:
        f.write(_C.dump())
        f.flush()
        f.close()

if __name__ == '__main__':
    save_cfg_default()