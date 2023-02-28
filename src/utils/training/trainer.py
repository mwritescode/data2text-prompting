import os
import re
import tqdm
import wandb
import random
import numpy as np

import torch
import evaluate
from torch import optim
from transformers import get_linear_schedule_with_warmup, Adafactor

from src.config.config import get_cfg_defaults

class Trainer:
    def __init__(self, model, tokenizer, train_loader, val_loader=None, val_loader_gen=None, test_loader=None, device='cuda', cfg=get_cfg_defaults()):
        self._set_seed(seed=cfg.SYSTEM.SEED)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_loader_gen = val_loader_gen
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = cfg.TRAIN.NUM_EPOCHS
        self.eval_interval = cfg.TRAIN.EVAL_INTERVAL
        self.eval_gen_interval = cfg.TRAIN.EVAL_GEN_INTERVAL
        self.bleu = evaluate.load("bleu")

        num_training_steps = cfg.TRAIN.NUM_EPOCHS * len(train_loader)

        # Use adafactor for T5
        is_t5 = 't5' in self.model.__class__.__name__.lower()
        if is_t5:
            self.optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=cfg.TRAIN.LR)
        else:
            self.optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer, 
            num_warmup_steps=2000 if is_t5 else 0,
            num_training_steps=num_training_steps
            )
        
        self.cfg = cfg
    
    def _get_model_data(self, data, use_labels=True):
        in_data = {
            'input_ids': data[0]['input_ids'].to(self.device),
            'attention_mask': data[0]['attention_mask'].to(self.device)
        }
        if use_labels:
            labels = data[1]['input_ids'].to(self.device)
            labels[labels == self.tokenizer.pad_token_id] = -100
            in_data['labels'] = labels
    
        if len(data) > 2:
            # Control prefixes setting
            cond_dict = {}
            for i, key in zip(range(2, len(data)), self.cfg.MODEL.INPUT_DEP_PREFIXES):
                cond_dict[key[0]] = data[i].to(self.device)
        return in_data
    
    def _train_epoch(self, i):
        epoch_loss = 0.0
        current_step = 1
        self.model.train()
        pbar = tqdm.tqdm(self.train_loader, desc=f'Epoch {i}: Training...', total=len(self.train_loader))
        for data in pbar:
            self.optimizer.zero_grad(set_to_none=True)

            data = self._get_model_data(data)
            loss = self.model(**data).loss
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss / current_step})

            loss.backward()
            if 't5' not in self.model.__class__.__name__.lower():
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            current_step += 1
        return {'loss': epoch_loss / current_step}
    
    @torch.no_grad()
    def _eval_epoch(self, i):
        val_loss = 0.0
        current_step = 1
        self.model.eval()

        pbar = tqdm.tqdm(self.val_loader, desc=f'Epoch {i}: Eval...', total=len(self.val_loader))
        for data in pbar:
            data = self._get_model_data(data)
            loss = self.model(**data).loss
        
            val_loss += loss.item()
            pbar.set_postfix({'loss': val_loss / current_step})

            current_step += 1

        return {'loss': val_loss / current_step}
    
    def _save_checkpoint(self, epoch, path, resume=True):
        if resume:
            # If it's kmclr also save the values in the kmeans.cluster_centers
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'wandb_id': wandb.run.id
                }, path)
        else:
            self.model.save_pretrained(path)
    
    def _restore_checkpoint(self, path, model_only=False):
        print('Restoring checkpoint ', path)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        starting_epoch, wandb_id = 0, 0
        if not model_only:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            starting_epoch = checkpoint['epoch'] + 1
            wandb_id = checkpoint['wandb_id']
        return starting_epoch, wandb_id
    
    def _set_seed(self, seed=42):
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
    
    def _get_generation_kwargs(self):
        shared_kwargs = {
            'max_length': 300,
            'min_length': 5,
            'pad_token_id': self.tokenizer.pad_token_id,
            'early_stopping': True
        }
        if 'gpt2' in self.model.__class__.__name__.lower():
            shared_kwargs['bad_words_ids'] = [[628], [198]]
        if self.cfg.TRAIN.EVAL_GEN_MODE == 'beam':
            shared_kwargs.update({
                'num_beams': 5,
                'top_p': 0.9,
                'do_sample': False,
                'num_return_sequences': 1
            })
        else:
            shared_kwargs.update({
                'penalty_alpha': 0.1, 
                'top_k': 5,
            })
        return shared_kwargs
    
    def _generate(self, data, outfolder='dev'):
        self.model.eval()
        pbar = tqdm.tqdm(data, desc=f'Generating...', total=len(data))
        gen_kwargs = self._get_generation_kwargs()
        outfolder = os.path.join(
            self.cfg.TRAIN.EVAL_BASE_FOLDER, 
            wandb.run.name,
            outfolder)
        os.makedirs(outfolder, exist_ok=True)
        all_references = []
        all_generated = []
        for data in pbar:
            in_data = self._get_model_data(data, use_labels=False)
            references = [list(e) for e in data[1]]
            references_log = ['\n'.join(list(e)) for e in data[1]]
            generated = self.model.generate(
                **in_data, ** gen_kwargs)
            if 'gpt' in self.model.__class__.__name__.lower():
                generated = generated[:, data[0]['length']:]
            generated = self.tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            all_references.extend(references)
            all_generated.extend(generated)
            with open(os.path.join(outfolder, 'references.txt'), 'a', encoding='utf-8') as outfile:
                outfile.write('\n\n'.join(references_log) + '\n\n')
            with open(os.path.join(outfolder, 'generated.txt'), 'a', encoding='utf-8') as outfile:
                outfile.write('\n\n'.join(generated) + '\n\n')
        return all_references, all_generated
    
    def _save_gen_results_to_wandb(self, data, outfolder):
        references, generated = self._generate(data, outfolder=outfolder)
        print(len(references), len(generated))
        table_data = [[ref, gen] for ref, gen in zip(references, generated) ]
        gen_table = wandb.Table(columns=['references', 'generated'], data=table_data)
        wandb.log({f'generated-{outfolder}': gen_table}, commit=False)
        return references, generated

    def fit(self):  
        starting_epoch = 0
        os.makedirs(self.cfg.CHECKPOINT.SAVE_TO_FOLDER, exist_ok=True)
        if self.cfg.CHECKPOINT.RESTORE:
            starting_epoch, wandb_id = self._restore_checkpoint(path=self.cfg.CHECKPOINT.RESTORE_FROM)
        else:
            wandb_id = wandb.util.generate_id()
        wandb_run_name = f'{re.sub(r"^.*?/", "", self.cfg.MODEL.PLM)}-{self.cfg.MODEL.TYPE}'
        if self.cfg.LOG.RUN_NAME_POSTFIX != '':
            wandb_run_name += f'-{self.cfg.LOG.RUN_NAME_POSTFIX}'

        wandb.init(
            # set the wandb project where this run will be logged
            project=self.cfg.LOG.WANDB_PROJECT,
            name=wandb_run_name,
            id=wandb_id,
            resume='allow',
    
            # track hyperparameters and run metadata
            config={
            **self.cfg,
            "dataset": self.cfg.TRAIN.DATASET,
            })

        val_bleus = []
        previous_bleu = 0.0
        for i in range(starting_epoch, self.num_epochs):
            log_dict = self._train_epoch(i)
            log_dict['lr'] = self.scheduler.get_last_lr()[0]

            if (i % self.eval_interval) == 0:
                eval_log_dict = self._eval_epoch(i)
                wandb_dict = {'train':log_dict, 'val': eval_log_dict}
            else:
                wandb_dict = {'train': log_dict}
            
            if ((i+1) % self.eval_gen_interval) == 0:
                references, generated = self._save_gen_results_to_wandb(self.val_loader_gen, outfolder=f'val-{i}')
                val_results = self.bleu.compute(predictions=generated, references=references)
                print(f'VALIDATION RESULTS EPOCH {i}:', val_results)
                val_bleus.append(val_results['bleu'])
                wandb_dict['val']['bleu'] = val_results['bleu']
            
            wandb.log(wandb_dict, step=i, commit=True)

            if (i % self.cfg.CHECKPOINT.INTERVAL) == 0 and val_bleus[-1] >= previous_bleu:
                    path = os.path.join(self.cfg.CHECKPOINT.SAVE_TO_FOLDER, f'epoch_{i}.pt')
                    self._save_checkpoint(epoch=i, path=path, resume=True)
                    path_to_remove = os.path.join(self.cfg.CHECKPOINT.SAVE_TO_FOLDER, f'epoch_{i-1}.pt')
                    if os.path.exists(path_to_remove):
                        os.remove(path_to_remove)
                        
            previous_bleu = val_bleus[-1]
            
        optimal_idx = np.argmax(val_bleus)
        restore_path = os.path.join(self.cfg.CHECKPOINT.SAVE_TO_FOLDER, f'epoch_{optimal_idx}.pt')
        _ = self._restore_checkpoint(restore_path, model_only=True)

        path = os.path.join(self.cfg.CHECKPOINT.SAVE_TO_FOLDER, 'final')
        self._save_checkpoint(epoch=i, path=path, resume=False)

        wandb.run.summary["val_bleu"] = val_bleus[optimal_idx]

        references, generated = self._save_gen_results_to_wandb(self.test_loader, outfolder=f'test-final')
        test_results = self.bleu.compute(predictions=generated, references=references)
        print('TEST RESULTS:', test_results)
        wandb.run.summary["test_bleu"] = test_results['bleu']
        
        artifact = wandb.Artifact(wandb_run_name, type='model')
        artifact.add_dir(path)
        wandb.log_artifact(artifact)

        wandb.finish()





