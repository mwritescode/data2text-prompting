import copy
import torch
from abc import ABC, abstractmethod

class DataCollator(ABC):
    def __init__(self, has_category=False, has_polarity=False, tokenizer=None) -> None:
        super().__init__()
        self.has_category = has_category
        self.has_polarity = has_polarity
        self.tokenizer = tokenizer

    def __call__(self, train=True):
        return self.train_collate_fn if train else self.test_collate_fn

    @abstractmethod
    def train_collate_fn(self, batch):
        pass

    @abstractmethod
    def test_collate_fn(self, batch):
        pass

    def _optionally_include_cat(self, batch):
        out = ()
        if self.has_category:
            cat = torch.tensor([elem[2] for elem in batch])
            out = (cat, )
        return out
    
    def _optionally_include_polarity(self, batch):
        out = ()
        idx = 3 if self.has_category else 2
        if self.has_polarity:
            pol = torch.tensor([elem[idx] for elem in batch])
            out = (pol, )
        return out

class DataCollatorForDecoderOnlyModel(DataCollator):
    def __init__(self, has_category=False, tokenizer=None, separator='</s>'):
        super().__init__(has_category=has_category, tokenizer=tokenizer)
        self.separator = separator
    
    def train_collate_fn(self, batch):
        input_str = [' {} {} {} {}'.format(elem[0], self.separator, elem[1], self.separator) for elem in batch]
        src = [' {} {}'.format(elem[0], self.separator) for elem in batch]

        input_tok = self.tokenizer(input_str, return_tensors='pt', padding=True)
        src_tok = self.tokenizer(src, return_length=True)

        label_tok = copy.deepcopy(input_tok)
        for idx, elem in enumerate(label_tok.input_ids):
            length = src_tok.length[idx]
            elem[:length] = torch.LongTensor([self.tokenizer.pad_token_id for _ in range(length)])

        out = (input_tok, label_tok) + self._optionally_include_cat(batch) + self._optionally_include_polarity(batch)

        return out

    def test_collate_fn(self, batch):
        input_str =  [' {} {}'.format(elem[0], self.separator) for elem in batch]
        target = [elem[1] for elem in batch]

        input_tok = self.tokenizer(input_str, return_tensors='pt', padding=True, return_length=True)
        out = (input_tok, target) + self._optionally_include_cat(batch) + self._optionally_include_polarity(batch)

        return out


class DataColatorForEncoderDecoderModel(DataCollator):
    def __init__(self, has_category=False, tokenizer=None, t5_preamble='') -> None:
        super().__init__(has_category=has_category, tokenizer=tokenizer)
        self.t5_preamble = t5_preamble
        self.is_t5 = 't5' in tokenizer.__class__.__name__.lower()
        
    def _process_input(self, text):
        new_triplet = text.replace('| ', '& ').replace(' : ', ' | ')
        new_text = new_triplet[2:]
        return new_text

    def _get_input_str(self, batch):
        if self.is_t5:
            input_str = [self.t5_preamble + self._process_input(elem[0]) for elem in batch]
        else:
            input_str = [elem[0] for elem in batch]
        return input_str

    def train_collate_fn(self, batch):
        input_str = self._get_input_str(batch)
        target = [elem[1] for elem in batch]

        input_tok = self.tokenizer(input_str, return_tensors='pt', padding=True)
        label_tok = self.tokenizer(target, return_tensors='pt', padding=True)

        out = (input_tok, label_tok) + self._optionally_include_cat(batch) + self._optionally_include_polarity(batch)
    
        return out
    
    def test_collate_fn(self, batch):
        input_str = self._get_input_str(batch)
        target = [elem[1] for elem in batch]

        input_tok = self.tokenizer(input_str, return_tensors='pt', padding=True)
        out = (input_tok, target) + self._optionally_include_cat(batch) + self._optionally_include_polarity(batch)
        
        return out

