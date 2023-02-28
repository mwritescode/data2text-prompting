import pandas as pd
from natsort import natsorted
from torch.utils.data import Dataset

from src.data.utils.matching import UnseenCategoryMatcher

DATAPATH = 'data/USMLE-Symp_triplets_v2.pickle'

POLARITY_MATCHING = {'positive': 0, 'negative': 1}

class USMLESymp(Dataset):
    def __init__(self, split, data_path=DATAPATH, explode_dev=False, include_category=False, include_polarity=False):
        try:
            self.data = pd.read_pickle(data_path)
        except:
            self.data = pd.read_parquet(data_path.replace('.pickle', '.parquet'))
        if include_category:
            cats = natsorted(self.data[self.data['split'] == 'train']['HPO_cat'].unique())
            self.category_mapping = {cat: i for i, cat in zip(range(len(cats)), cats)}
            self.matcher = UnseenCategoryMatcher(model='glove', dataset=self.data, dataset_name='USMLE-Symp', seen_categories=list(self.category_mapping.keys()))
        self.data = self.data[self.data['split'] == split].drop('original_idx', axis='columns')
        self.include_category = include_category
        self.include_polarity = include_polarity
        self.split = split
        
        self.data['triplet'] = self.data.triplet.apply(self.__preprocess_triplets)
    
        if split == 'train' or (split == 'dev' and explode_dev):
            self.data = self.data.explode('texts', ignore_index=True)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        elem = self.data.iloc[index]
        out = (elem['triplet'], elem['texts'])
        if self.include_category:
            category = elem['HPO_cat']
            if category not in self.category_mapping.keys():
                category = self.matcher.match(category)
            out += (self.category_mapping(category))
        
        if self.include_polarity:
            out += (POLARITY_MATCHING[elem['polarity']])
           
        return out
    
    def __preprocess_triplets(self, triplets):
        triplets_as_str = ' | ' + ' | '.join([f'{triplet[0]} : {triplet[1]} : {triplet[2]}' for triplet in triplets])
        return triplets_as_str