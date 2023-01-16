import pandas as pd
from torch.utils.data import Dataset

DATAPATH = 'data/USMLE-Symp_triplets.pickle'

class USMLESymp(Dataset):
    def __init__(self, split, data_path=DATAPATH, explode_dev=False):
        self.data = pd.read_pickle(data_path)
        self.data = self.data[self.data['split'] == split].drop('original_idx', axis='columns')
        
        self.data['triplet'] = self.data.triplet.apply(self.__preprocess_triplets)
    
        if split == 'train' or (split == 'dev' and explode_dev):
            self.data = self.data.explode('texts', ignore_index=True)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        elem = self.data.iloc[index]
        out = (elem['triplet'], elem['texts'])
        return out
    
    def __preprocess_triplets(self, triplets):
        triplets_as_str = ' | ' + ' | '.join([f'{triplet[0]} : {triplet[1]} : {triplet[2]}' for triplet in triplets])
        return triplets_as_str