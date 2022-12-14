import os
import re
import numpy as np
import pandas as pd
from datasets import load_dataset
import gensim.downloader as gloader

CAMEL_CASE_SPLIT = r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))'

MODELS_MAPPING = {
    'glove': 'glove-wiki-gigaword-100',
    'fasttext': 'fasttext-wiki-news-subwords-300',
    'word2vec': 'word2vec-google-news-300'
}

class UnseenCategoryMatcher:
    def __init__(self, dataset, dataset_name, seen_categories, model='glove') -> None:
        self.seen_categories = seen_categories
        filename = os.path.join('src', 'data', 'utils', f'{model}_{dataset_name}_match.csv')
        if not os.path.exists(filename):
            pretrained_embeds = self._download_embeddings(model)
            self.lookup = self._build_lookup_table(pretrained_embeds, dataset, filename)
        else:
            self.lookup = pd.read_csv(filename)

    def match(self, category):
        out = self.lookup[self.lookup['category'] == category].reset_index(drop=True)
        return out['match'].iloc[0]
    
    def _build_lookup_table(self, pretrained_embeddings, dataset, filename):
        unseen_vals = set(dataset.unique('category')) - set(self.seen_categories)
        matches = {'category': [], 'match': []}
        lowercase_cats = [[v.lower() for v in re.findall(CAMEL_CASE_SPLIT, cat)] for cat in self.seen_categories]
        for val in unseen_vals:
            as_list_val = [v.lower() for v in re.findall(CAMEL_CASE_SPLIT, val)]
            distances = [pretrained_embeddings.n_similarity(as_list_val, category) for category in lowercase_cats]
            matches['match'].append(self.seen_categories[np.argmax(distances)])
            matches['category'].append(val)
        lookup_table = pd.DataFrame(matches)
        lookup_table.to_csv(filename, index=None)
        return lookup_table

    def _download_embeddings(self, model_name='glove'):
        download_path = MODELS_MAPPING[model_name]
        try:
            emb_model = gloader.load(download_path)
        except ValueError as e:
            print("Invalid embedding model name!")
            raise e
        return emb_model
    
    
