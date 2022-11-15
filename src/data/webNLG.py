from datasets import load_dataset

from torch.utils.data import Dataset

COLS_TO_KEEP = ['modified_triple_sets', 'lex', 'category']

TESTSET_CATEGORY_MAP = {
    'a': ['testdata_with_lex', 'testdata_unseen_with_lex'],
    's': ['testdata_with_lex'],
    'u': ['testdata_unseen_with_lex']
}

CAT2IDX = {
    'Airport': 0,
    'Astronaut': 1,
    'Building': 2,
    'City': 3,
    'ComicsCharacter': 4,
    'Food': 5,
    'Monument': 6,
    'SportsTeam': 7,
    'University': 8,
    'WrittenWork': 9
}

class webNLG(Dataset):
    def __init__(self, split, test_mode='a', include_category=False):
        self.data = load_dataset("web_nlg", 'webnlg_challenge_2017', split=split)
        self.split = split
        self.include_category = include_category
        if split == 'test':
            self.data = self.data.filter(lambda x: x['test_category'] in TESTSET_CATEGORY_MAP[test_mode])
        self.data = self.data.remove_columns([col for col in self.data.column_names if col not in COLS_TO_KEEP])
        self.data = self.data.map(self.__preprocess_row, remove_columns=['modified_triple_sets', 'lex'])
        if split == 'train':
            self.data = self.data.filter(lambda x: len(x['label']) > 0)
        self.data = self.data.to_pandas()
        if split == 'train':
            self.data = self.data.explode(column='label', ignore_index=True)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        elem = self.data.iloc[index]
        out = (elem['text'], elem['label'])
        if self.include_category:
            out += (elem['category'], )
        return out
    
    def __preprocess_row(self, row):
        row['text'] = ' | '.join([triplet.replace(" | ", " : ") for triplet in row['modified_triple_sets']['mtriple_set'][0]])
        row['text'] = '| ' + row['text']
        if self.split == 'train':
            row['label'] = [text for i, text in enumerate(row['lex']['text']) if row['lex']['comment'][i] == 'good']
        else:
            row['label'] = row['lex']['text']
        
        if self.include_category:
            row['category'] = CAT2IDX[row['category']]
        return row