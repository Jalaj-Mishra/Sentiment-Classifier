
# ------------------------------------------------------------------ #
# IMDB Dataset

import torch
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset



dataset = load_dataset('imdb')
print(dataset)

print(dataset['train'][0]['text'[:200]])
print(dataset['train'][0]['label'])



# ---------------------------------------------- #
# training on small data.

train_data = dataset['train'].select(range(2000))
test_data = dataset['test'].select(range(500))


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.texts = [d['text'] for d in data]
        self.labels = [d['label'] for d in  data]
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length =self.max_length, return_tensors='pt')
        return { 
            'input_ids': encoding['input_ids'].squeeze(0), 
            'attention_mask': encoding['attention_mask'].squeeze(0), 
            'label': torch.tensor(self.labels[idx], dtype=torch.long) 
        }


train_dataset = IMDBDataset(train_data, tokenizer)
test_dataset = IMDBDataset(test_data, tokenizer)\

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f'Train batches:{len(train_loader)}') 
print(f'Test batches: {len(test_loader)}')
