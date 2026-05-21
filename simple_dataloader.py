# Importing required Libraries.
from torch.utils.data import Dataset, DataLoader


# Custom Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]



# Fake Data
texts = ['great movie', 'terrible film', 'loved it', 'waste of time', 'amazing']
labels = [1, 0, 1, 0, 1]


# Dataset + DataLoader
dataset = SentimentDataset(texts, labels)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterating over the data in batches
for batch_texts, batch_labels in loader:
    print('Batch texts: ', batch_texts)
    print('Batch labels:', batch_labels)
    print('---')
