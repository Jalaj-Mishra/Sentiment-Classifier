
# Imports
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# CONFIG
MODEL_NAME='bert-base-uncased'
MAX_LENGTH='128'
BATCH_SIZE='16'
EPOCHS='3'
LR='2e-5'
TRAIN_SIZE=2000
TEST_SIZE=500
SAVE_PATH='./models/bert_sentiment.pth'
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DEVICE
print(f'Using device: {DEVICE}')



# DATASET
class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.texts = [d['texts'] for d in data]
        self.labels = [d['labels'] for d in data]
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        enc = self.tokeinzer(self.texts[index], 
            padding='max_lenght', 
            truncation=True, 
            max_length=MAX_LENGTH, 
            return_tensors='pt'
        )

        return {
            'input_ids': enc['input_ids'].squeeze(0), 
            'attention_mask':enc['attention_mask'].squeeze(0), 
            'label': torch.tensor(self.labels[index], dtype=torch.long)} 



# LOAD DATASET
print('Loading dataset...')

dataset = load_dataset('imdb')
train_data = dataset['train'].select(range(TRAIN_SIZE))
test_data = dataset['test'].select(range(TEST_SIZE))
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

train_loader = DataLoader(IMDBDataset(train_data, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(IMDBDataset(test_data, tokenizer), batch_size=BATCH_SIZE)

print(f'Train: {TRAIN_SIZE} | Test: {TEST_SIZE} | Batches/epoch: {len(train_loader)}')



# MODEL
print('Loading BERT model...')

model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)



# OPTIMIZER
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)


# TRAINING LOOP
def train_epoch(model, loader, device, scheduler):
    model.train
    total_loss, preds_all, labels_all = 0, [], []
    for i, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss

        logits = outputs.logits

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

        total_loss += loss.item()

        preds_all.extend(logits.argmax(dim=1).cpu().np())

        labels_all.extend(labels.cpu().np())

        if (i+1) % 20 == 0:
            print(f' Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}')
            acc = accuracy_score(labels_all,preds_all)

        return total_loss / len(loader), acc


def eval_epoch(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds_all.extend(outputs.logits.argmax(dim=1).cpu().np())
            labels_all.extend(labels.cpu().np())
    return accuracy_score(labels_all, preds_all), preds_all, labels_all



# RUN TRAINING

best_acc = 0
for epoch in range(EPOCHS):
    print(f'\n=== EPOCH {epoch+1}/{EPOCHS} ===')
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
    val_acc, _, _ = eval_epoch(model, test_loader, DEVICE)
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')

    if val_acc > best_acc:
        best_acc = val_acc torch.save(model.state_dict(), SAVE_PATH)
        print(f' Model saved! Best accuracy: {best_acc:.4f}') print(f'\nTraining complete! Best Val Accuracy: {best_acc:.4f}')
        
