import torch
from transformers import BertForSequenceClassification, BertTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = './models/bert_sentiment.pth'



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()



def predict(text):
    enc = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids = enc['input_ids'].to(DEVICE), attention_mask=enc['attention_mask'].to(DEVICE))
        probs = torch.softmax(outputs.logits, dim=1)
        label = probs.argmax().item()
        conf = probs[0][label].item()
    return 'POSITIVE ' if label == 1 else 'NEGATIVE ', round(conf*100, 1)



test_sentences = ['''
This movie was absolutely incredible, I loved every minute!', 'Terrible film, complete
waste of time and money.', 'The acting was brilliant and the story was gripping.', 'I
fell asleep halfway through, so boring.', 'One of the best films I have seen in years!',
'''
]

for text in test_sentences:
    sentiment, confidence = predict(text) 
    print(f'{sentiment} ({confidence}%) | {text[:60]}')