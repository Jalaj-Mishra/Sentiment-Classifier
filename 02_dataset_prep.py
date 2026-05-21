from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = 'This movie was absolutely fantastic!'
tokens = tokenizer(text, return_tensors='pt')
print('Input IDs: ', tokens['input_ids'][0])
print('Attention Masks: ', tokens['attention_mask'])


# decoding back to original
decoded = tokenizer.decode(tokens['input_ids'][0])
print(decoded)