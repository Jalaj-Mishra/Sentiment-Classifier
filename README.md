
# BERT Sentiment Classifier Fine-tuned BERT model for sentiment analysis using PyTorch and HuggingFace Transformers. 

## Tech Stack - PyTorch 2.x - HuggingFace Transformers (bert-base-uncased) - IMDB Dataset (50K reviews) 

## Results - Validation Accuracy: ~92%- Training samples: 2,000 - Epochs: 3 

## Run 
```
bash pip install -r requirements.txt

python 03_train.py python 04_inference.py 

``` 

## Architecture Input Text → BERT

Tokenizer → BERT Encoder (12 layers) → [CLS] token → Linear Classifier → Positive/Negative