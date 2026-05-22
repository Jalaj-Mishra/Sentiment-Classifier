# Imports
import boto3
from dotenv import load_dotenv
import os

load_dotenv()


# Connect to s3
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)



# Create bucket 
BUCKET = 'jalaj-ml-models'  
s3.create_bucket(Bucket=BUCKET) 

# listing buckets
print(s3.list_buckets())


# Model upload karo
s3.upload_file('./models/bert_sentiment.pth', BUCKET, 'bert_sentiment.pth')
print(f'Model uploaded to s3://{BUCKET}/bert_sentiment.pth') 

# Download karo (deployment ke waqt) 
s3.download_file(BUCKET, 'bert_sentiment.pth', './models/bert_sentiment_downloaded.pth')
print('Model downloaded from S3')