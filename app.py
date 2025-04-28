from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertModel
import torch

app = FastAPI()

# Load DistilBERT model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Input data model
class TextPair(BaseModel):
    text1: str
    text2: str

# Function to calculate BERT similarity
def calculate_similarity(text1, text2):
    inputs = tokenizer(text1, text2, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings1 = outputs.last_hidden_state[:, 0, :]
    embeddings2 = outputs.last_hidden_state[:, 1, :]
    similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
    return similarity.item()

@app.post('/similarity')
async def similarity(text_pair: TextPair):
    try:
        similarity_score = calculate_similarity(text_pair.text1, text_pair.text2)
        return {"similarity_score": similarity_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get('/')
def read_root():
    return {"message": "FastAPI with DistilBERT is up and running!"}
