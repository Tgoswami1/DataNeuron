from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Input model
class TextPair(BaseModel):
    text1: str
    text2: str

# Similarity calculation
def calculate_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2)
    return similarity.item()

@app.post("/similarity")
async def similarity_endpoint(text_pair: TextPair):
    try:
        score = calculate_similarity(text_pair.text1, text_pair.text2)
        return {"similarity_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "API working successfully!"}
