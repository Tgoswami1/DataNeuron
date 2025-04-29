from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/similarity")
async def get_similarity(text_pair: TextPair):
    try:
        emb1 = model.encode(text_pair.text1, convert_to_tensor=True)
        emb2 = model.encode(text_pair.text2, convert_to_tensor=True)
        score = util.pytorch_cos_sim(emb1, emb2)
        return {"similarity_score": score.item()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "API is working!"}
