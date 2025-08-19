from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from llm_pipeline import ask_nba, get_health

load_dotenv()

app = FastAPI(title="NBA AnalyiXpert API", version="0.1.0")

# Allow frontend (Vite dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all domains (safe enough if just read-only API)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str

@app.get("/health")
def health():
    return get_health()

@app.post("/ask")
async def ask_bot(query: Query):
    try:
        md, rows, chart = ask_nba(query.question)
        return {
            "answer": md,
            "rows": rows,
            "chart": chart,  # may be None if no chart requested or failed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
