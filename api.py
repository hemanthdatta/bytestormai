# api.py

import os
from re import M
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Import your existing functions & pipeline from main.py
from main import ingest_documents, ingest_structured, load_structured, pipeline

app = FastAPI(title="RAG API")

# Allow all origins (adjust in prod!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory where uploaded files are stored temporarily
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Ensure your API keys are set in the environment
# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
# GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY",  "").strip()
MISTRAL_API_KEY = 'VrAMhIHO61FjHTAYeibtLmla52bWnorV'
GEMINI_API_KEY = 'AIzaSyBS2npulOMMZ9WRj7b-UpoYHXVSa0Jju4o'
if not MISTRAL_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("Set MISTRAL_API_KEY and GEMINI_API in env before starting.")

# Load any pre-existing structured index on startup
@app.on_event("startup")
def startup_event():
    load_structured()


class QueryRequest(BaseModel):
    question: str


@app.post("/ingest", summary="Ingest unstructured docs (txt, pdf, docx, pptx, images)")
async def api_ingest(files: List[UploadFile] = File(...)):
    paths = []
    try:
        for f in files:
            dest = os.path.join(UPLOAD_DIR, f.filename)
            with open(dest, "wb") as out:
                out.write(await f.read())
            paths.append(dest)

        ingest_documents(paths)
        return {"status": "ok", "ingested": [os.path.basename(p) for p in paths]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest error: {e}")


@app.post("/ingest_structured", summary="Ingest structured data (csv, xlsx, etc)")
async def api_ingest_structured(files: List[UploadFile] = File(...)):
    paths = []
    try:
        for f in files:
            dest = os.path.join(UPLOAD_DIR, f.filename)
            with open(dest, "wb") as out:
                out.write(await f.read())
            paths.append(dest)

        ingest_structured(paths)
        return {"status": "ok", "current_db": os.path.basename(paths[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Structured ingest error: {e}")


@app.post("/query", summary="Ask a question of the RAG system")
def api_query(req: QueryRequest):
    try:
        out = pipeline.invoke({"question": req.question})
        answer = out.get("answer")
        if not answer:
            raise ValueError("No answer found.")
        return {"question": req.question, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
