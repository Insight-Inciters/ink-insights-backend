from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your website to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return {"message": "Ink Insights backend is working!"}

@app.post("/upload")
async def upload(request: Request):
    data = await request.json()
    text = data.get("text", "")
    filename = data.get("filename", "unknown.txt")

    # Fake NLP results for now
    result = {
        "filename": filename,
        "word_count": len(text.split()),
        "sentiment": {"positive": 0.73, "neutral": 0.15, "negative": 0.12},
        "emotions": {"joy": 0.45, "anger": 0.10, "sadness": 0.15, "fear": 0.05, "surprise": 0.25},
        "readability": {"score": 68.3, "level": "Good"},
        "keywords": ["creativity", "journey", "expression", "growth"]
    }

    return JSONResponse(result)
