from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from textblob import TextBlob
from rake_nltk import Rake
from collections import Counter
from textstat import flesch_reading_ease
import nltk

# --- NLTK setup ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = FastAPI()

# --- Allow all CORS for frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request schema ---
class UploadRequest(BaseModel):
    text: str
    filename: str

# --- Root check ---
@app.get("/")
async def root():
    return {"message": "Ink Insights API active ✅"}

# --- Upload and analyze text ---
@app.post("/upload")
async def analyze_text(data: UploadRequest):
    text = data.text.strip()
    filename = data.filename

    # Word count
    words = text.split()
    word_count = len(words)

    # Sentiment
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment = {
        "positive": round(max(polarity, 0), 2),
        "neutral": round(1 - abs(polarity), 2),
        "negative": round(abs(min(polarity, 0)), 2),
        "overall": "positive" if polarity > 0.05 else "negative" if polarity < -0.05 else "neutral"
    }

    # Emotions (approx.)
    emotions = {
        "joy": round(max(polarity, 0) * 0.8, 2),
        "sadness": round(abs(min(polarity, 0)) * 0.7, 2),
        "anger": round(abs(min(polarity, 0)) * 0.3, 2),
        "fear": round(abs(min(polarity, 0)) * 0.2, 2),
        "surprise": round(abs(polarity) * 0.5, 2)
    }

    # Readability
    try:
        score = flesch_reading_ease(text)
        level = (
            "Easy" if score > 80 else
            "Good" if score > 60 else
            "Challenging" if score > 30 else
            "Very Hard"
        )
    except Exception:
        score, level = 0, "N/A"
    readability = {"score": round(score, 2), "level": level}

    # Keywords
    rake = Rake()
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases()[:10]
    common_words = [w.lower() for w, _ in Counter([w.lower() for w in words if len(w) > 3]).most_common(10)]
    keywords = ranked_phrases if ranked_phrases else common_words

    # Summary
    summary_sentences = blob.sentences[:2]
    summary_text = " ".join(str(s) for s in summary_sentences)

    return {
        "filename": filename,
        "word_count": word_count,
        "sentiment": sentiment,
        "emotions": emotions,
        "readability": readability,
        "keywords": keywords,
        "summary": summary_text
    }

# --- Delete (stateless confirmation) ---
@app.delete("/delete")
async def delete_data():
    return {"message": "✅ All temporary data cleared (nothing stored)."}
