from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from textblob import TextBlob
from rake_nltk import Rake
import nltk
from collections import Counter
from textstat import flesch_reading_ease

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = FastAPI()

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can later restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UploadRequest(BaseModel):
    text: str
    filename: str

@app.get("/")
def home():
    return {"message": "Ink Insights backend is running ✅"}

@app.post("/upload")
async def upload_file(data: UploadRequest):
    text = data.text.strip()
    filename = data.filename

    # --- Word count ---
    words = text.split()
    word_count = len(words)

    # --- Sentiment analysis ---
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment = {
        "positive": round(max(sentiment_polarity, 0), 2),
        "neutral": round(1 - abs(sentiment_polarity), 2),
        "negative": round(abs(min(sentiment_polarity, 0)), 2),
        "overall": "positive" if sentiment_polarity > 0.05 else "negative" if sentiment_polarity < -0.05 else "neutral"
    }

    # --- Emotion approximation ---
    emotions = {
        "joy": round(max(sentiment_polarity, 0) * 0.8, 2),
        "sadness": round(abs(min(sentiment_polarity, 0)) * 0.7, 2),
        "anger": round(abs(min(sentiment_polarity, 0)) * 0.3, 2),
        "fear": round(abs(min(sentiment_polarity, 0)) * 0.2, 2),
        "surprise": round(abs(sentiment_polarity) * 0.5, 2)
    }

    # --- Readability score ---
    try:
        score = flesch_reading_ease(text)
        level = "Easy" if score > 80 else "Good" if score > 60 else "Challenging" if score > 30 else "Very Hard"
    except Exception:
        score, level = 0, "N/A"

    readability = {"score": round(score, 2), "level": level}

    # --- Keyword extraction ---
    rake = Rake()
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases()[:10]
    # Also count simple word frequencies
    counts = Counter([w.lower() for w in words if len(w) > 3])
    common_words = [w for w, _ in counts.most_common(10)]

    keywords = ranked_phrases if ranked_phrases else common_words

    # --- Summary (short version for dashboard) ---
    summary = blob.sentences[:2]
    summary_text = " ".join(str(s) for s in summary)

    # --- Response JSON ---
    return {
        "filename": filename,
        "word_count": word_count,
        "sentiment": sentiment,
        "emotions": emotions,
        "readability": readability,
        "keywords": keywords,
        "summary": summary_text
    }
