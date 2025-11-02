from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from textblob import TextBlob
from rake_nltk import Rake
from collections import Counter
from textstat import flesch_reading_ease
import nltk

# --- Required NLTK data ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = FastAPI()

# --- Enable CORS for frontend access ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow everything for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UploadRequest(BaseModel):
    text: str
    filename: str

@app.get("/")
def home():
    return {"message": "Ink Insights upload endpoint active ✅"}

@app.post("/upload")
async def upload_text(data: UploadRequest):
    """
    Main NLP processing endpoint.
    Receives text + filename and returns analysis results.
    """
    text = data.text.strip()
    filename = data.filename

    # --- Word count ---
    words = text.split()
    word_count = len(words)

    # --- Sentiment ---
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment = {
        "positive": round(max(sentiment_polarity, 0), 2),
        "neutral": round(1 - abs(sentiment_polarity), 2),
        "negative": round(abs(min(sentiment_polarity, 0)), 2),
        "overall": "positive" if sentiment_polarity > 0.05 else "negative" if sentiment_polarity < -0.05 else "neutral"
    }

    # --- Emotions (approximate mapping) ---
    emotions = {
        "joy": round(max(sentiment_polarity, 0) * 0.8, 2),
        "sadness": round(abs(min(sentiment_polarity, 0)) * 0.7, 2),
        "anger": round(abs(min(sentiment_polarity, 0)) * 0.3, 2),
        "fear": round(abs(min(sentiment_polarity, 0)) * 0.2, 2),
        "surprise": round(abs(sentiment_polarity) * 0.5, 2)
    }

    # --- Readability ---
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

    # --- Keywords extraction ---
    rake = Rake()
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases()[:10]

    # simple word frequency fallback
    counts = Counter([w.lower() for w in words if len(w) > 3])
    common_words = [w for w, _ in counts.most_common(10)]
    keywords = ranked_phrases if ranked_phrases else common_words

    # --- Quick summary ---
    summary_sentences = blob.sentences[:2]
    summary_text = " ".join(str(s) for s in summary_sentences)

    # --- Final response ---
    return {
        "filename": filename,
        "word_count": word_count,
        "sentiment": sentiment,
        "emotions": emotions,
        "readability": readability,
        "keywords": keywords,
        "summary": summary_text
    }

@app.delete("/delete")
async def delete_data():
    """
    Since this backend is stateless (doesn't save files),
    we just confirm deletion for the frontend.
    """
    return {"message": "✅ Temporary data cleared. Nothing stored on server."}
