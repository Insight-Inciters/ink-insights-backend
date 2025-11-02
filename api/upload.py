from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from textblob import TextBlob
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import re

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return {"message": "Ink Insights NLP backend is live!"}


@app.post("/upload")
async def upload(request: Request):
    """Main endpoint that receives user text and returns NLP feature results"""
    data = await request.json()
    text = data.get("text", "")
    filename = data.get("filename", "unknown.txt")

    if not text.strip():
        return JSONResponse({"error": "No text provided"}, status_code=400)

    # -------------------------
    # 1️⃣ Keyword & Frequency
    # -------------------------
    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    freq = Counter(words)
    top_keywords = freq.most_common(10)
    keyword_data = {
        "unique": len(freq),
        "top_keywords": [{"word": w, "count": c} for w, c in top_keywords],
        "suggestions": "Try diversifying your word choices to improve clarity and avoid repetition."
    }

    # -------------------------
    # 2️⃣ Themes & Clusters
    # -------------------------
    vectorizer = CountVectorizer(stop_words="english", max_features=6)
    X = vectorizer.fit_transform([text])
    terms = vectorizer.get_feature_names_out()
    theme_data = {
        "themes": list(terms),
        "suggestions": "Focus on connecting your central ideas and expanding supporting details."
    }

    # -------------------------
    # 3️⃣ Sentiment Analysis
    # -------------------------
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        tone = "positive"
    elif polarity < -0.1:
        tone = "negative"
    else:
        tone = "neutral"
    sentiment_data = {
        "positive": round(max(polarity, 0) * 100, 2),
        "neutral": round((1 - abs(polarity)) * 50, 2),
        "negative": round(max(-polarity, 0) * 100, 2),
        "summary": f"Your overall tone is {tone}.",
        "suggestions": f"Balance emotional tone for improved readability and engagement."
    }

    # -------------------------
    # 4️⃣ Emotion Detection
    # -------------------------
    emotion_words = {
        "joy": ["happy", "joy", "delight", "pleasure"],
        "anger": ["angry", "hate", "furious", "rage"],
        "sadness": ["sad", "cry", "tear", "grief"],
        "fear": ["fear", "afraid", "scared", "worry"],
        "surprise": ["surprise", "shock", "amazed", "astonished"]
    }
    emotion_scores = {emo: sum(w in text.lower() for w in lst) for emo, lst in emotion_words.items()}
    dominant = max(emotion_scores, key=emotion_scores.get) if emotion_scores else "neutral"
    emotion_data = {
        "emotions": emotion_scores,
        "dominant": dominant,
        "suggestions": f"Dominant emotion detected: {dominant}. Adjust tone to match your intent."
    }

    # -------------------------
    # 5️⃣ Overall Summary
    # -------------------------
    summary = {
        "filename": filename,
        "word_count": len(words),
        "readability_score": 70 + round(polarity * 10, 2),
        "keywords": keyword_data,
        "themes": theme_data,
        "sentiment": sentiment_data,
        "emotions": emotion_data,
    }

    return JSONResponse(summary)
