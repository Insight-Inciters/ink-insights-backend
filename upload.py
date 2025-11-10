from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import nltk
import numpy as np
import re, os, json, tempfile, threading
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# === NLTK setup ===
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# === Globals ===
_MODEL = None
_MODE_NAME = "none"

# === Safe lightweight GloVe loader ===
def get_glove_model():
    """Try to load small GloVe; fallback to None if memory too low."""
    global _MODEL, _MODE_NAME
    if _MODEL is not None:
        return _MODEL

    try:
        import gensim.downloader as api
        api.BASE_DIR = os.path.join(tempfile.gettempdir(), "gensim-data")
        os.makedirs(api.BASE_DIR, exist_ok=True)

        print("🔄 Loading GloVe Twitter 25D (Render-safe)...")
        _MODEL = api.load("glove-twitter-25", return_path=False)
        _MODE_NAME = "GloVe"
        print("✅ GloVe loaded successfully (25D).")
    except Exception as e:
        print(f"⚠️ GloVe load failed: {e}")
        _MODEL = None
        _MODE_NAME = "TFIDF"
    return _MODEL

# === Fallback: TF-IDF Embedding ===
def tfidf_embeddings(tokens):
    """Generate 2D pseudo-semantic vectors using TF-IDF if GloVe not available."""
    if not tokens:
        return np.zeros((0, 2))
    text = [" ".join(tokens)]
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(text).toarray()
    pca = PCA(n_components=2)
    return pca.fit_transform(X.T)

# === FastAPI setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "https://inkinsights.vercel.app",
        "https://ink-insights-backend.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Schema ===
class TextRequest(BaseModel):
    text: str
    filename: str = "document.txt"

# === Core Helpers ===
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s,.!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# === Sentiment Analysis ===
def analyze_sentiment(text: str):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)
    mood = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"

    pos = max(0, np.log1p(max(polarity, 0)) * 100)
    neg = max(0, np.log1p(max(-polarity, 0)) * 100)
    neu = 100 - (pos + neg)
    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "positive": round(pos, 2),
        "neutral": round(neu, 2),
        "negative": round(neg, 2),
        "mood": mood,
    }

def calc_readability(text: str):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    if not sentences or not words:
        return 0.0
    num_syllables = sum(sum(ch in "aeiouy" for ch in w.lower()) for w in words)
    score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (num_syllables / len(words))
    return max(min(score, 100), 0)

def extract_keywords(text: str, top_n=10):
    blob = TextBlob(text.lower())
    words = [w for w in blob.words if w.isalpha() and w not in stopwords.words("english")]
    freq = Counter(words)
    return [{"token": w, "count": c} for w, c in freq.most_common(top_n)]

def emotion_scores(text: str):
    lex = {
        "anger": ["angry", "mad", "furious", "rage"],
        "sadness": ["sad", "down", "cry", "unhappy"],
        "fear": ["fear", "scared", "afraid", "nervous"],
        "joy": ["happy", "joy", "love", "smile"],
        "surprise": ["surprised", "shocked", "amazed"],
        "disgust": ["disgusted", "gross", "vile"],
    }
    tokens = [w.lower() for w in word_tokenize(text)]
    total = len(tokens) or 1
    counts = {k: round(sum(tokens.count(w) for w in v) / total, 3) for k, v in lex.items()}
    return {"breakdown": counts}

def generate_summary(text: str):
    sents = sent_tokenize(text)
    if len(sents) <= 3:
        return text
    ranked = sorted(sents, key=lambda s: abs(TextBlob(s).sentiment.polarity), reverse=True)
    return " ".join(ranked[:3])

# === Thematic Clustering ===
def find_themes(text: str):
    model = get_glove_model()
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha() and w not in stopwords.words("english")]
    if not tokens:
        return [], []

    kept, vectors = [], []
    if model:
        for w in dict.fromkeys(tokens):
            if w in model:
                kept.append(w)
                vectors.append(model[w])
    else:
        coords = tfidf_embeddings(tokens)
        kept = list(dict.fromkeys(tokens))
        vectors = coords.tolist()

    if not vectors:
        return [], []

    X = np.array(vectors)
    if X.shape[0] < 2:
        return [], []

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    n_clusters = min(6, max(2, len(set(kept)) // 5))
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = km.fit_predict(coords)

    points = [{"x": float(coords[i, 0]), "y": float(coords[i, 1]),
               "label": kept[i], "cluster": int(labels[i])} for i in range(len(kept))]

    clusters = []
    for cid in range(n_clusters):
        words = [p["label"] for p in points if p["cluster"] == cid]
        clusters.append({
            "id": cid,
            "label": ", ".join(words[:10]),
            "count": len(words),
            "x": float(np.mean([p["x"] for p in points if p["cluster"] == cid])),
            "y": float(np.mean([p["y"] for p in points if p["cluster"] == cid])),
        })

    return points, clusters

# === Routes ===
@app.get("/")
async def home():
    return {"message": f"Ink Insights backend running with {_MODE_NAME} embeddings."}

@app.post("/analyze")
async def analyze_text(req: TextRequest):
    text = clean_text(req.text)
    if not text:
        return {"error": "Empty text"}

    sentiment = analyze_sentiment(text)
    readability = calc_readability(text)
    emotions = emotion_scores(text)
    keywords = extract_keywords(text)
    summary = generate_summary(text)
    themes_points, clusters = find_themes(text)

    emotion_tone = emotions["breakdown"].get("joy", 0) - emotions["breakdown"].get("sadness", 0)
    blended_tone = round(0.7 * (sentiment["positive"] - sentiment["negative"]) + 0.3 * (emotion_tone * 100), 2)

    result = {
        "filename": req.filename,
        "sentiment": sentiment,
        "readability": readability,
        "emotions": emotions,
        "keywords": {"list": keywords},
        "summary": summary,
        "themes": {"points": themes_points, "clusters": clusters},
        "blended_tone": blended_tone,
        "mode": _MODE_NAME,
    }

    with open("last_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result

# === Background preload ===
threading.Thread(target=get_glove_model, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Ink Insights backend on port {port} ...")
    uvicorn.run("upload:app", host="0.0.0.0", port=port, reload=False)
