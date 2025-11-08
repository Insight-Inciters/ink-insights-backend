from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk
import numpy as np
import re
import os
import json
from functools import lru_cache
import threading
import time

# === New imports for semantic embeddings ===
from gensim.downloader import load
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ======= NLTK setup =======
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# ======= Cached GloVe load =======
@lru_cache(maxsize=1)
def get_glove_model():
    print("🔄 Loading compact GloVe embeddings (first time only)...")
    try:
        model = load("glove-twitter-25")  # ⚡ Faster & smaller
        print("✅ GloVe model loaded successfully.")
        return model
    except Exception as e:
        print(f"⚠️ GloVe load failed: {e}")
        return None

_MODEL = get_glove_model()

# ======= FastAPI setup =======
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

# ======= Schema =======
class TextRequest(BaseModel):
    text: str
    filename: str = "document.txt"


# ======= Progress tracking =======
progress = {"status": "idle", "percent": 0, "message": ""}

def update_progress(p, msg=""):
    global progress
    progress["percent"] = int(p)
    progress["message"] = msg
    progress["status"] = "running"

@app.get("/progress")
async def get_progress():
    return progress


# ======= Core helpers =======
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s,.!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ======= Sentiment Analysis =======
def analyze_sentiment(text: str):
    blob = TextBlob(text)
    sentences = blob.sentences or [blob]

    pos_sum, neg_sum, weight_sum = 0.0, 0.0, 0.0
    negation_pattern = re.compile(r"\b(no|not|never|none|n't|hardly|rarely)\b", re.I)

    for sent in sentences:
        s_text = str(sent).lower()
        p = sent.sentiment.polarity
        w = max(1, len(s_text.split()) / 5)

        if negation_pattern.search(s_text):
            p = -p * 0.8

        if p > 0.05:
            pos_sum += p * w
        elif p < -0.05:
            neg_sum += abs(p) * w
        weight_sum += w

    n = max(weight_sum, 1)
    pos = (pos_sum / n) * 100
    neg = (neg_sum / n) * 100
    pos = np.log1p(pos) * 20
    neg = np.log1p(neg) * 20
    pos, neg = min(pos, 100), min(neg, 100)
    neu = max(0.0, 100 - (pos + neg))

    polarity = round(blob.sentiment.polarity, 3)
    subjectivity = round(blob.sentiment.subjectivity, 3)

    if polarity > 0.1:
        mood = "positive"
    elif polarity < -0.1:
        mood = "negative"
    else:
        mood = "neutral"

    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "positive": round(pos, 2),
        "neutral": round(neu, 2),
        "negative": round(neg, 2),
        "mood": mood,
    }


def count_syllables(word):
    vowels = "aeiouy"
    count = 0
    prev = False
    for ch in word.lower():
        if ch in vowels and not prev:
            count += 1
        prev = ch in vowels
    if word.endswith("e"):
        count = max(1, count - 1)
    return max(count, 1)


def calc_readability(text: str):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    num_sent, num_words = len(sentences), len(words)
    if num_sent == 0 or num_words == 0:
        return 0.0
    num_syllables = sum(count_syllables(w) for w in words)
    score = 206.835 - 1.015 * (num_words / num_sent) - 84.6 * (num_syllables / num_words)
    return max(min(score, 100), 0)


def extract_keywords(text: str, top_n=10):
    blob = TextBlob(text.lower())
    words = [w for w in blob.words if w.isalpha() and w not in stopwords.words("english")]
    if not words:
        return []

    freq = Counter(words)
    for np in blob.noun_phrases:
        tokens = np.split()
        if len(tokens) > 1:
            for word in tokens:
                if word in freq:
                    freq[word] += 2

    if len(words) < 5:
        for w in freq:
            freq[w] = 1

    return [{"token": w, "count": c} for w, c in freq.most_common(top_n)]


def calculate_keyness(text: str, reference_text: str = None, top_n: int = 10):
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    if not tokens:
        return []

    freq_user = Counter(tokens)
    if reference_text:
        ref_tokens = [w.lower() for w in word_tokenize(reference_text) if w.isalpha()]
    else:
        ref_tokens = stopwords.words("english")
    freq_ref = Counter(ref_tokens)

    vocab = set(freq_user) | set(freq_ref)
    total_user = sum(freq_user.values())
    total_ref = sum(freq_ref.values())

    keyness_scores = []
    for word in vocab:
        O1 = freq_user.get(word, 0)
        O2 = freq_ref.get(word, 0)
        E1 = total_user * (O1 + O2) / (total_user + total_ref + 1e-9)
        E2 = total_ref * (O1 + O2) / (total_user + total_ref + 1e-9)

        def ll(obs, exp):
            return 0 if obs == 0 else obs * np.log(obs / exp)

        llr = 2 * (ll(O1, E1) + ll(O2, E2))
        keyness_scores.append((word, llr, O1))

    top = sorted(keyness_scores, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"token": w, "keyness": round(k, 3), "count": c} for w, k, c in top]


def emotion_scores(text: str):
    emotion_lex = {
        "anger": ["angry", "mad", "furious", "rage", "irritated", "annoyed"],
        "sadness": ["sad", "down", "depressed", "lonely", "unhappy", "tearful"],
        "fear": ["fear", "scared", "terrified", "afraid", "nervous", "worried"],
        "joy": ["happy", "joy", "delight", "love", "pleasure", "excited", "glad"],
        "surprise": ["surprised", "shocked", "amazed", "astonished"],
        "disgust": ["disgusted", "repulsed", "gross", "sickened", "vile"],
    }
    tokens = [w.lower() for w in word_tokenize(text)]
    emo_counts = {k: 0 for k in emotion_lex}
    total = len(tokens)
    for emo, words in emotion_lex.items():
        emo_counts[emo] = sum(tokens.count(w) for w in words)
    if total > 0:
        emo_counts = {k: round(v / total, 3) for k, v in emo_counts.items()}
    return {"breakdown": emo_counts}


def generate_summary(text: str):
    sentences = sent_tokenize(text)
    if len(sentences) <= 4:
        return " ".join(sentences)
    ranked = sorted(sentences, key=lambda s: abs(TextBlob(s).sentiment.polarity), reverse=True)
    return " ".join(ranked[:4]).strip()


def find_themes(text: str, max_words=150):
    model = _MODEL
    if not model:
        return [], []

    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha() and w not in stopwords.words("english")]
    unique_tokens = list(dict.fromkeys(tokens))[:max_words]

    vectors, kept = [], []
    for w in unique_tokens:
        if w in model:
            vectors.append(model[w])
            kept.append(w)

    if not vectors:
        return [], []

    X = np.array(vectors)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    n_clusters = max(2, min(6, len(kept) // 10))
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = km.fit_predict(X)

    points = [
        {"x": float(coords[i, 0]), "y": float(coords[i, 1]), "label": kept[i],
         "cluster": int(labels[i]), "count": tokens.count(kept[i])}
        for i in range(len(kept))
    ]

    clusters = []
    for cid in range(n_clusters):
        cluster_words = [p["label"] for p in points if p["cluster"] == cid]
        clusters.append({
            "id": cid,
            "label": ", ".join(cluster_words[:10]),
            "count": len(cluster_words),
            "x": float(np.mean([p["x"] for p in points if p["cluster"] == cid])),
            "y": float(np.mean([p["y"] for p in points if p["cluster"] == cid])),
        })

    # Normalize
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    norm = lambda arr: [(v - min(arr)) / (max(arr) - min(arr) + 1e-9) for v in arr]
    for i, p in enumerate(points):
        p["x"] = round(norm(xs)[i], 4)
        p["y"] = round(norm(ys)[i], 4)

    return points, clusters


@app.get("/")
async def home():
    return {"message": "Ink Insights backend is working!"}


@app.post("/analyze")
async def analyze_text(req: TextRequest):
    global progress
    progress = {"status": "running", "percent": 0, "message": "Starting analysis"}

    text = clean_text(req.text)
    if not text:
        progress = {"status": "error", "message": "Empty text"}
        return {"error": "Empty text"}

    update_progress(10, "Analyzing sentiment")
    sentiment = analyze_sentiment(text)

    update_progress(25, "Extracting emotions")
    emotions = emotion_scores(text)

    update_progress(40, "Extracting keywords")
    keywords = extract_keywords(text)

    update_progress(55, "Computing keyness")
    keyness = calculate_keyness(text)

    update_progress(70, "Clustering semantic themes")
    themes_points, clusters = find_themes(text)

    update_progress(90, "Finalizing summary")
    readability = calc_readability(text)
    summary = generate_summary(text)

    word_count = len(word_tokenize(text))
    sentence_count = len(sent_tokenize(text))
    emotion_tone = emotions["breakdown"].get("joy", 0) - emotions["breakdown"].get("sadness", 0)
    blended_tone = round(0.7 * (sentiment["positive"] - sentiment["negative"]) + 0.3 * (emotion_tone * 100), 2)

    data = {
        "filename": req.filename,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "readability": readability,
        "sentiment": sentiment,
        "blended_tone": blended_tone,
        "keywords": {"list": keywords},
        "keyness": {"list": keyness},
        "emotions": emotions,
        "summary": summary,
        "themes": {"points": themes_points, "clusters": clusters or []},
    }

    try:
        with open("last_report.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    update_progress(100, "Analysis complete")
    progress["status"] = "done"

    return data

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("upload:app", host="0.0.0.0", port=port)
