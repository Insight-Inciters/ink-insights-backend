from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import re
import math
import os
import json
from collections import Counter, defaultdict

# ===== NLP deps (only things you already listed) =====
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake
from textblob import TextBlob
import textstat

# ---- NLTK lazy bootstrap (safe for Vercel) ----
NLTK_DATA_DIR = "/tmp/nltk_data"
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

def _ensure_nltk():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", download_dir=NLTK_DATA_DIR)
    try:
        sent_tokenize("test")
    except LookupError:
        nltk.download("punkt", download_dir=NLTK_DATA_DIR)
    try:
        SentimentIntensityAnalyzer()
    except Exception:
        nltk.download("vader_lexicon", download_dir=NLTK_DATA_DIR)

_ensure_nltk()

# ===== FastAPI =====
app = FastAPI(title="Ink Insights API")

# CORS: allow your frontend(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://inkinsights.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeIn(BaseModel):
    text: str
    filename: str = "unknown.txt"

def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _count_words(text: str) -> int:
    tokens = re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE)
    return len(tokens)

def _read_time_minutes(words: int, wpm: int = 200) -> int:
    return max(1, math.ceil(words / max(1, wpm)))

def _keyword_stats(text: str, top_n: int = 20) -> Dict[str, Any]:
    # RAKE keyphrases
    rake = Rake(min_length=1, max_length=3)
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases()[:top_n]

    # Unigram frequencies
    sw = set(stopwords.words("english"))
    tokens = [t.lower() for t in re.findall(r"\b[\w’'-]+\b", text)]
    tokens = [t for t in tokens if t not in sw and not re.fullmatch(r"[-’']+", t)]
    freqs = Counter(tokens)

    top_items = freqs.most_common(top_n)
    unique = len(set(tokens))
    top_word = top_items[0][0] if top_items else None

    # N-grams (bigrams/trigrams)
    bigrams = Counter(zip(tokens, tokens[1:])).most_common(10)
    trigrams = Counter(zip(tokens, tokens[1:], tokens[2:])).most_common(10)

    return {
        "unique": unique,
        "top": top_word,
        "list": [{"token": k, "count": v} for k, v in top_items],
        "ngrams": {
            "bigrams": [{"token": " ".join(k), "count": v} for k, v in bigrams],
            "trigrams": [{"token": " ".join(k), "count": v} for k, v in trigrams],
        },
    }

def _sentiment_block(text: str) -> Dict[str, Any]:
    sia = SentimentIntensityAnalyzer()

    sentences = sent_tokenize(text) or [text]
    timeline = []
    pos = neu = neg = 0
    for i, s in enumerate(sentences):
        scores = sia.polarity_scores(s)
        timeline.append({"i": i, "compound": scores["compound"]})
        # choose bucket with highest score (excl compound)
        buckets = {k: scores[k] for k in ("pos", "neu", "neg")}
        b = max(buckets, key=buckets.get)
        if b == "pos": pos += 1
        elif b == "neu": neu += 1
        else: neg += 1

    total = max(1, len(sentences))
    pos_pct = round(pos * 100 / total)
    neu_pct = round(neu * 100 / total)
    neg_pct = 100 - pos_pct - neu_pct

    return {
        "pos": pos_pct,
        "neu": neu_pct,
        "neg": neg_pct,
        "timeline": timeline,  # array of {i, compound}
    }

def _readability(text: str) -> Dict[str, Any]:
    try:
        score = float(textstat.flesch_reading_ease(text))
    except Exception:
        score = 0.0
    level = (
        "Very Easy" if score >= 90 else
        "Easy" if score >= 80 else
        "Fairly Easy" if score >= 70 else
        "Good" if score >= 60 else
        "Fairly Difficult" if score >= 50 else
        "Difficult" if score >= 30 else
        "Very Confusing"
    )
    return {"score": round(score, 1), "level": level}

def _summary(text: str, max_sents: int = 3) -> str:
    # Super-light extractive summary: rank sentences by TextBlob polarity magnitude
    sentences = sent_tokenize(text)
    if not sentences:
        return ""
    scored = sorted(
        ((abs(TextBlob(s).sentiment.polarity), i, s) for i, s in enumerate(sentences)),
        reverse=True,
    )
    picked = sorted(scored[:max_sents], key=lambda x: x[1])
    return " ".join(s for _, _, s in picked)

def _emotion_stub(text: str) -> Dict[str, Any]:
    """
    Minimal emotion heuristic (no extra deps):
    we'll count a small seed lexicon per emotion.
    Replace later with NRC if you add 'nrclex'.
    """
    seeds = {
        "joy": {"joy", "happy", "delight", "smile", "pleasure", "love", "cheer"},
        "anger": {"anger", "angry", "rage", "furious", "irritate", "annoy"},
        "sadness": {"sad", "sorrow", "grief", "cry", "tears", "lonely"},
        "fear": {"fear", "scared", "afraid", "terror", "panic", "worry"},
        "surprise": {"surprise", "astonish", "amaze", "sudden", "shock"},
    }
    counts = {k: 0 for k in seeds}
    toks = [t.lower() for t in re.findall(r"\b[\w’'-]+\b", text)]
    for t in toks:
        for emo, lex in seeds.items():
            if t in lex:
                counts[emo] += 1
    # normalize to percents
    total = sum(counts.values()) or 1
    breakdown = {k: round(v * 100 / total) for k, v in counts.items()}
    dominant = max(counts, key=counts.get) if total > 1 else "neutral"
    # naive emotional arc: rolling average of compound
    sia = SentimentIntensityAnalyzer()
    sents = sent_tokenize(text) or [text]
    arc = []
    window = 5
    vals = [sia.polarity_scores(s)["compound"] for s in sents]
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        avg = sum(vals[lo:i+1]) / (i - lo + 1)
        arc.append({"i": i, "value": round(avg, 3)})
    return {"dominant": dominant, "breakdown": breakdown, "arc": arc}

def _themes_stub(keywords: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cheap 'themes' using RAKE phrases + frequent tokens.
    """
    phrases = keywords.get("list", [])
    top_terms = [x["token"] for x in phrases[:6]]
    clusters = max(1, math.ceil(len(phrases) / 5))
    return {"clusters": clusters, "top_theme": top_terms or ["—"]}

@app.get("/")
async def home():
    return {"message": "Ink Insights backend is working!"}

@app.post("/analyze")
async def analyze(req: AnalyzeIn):
    text = _normalize_ws(req.text or "")
    words = _count_words(text)

    # Safety & limits
    if words == 0:
        return JSONResponse({"error": "Empty text"}, status_code=400)
    if words > 5000:
        return JSONResponse({"error": "Word limit exceeded (max 5000)"}, status_code=400)

    # Compute
    kw = _keyword_stats(text)
    se = _sentiment_block(text)
    rd = _readability(text)
    em = _emotion_stub(text)
    th = _themes_stub(kw)
    read_time = _read_time_minutes(words)

    # One-paragraph extractive summary
    summary = _summary(text)

    # Build preview text for highlighting on Features page
    preview_text = (text[:8000] + "…") if len(text) > 8000 else text

    payload = {
        "meta": {
            "filename": req.filename,
            "words": words,
            "read_time_min": read_time,
        },
        "summary": summary,
        "readability": rd,
        "keywords": {
            **kw,
            "previewText": preview_text
        },
        "sentiment": se,
        "emotions": em,
        "themes": th,
        # nothing is stored server-side; response only
        "storage": {"persisted": False}
    }

    return JSONResponse(payload)

# Optional ping to “delete” — no-op (you said you won’t store)
@app.post("/delete")
async def delete_endpoint():
    # Nothing to delete on server; reply success for frontend UX.
    return JSONResponse({"ok": True, "server_deleted": False})
