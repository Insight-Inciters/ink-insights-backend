# upload.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from textblob import TextBlob
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk
import numpy as np
import re



# ======= NLTK setup =======
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True) 
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ======= FastAPI setup =======
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://inkinsights.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======= Schema =======
class TextRequest(BaseModel):
    text: str
    filename: str = "document.txt"


# ======= Core helpers =======

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s,.!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def analyze_sentiment(text: str):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    mood = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
    return {
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "mood": mood,
    }


def count_syllables(word):
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev = False
    for ch in word:
        if ch in vowels and not prev:
            count += 1
        prev = ch in vowels
    if word.endswith("e"):
        count = max(1, count - 1)
    return max(count, 1)


def calc_readability(text: str):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    num_sent = len(sentences)
    num_words = len(words)
    num_syllables = sum(count_syllables(w) for w in words)
    if num_sent == 0 or num_words == 0:
        return 0.0
    score = 206.835 - 1.015 * (num_words / num_sent) - 84.6 * (num_syllables / num_words)
    return max(min(score, 100), 0)


def extract_keywords(text: str, top_n=10):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w.isalpha() and w not in stop_words]
    freq = Counter(words).most_common(top_n)
    return [{"token": w, "count": c} for w, c in freq]


def emotion_scores(text: str):
    emotion_lex = {
        "joy": ["happy", "joy", "delight", "love", "pleasure", "excited"],
        "anger": ["angry", "mad", "furious", "rage", "irritated"],
        "sadness": ["sad", "down", "depressed", "cry", "lonely"],
        "fear": ["fear", "scared", "terrified", "afraid", "nervous"],
        "surprise": ["surprised", "shocked", "amazed", "astonished"],
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
    if len(sentences) <= 3:
        return " ".join(sentences)
    blob = TextBlob(text)
    ranked = sorted(sentences, key=lambda s: abs(TextBlob(s).sentiment.polarity), reverse=True)
    summary = " ".join(ranked[:3])
    return summary.strip()


# ======= Semantic clustering helpers =======
_LEMM = WordNetLemmatizer()

def _normalize_token(t: str) -> str:
    t = re.sub(r"[^a-zA-Z']", "", t.lower())
    if not t:
        return t
    lemmas = [
        _LEMM.lemmatize(t, "n"),
        _LEMM.lemmatize(t, "v"),
        _LEMM.lemmatize(t, "a"),
        _LEMM.lemmatize(t, "r"),
    ]
    return min(lemmas, key=len)


def _wordnet_similarity(a: str, b: str) -> float:
    if a == b or not a or not b:
        return 1.0 if a == b and a else 0.0
    syn_a = wn.synsets(a)
    syn_b = wn.synsets(b)
    best = 0.0
    for sa in syn_a:
        for sb in syn_b:
            sim = sa.wup_similarity(sb)
            if sim and sim > best:
                best = sim
    return float(best or 0.0)


def _char_bigrams(s: str) -> set:
    return {s[i:i+2] for i in range(len(s)-1)} if len(s) > 1 else {s}


def _jaccard_chars(a: str, b: str) -> float:
    A, B = _char_bigrams(a), _char_bigrams(b)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def _semantic_similarity(a: str, b: str) -> float:
    wn_sim = _wordnet_similarity(a, b)
    if wn_sim >= 0.2:
        return wn_sim
    return max(wn_sim, _jaccard_chars(a, b))


def _classical_mds(D: np.ndarray, dim: int = 2) -> np.ndarray:
    """Perform classical multidimensional scaling (MDS)."""
    n = D.shape[0]
    if n == 0:
        return np.zeros((0, dim))

    # Centering matrix
    J = np.eye(n) - np.ones((n, n)) / n
    # Double-centering formula
    B = -0.5 * J @ (D ** 2) @ J

    # Eigen decomposition
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Keep only positive eigenvalues
    pos_mask = vals > 1e-9
    vals = vals[pos_mask][:dim]
    vecs = vecs[:, pos_mask][:, :dim]

    if len(vals) == 0:
        return np.zeros((n, dim))

    # Force vals to be 1D numeric array
    vals = np.array(vals, dtype=float).flatten()

    # Scale coordinates
    try:
        X = vecs[:, :len(vals)] @ np.diag(np.sqrt(vals))
    except Exception:
        # In case of mismatched shapes
        X = np.zeros((n, dim))

    # Pad to ensure 2D (x,y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] < dim:
        X = np.pad(X, ((0, 0), (0, dim - X.shape[1])), mode="constant")

    return np.nan_to_num(X)

def _greedy_clusters(points: np.ndarray, threshold: float = 0.3):
    """
    Simple greedy clustering of embedding vectors based on cosine similarity.
    Returns a list of cluster labels (integers).
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if points is None or len(points) == 0:
        return []

    sim = cosine_similarity(points)
    n = len(points)
    assigned = np.zeros(n, dtype=bool)
    clusters = [-1] * n
    cid = 0

    for i in range(n):
        if assigned[i]:
            continue
        clusters[i] = cid
        assigned[i] = True
        # Assign all points that are sufficiently similar
        for j in range(i + 1, n):
            if not assigned[j] and sim[i, j] >= 1 - threshold:
                clusters[j] = cid
                assigned[j] = True
        cid += 1

    # Safety: ensure list, not scalar
    if isinstance(clusters, int):
        clusters = [clusters]
    return clusters



def find_themes(keywords, max_points=60):
    kw = keywords[:max_points]
    if not kw:
        return []

    tokens = [_normalize_token(k["token"]) for k in kw]
    counts = [int(k["count"]) for k in kw]
    labels = [k["token"] for k in kw]

    n = len(tokens)
    S = np.zeros((n, n), dtype=float)
    for i in range(n):
        S[i, i] = 1.0
        for j in range(i + 1, n):
            sim = _semantic_similarity(tokens[i], tokens[j])
            S[i, j] = S[j, i] = sim

    D = 1.0 - np.clip(S, 0.0, 1.0)
    X = _classical_mds(D, dim=2)
    X = np.nan_to_num(X)

    clusters = _greedy_clusters(S, threshold=0.48)
    idx_to_cluster = {i: c for i, c in enumerate(clusters)}


    xs, ys = X[:, 0], X[:, 1]
    def _norm(arr):
        a, b = float(np.min(arr)), float(np.max(arr))
        return (arr - a) / (b - a + 1e-9)
    xs_n, ys_n = _norm(xs), _norm(ys)

    points = []
    for i in range(n):
        points.append({
            "x": round(float(xs_n[i]), 4),
            "y": round(float(ys_n[i]), 4),
            "label": labels[i],
            "cluster": int(idx_to_cluster.get(i, 0)),
            "count": counts[i],
        })
    return points


# ======= Routes =======
@app.get("/")
async def home():
    return {"message": "Ink Insights backend is working!"}


@app.post("/analyze")
async def analyze_text(req: TextRequest):
    text = clean_text(req.text)
    if not text:
        return {"error": "Empty text"}

    word_count = len(word_tokenize(text))
    sentence_count = len(sent_tokenize(text))
    sentiment = analyze_sentiment(text)
    readability = calc_readability(text)
    keywords = extract_keywords(text)
    emotions = emotion_scores(text)
    summary = generate_summary(text)
    themes = find_themes(keywords)

    return {
        "filename": req.filename,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "readability": readability,
        "sentiment": sentiment,
        "keywords": {"list": keywords},
        "emotions": emotions,
        "summary": summary,
        "themes": {"points": themes},
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # ✅ works both locally & on Render
    uvicorn.run("upload:app", host="0.0.0.0", port=port)
    
