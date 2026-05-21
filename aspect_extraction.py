"""
aspect_extraction.py
====================
Unified aspect extraction module for the entire project.

Provides:
  - spaCy-based sentence splitting (with regex fallback)
  - A single, comprehensive keyword dictionary for aspect detection
  - A scored keyword-matching function shared by all pipelines

Every file that needs aspect extraction should import from here:
    from aspect_extraction import split_into_sentences, detect_aspect, ASPECT_KEYWORDS
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Try loading spaCy once ──────────────────────────────────────────────────
try:
    import spacy as _spacy
    try:
        _nlp = _spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading en_core_web_sm…")
        from spacy.cli import download
        download("en_core_web_sm")
        _nlp = _spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except ImportError:
    _nlp = None
    SPACY_AVAILABLE = False
    logger.warning("spaCy not found. Falling back to regex sentence splitting.")


# ────────────────────────────────────────────────────────────────────────────
# CANONICAL ASPECT KEYWORD DICTIONARY
# ────────────────────────────────────────────────────────────────────────────
ASPECT_KEYWORDS = {
    "Camera": [
        "camera", "cameras", "photo", "photos", "picture", "pictures",
        "image quality", "picture quality", "clarity", "sharpness",
        "selfie", "front camera", "rear camera", "telephoto", "ultrawide",
        "portrait mode", "macro mode", "night mode", "hdr", "stabilization",
        "optical zoom", "digital zoom", "lens", "sensor", "exposure",
    ],
    "Battery": [
        "battery", "battery life", "battery backup", "charge", "charging",
        "charging speed", "fast charging", "wireless charging", "charger",
        "power adapter", "power consumption",
        "drains fast", "drains quickly", "loses charge", "dies quickly",
        "needs frequent charging", "screen on time", "sot",
    ],
    "Performance": [
        "performance", "speed", "lag", "slow", "fast", "smooth", "snappy",
        "responsive", "responsiveness", "multitasking", "freeze", "freezes",
        "stutter", "stutters", "hang", "hangs", "choppy",
        "processor", "chip", "gpu", "cpu",
        "a14", "a15", "a16", "a17", "m1", "m2", "m3", "m3 pro", "m3 max",
    ],
    "Display": [
        "display", "screen", "lcd", "oled", "super retina", "retina",
        "brightness", "contrast", "color accuracy", "colour accuracy",
        "resolution", "refresh rate", "120hz", "90hz", "60hz", "promotion",
        "vivid colors", "washed out", "sunlight visibility", "glare",
        "viewing angles", "pixel density",
    ],
    "Design & Build": [
        "design", "build", "build quality", "material", "aluminium", "metal",
        "durability", "durable", "sleek", "thin", "lightweight", "premium feel",
        "matte finish", "glossy finish", "scratch", "scratches easily",
        "look", "looks", "feel in hand", "aesthetics",
    ],
    "Software & OS": [
        "ios", "macos", "software", "system", "os", "update", "updates",
        "bug", "bugs", "crash", "crashes", "glitch", "glitches",
        "ui", "ux", "user interface", "notifications",
        "apple ecosystem", "continuity", "handoff", "airdrop", "icloud",
    ],
    "Audio": [
        "audio", "sound", "speaker", "speakers", "bass", "treble",
        "loudness", "microphone", "mic", "call quality", "voice clarity",
        "stereo speakers", "muffled audio", "tinny sound",
    ],
    "Connectivity": [
        "wifi", "wi-fi", "bluetooth", "network", "cellular", "5g", "lte",
        "signal", "connectivity", "hotspot", "airdrop disconnect",
        "network drops", "weak signal", "unstable wifi",
    ],
    "Storage": [
        "storage", "space", "memory", "ram",
        "32gb", "64gb", "128gb", "256gb", "512gb", "1tb",
        "running out of space", "not enough storage",
    ],
    "Price": [
        "price", "pricing", "cost", "expensive", "overpriced", "too costly",
        "cheap", "value for money", "worth the price", "not worth it",
        "premium pricing",
    ],
    "Heating / Thermals": [
        "heat", "heating", "heats", "heats up", "gets hot", "overheats",
        "thermal throttling", "hot while charging", "hot during gaming",
    ],
}

# Pre-build a flat keyword → aspect lookup for fast sentence-level matching
_KW_TO_ASPECT = {}
for _asp, _kws in ASPECT_KEYWORDS.items():
    for _kw in _kws:
        if _kw not in _KW_TO_ASPECT:  # first definition wins
            _KW_TO_ASPECT[_kw] = _asp

# A simple flat list for backward compatibility with sentiment_pipeline.py
ASPECT_LIST = list(ASPECT_KEYWORDS.keys())


# ────────────────────────────────────────────────────────────────────────────
# SENTENCE / CLAUSE SPLITTING
# ────────────────────────────────────────────────────────────────────────────

# Conjunctions that typically separate distinct thoughts/aspects mid-sentence.
_CONJUNCTION_RE = re.compile(
    r'(?i)\s*\b(?:but|however|though|although|yet|also|additionally|on the other hand|whereas|while)\b\s*'
)

# Coordinating conjunction / comma split — used only when parts belong to
# *different* aspect categories (avoids over-splitting).
_AND_RE = re.compile(r'(?i)\s*(?:,\s*and|\band\b)\s*')

# Minimum word count for a sub-clause to be kept (avoids tiny fragments).
_MIN_CLAUSE_WORDS = 3


def _aspect_category(text: str) -> str:
    """Return the top-scoring aspect for *text*, or 'Other'."""
    t = text.lower()
    best, best_score = "Other", 0
    for asp, kws in ASPECT_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in t)
        if score > best_score:
            best_score, best = score, asp
    return best


def _sub_split_on_contrast(sentence: str) -> list[str]:
    """
    Sub-split a single sentence on coordinating/contrast conjunctions.

    Stage 1 — contrast conjunctions (but, however, though, also, …):
    "The battery is amazing but the screen scratches easily."
    → ["The battery is amazing", "the screen scratches easily."]

    Stage 2 — coordinating "and" / comma+and:
    Only splits on "and" when the two parts contain *different* detected
    aspect categories.  This fixes cases like:
    "the camera is bad and the screen is good"
    → ["the camera is bad", "the screen is good"]
    without over-splitting sentences like "fast and smooth".

    After Stage 1 succeeds, Stage 2 is also applied to each sub-part
    so that "X and Y but Z" correctly splits all three clauses.

    Only keeps parts with >= _MIN_CLAUSE_WORDS words.
    If not enough meaningful parts remain, returns the original sentence.
    """
    def _try_and_split(part: str) -> list[str]:
        """Attempt to split a clause on 'and' when parts have different aspects."""
        and_parts = _AND_RE.split(part)
        and_parts = [p.strip() for p in and_parts if p.strip()]
        if len(and_parts) >= 2:
            meaningful_and = [p for p in and_parts if len(p.split()) >= _MIN_CLAUSE_WORDS]
            if len(meaningful_and) >= 2:
                categories = [_aspect_category(p) for p in meaningful_and]
                if len(set(categories)) > 1:  # different aspects → split
                    return meaningful_and
        # No and-split warranted — return as-is if long enough
        if len(part.split()) >= _MIN_CLAUSE_WORDS:
            return [part]
        return []

    # Stage 1: contrast split
    parts = _CONJUNCTION_RE.split(sentence)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 1:
        # No contrast split happened — try Stage 2 directly
        result = _try_and_split(sentence)
        return result if result else [sentence]

    # Contrast split succeeded — also try and-split on each sub-part
    clauses = []
    for part in parts:
        clauses.extend(_try_and_split(part))

    # If sub-splitting discarded too much, return the original
    return clauses if clauses else [sentence]


def _regex_split(text: str) -> list[str]:
    """Fallback regex-based clause splitter (no spaCy)."""
    parts = re.split(r'[.!?]', text)
    clauses = []
    for part in parts:
        part = part.strip()
        if part:
            clauses.extend(_sub_split_on_contrast(part))
    return clauses


def split_into_sentences(text: str) -> list[str]:
    """
    Two-stage sentence / clause splitter:

      1. **spaCy** splits on proper sentence boundaries
         (handles abbreviations, decimals, etc. correctly).
      2. Each sentence is then **sub-split on contrast words**
         (but, however, though, …) so that opposing sentiments
         within a single sentence get separated.

    Falls back to regex splitting if spaCy is unavailable.
    Returns a non-empty list (at minimum, the original text itself).
    """
    if not text or not isinstance(text, str) or not text.strip():
        return []

    if SPACY_AVAILABLE:
        doc = _nlp(text)
        raw_sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if not raw_sents:
            raw_sents = [text.strip()]

        # Stage 2: sub-split each sentence on contrast conjunctions
        clauses = []
        for sent in raw_sents:
            clauses.extend(_sub_split_on_contrast(sent))
        return clauses if clauses else [text.strip()]
    else:
        parts = _regex_split(text)
        return parts if parts else [text.strip()]


# ────────────────────────────────────────────────────────────────────────────
# ASPECT DETECTION
# ────────────────────────────────────────────────────────────────────────────
def detect_aspect(text: str) -> str:
    """
    Score-based aspect detector.

    Counts keyword hits per aspect category and returns the category
    with the highest score.  Falls back to "Other" if nothing matches.
    """
    t = text.lower()
    best_aspect = "Other"
    best_score = 0

    for aspect, kws in ASPECT_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in t)
        if score > best_score:
            best_score = score
            best_aspect = aspect

    return best_aspect

def detect_all_aspects(text: str) -> list[str]:
    """
    Return *every* aspect whose keywords appear in `text`.
    Useful for generating ABSA training rows (one row per aspect).
    """
    t = text.lower()
    found = []
    for aspect, kws in ASPECT_KEYWORDS.items():
        if any(kw in t for kw in kws):
            found.append(aspect)
    return found


# ────────────────────────────────────────────────────────────────────────────
# ASPECT-CONDITIONED SENTIMENT INFERENCE
# ────────────────────────────────────────────────────────────────────────────
def map_sentiment_label(raw_label: str) -> str:
    """
    Normalize a raw model label to Positive / Negative / Neutral.
    Handles star-based ("5 stars"), text-based ("POSITIVE"), and raw labels ("LABEL_0").
    """
    u = str(raw_label).upper().strip()
    
    # 1. Exact text matches (highest priority)
    if u in ["POSITIVE", "POS", "LABEL_2", "5", "4", "5 STARS", "4 STARS"]:
        return "Positive"
    if u in ["NEGATIVE", "NEG", "LABEL_0", "1", "2", "1 STAR", "2 STARS"]:
        return "Negative"
    if u in ["NEUTRAL", "NEU", "LABEL_1", "3", "3 STARS"]:
        return "Neutral"
        
    # 2. Substring matches (fallback for varied formatting)
    if any(x in u for x in ["POSITIVE", "POS"]): return "Positive"
    if any(x in u for x in ["NEGATIVE", "NEG"]): return "Negative"
    
    return "Neutral"


def run_aspect_sentiment(
    text: str,
    aspect: str,
    classifier,
    max_length: int = 512,
) -> tuple[str, float]:
    """
    Run aspect-conditioned sentiment inference.

    Passes (text, aspect) as a sentence-pair to the classifier so the
    model can attend to the specific aspect when determining sentiment.

    Returns (label, confidence) where label is Positive/Negative/Neutral.

    Falls back to text-only inference if sentence-pair input fails.
    """
    if classifier is None:
        return ("Neutral", 0.5)

    try:
        # Aspect-conditioned: pass text + aspect as sentence pair
        result = classifier(
            {"text": text, "text_pair": aspect},
            truncation=True,
            max_length=max_length,
        )
        if isinstance(result, list):
            result = result[0]
        return (map_sentiment_label(result["label"]), round(result["score"], 4))
    except Exception:
        # Fallback: text-only inference (for older pipeline API versions)
        try:
            result = classifier(text, truncation=True, max_length=max_length)[0]
            return (map_sentiment_label(result["label"]), round(result["score"], 4))
        except Exception:
            return ("Neutral", 0.5)
