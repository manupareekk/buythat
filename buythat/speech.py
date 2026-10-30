"""Transcribe the spoken command and pull out what the user wants to buy."""
import re
from functools import lru_cache

FILLERS = {"um", "uh", "like", "please", "now", "right", "the", "a", "an", "one", "thing", "that", "this", "those", "these", "it",
           "i", "i'm", "im", "i'd", "i'll", "me", "my", "you", "can", "could", "some", "of", "want", "go", "ahead"}
TRIGGER = re.compile(r"\b(?:buy|get|order|purchase|find|want|shop|grab)\b(?:\s+me)?\s+(?:that|this|those|these|the)?\s*(.*)", re.I)
PRICE = re.compile(r"\b(?:under|below|less than|max|up to)\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
SIZE = re.compile(r"\bin (?:a )?size\s+(\w+)|\bsize\s+(\w+)", re.I)


@lru_cache(maxsize=1)
def _whisper():
    from faster_whisper import WhisperModel
    return WhisperModel("base.en", device="cpu", compute_type="int8")


def transcribe(path: str) -> str:
    segments, _ = _whisper().transcribe(path, beam_size=1, vad_filter=True, language="en",
                                        initial_prompt="Buy that jacket. Get me those sneakers under $100.")
    return " ".join(s.text.strip() for s in segments).strip()


def parse_command(text: str) -> dict:
    """'buy that green jacket under 200' -> {'target': 'green jacket', 'max_price': 200.0}"""
    t = text.strip().rstrip(".!?")
    max_price = None
    if m := PRICE.search(t):
        max_price = float(m.group(1))
        t = PRICE.sub("", t)
    size = None
    if m := SIZE.search(t):
        size = m.group(1) or m.group(2)
        t = SIZE.sub("", t)
    m = TRIGGER.search(t)
    rest = m.group(1) if m else t
    rest = re.split(r"\b(?:for me|on (?:the )?(?:tv|screen)|he'?s wearing|she'?s wearing|they'?re wearing|from (?:the )?(?:show|ad))\b", rest, flags=re.I)[0]
    words = [w for w in re.findall(r"[a-zA-Z'-]+", rest) if w.lower() not in FILLERS]
    target = " ".join(words).lower() or None
    return {"text": text, "target": target, "max_price": max_price, "size": size}
