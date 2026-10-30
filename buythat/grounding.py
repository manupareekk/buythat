"""Resolve "that <noun>" to a box on the frame with an open-vocabulary detector (OWLv2)."""
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from .device import DEVICE

MODEL_ID = "google/owlv2-base-patch16-ensemble"
# 768 instead of the native 960: ~1.8x faster on CPU, boxes agree with 960 at IoU>0.7 on our eval frames.
SIZE = int(os.environ.get("BUYTHAT_OWL_SIZE", "768"))
GENERIC = ["jacket", "shirt", "dress", "shoes", "sneakers", "handbag", "watch", "sunglasses", "hat",
           "lamp", "chair", "sofa", "headphones", "phone", "laptop", "bottle", "mug", "backpack", "car"]


@dataclass
class Box:
    label: str
    score: float
    xyxy: tuple[int, int, int, int]


@lru_cache(maxsize=1)
def _model():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    torch.set_num_threads(max(1, torch.get_num_threads()))
    proc = Owlv2Processor.from_pretrained(MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).eval().to(DEVICE)
    return proc, model


def detect(image: Image.Image, phrases: list[str], threshold: float = 0.12) -> list[Box]:
    proc, model = _model()
    queries = [f"a photo of a {p}" for p in phrases]
    inputs = proc(text=[queries], images=image, return_tensors="pt",
                  size={"height": SIZE, "width": SIZE}).to(DEVICE)
    with torch.inference_mode():
        out = model(**inputs, interpolate_pos_encoding=SIZE != 960)
    # OWLv2 pads the image to a square; boxes are relative to the padded size.
    side = max(image.size)
    res = proc.image_processor.post_process_object_detection(out, threshold=threshold, target_sizes=torch.tensor([[side, side]], device=DEVICE))[0]
    boxes = []
    for score, label, b in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
        x0, y0, x1, y1 = (int(max(0, v)) for v in b)
        x1, y1 = min(x1, image.width), min(y1, image.height)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        boxes.append(Box(phrases[label], float(score), (x0, y0, x1, y1)))
    return sorted(boxes, key=lambda b: b.score, reverse=True)


def _centrality(b: Box, w: int, h: int) -> float:
    cx, cy = (b.xyxy[0] + b.xyxy[2]) / 2 / w, (b.xyxy[1] + b.xyxy[3]) / 2 / h
    return 1 - min(1.0, np.hypot(cx - 0.5, cy - 0.5) / 0.71)


def ground(image: Image.Image, target: str | None) -> tuple[Box | None, list[Box]]:
    """Pick the box the user means. With a noun, best match wins; with bare "that", the most
    confident, central, reasonably sized product is chosen."""
    w, h = image.size
    if target:
        boxes = detect(image, [target])
        if boxes:
            return boxes[0], boxes
    boxes = detect(image, GENERIC, threshold=0.15)
    if not boxes:
        return None, []

    def rank(b: Box) -> float:
        area = (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1]) / (w * h)
        return b.score + 0.25 * _centrality(b, w, h) + 0.15 * min(area / 0.15, 1.0)

    return max(boxes, key=rank), boxes


def crop(image: Image.Image, box: Box, pad: float = 0.0) -> Image.Image:
    x0, y0, x1, y1 = box.xyxy
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return image.crop((max(0, x0 - dx), max(0, y0 - dy), min(image.width, x1 + dx), min(image.height, y1 + dy)))
