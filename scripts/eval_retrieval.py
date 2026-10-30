"""Benchmark: can we recover the exact catalog product after it's shown on a TV and photographed?

For each query we pull real products from Channel3, render a product photo onto a synthetic TV
(perspective, bezel, moire, room noise), run the full pipeline (screen rectify -> OWLv2 ground ->
Channel3 image search -> SigLIP rerank) and check where the original product id lands.
Writes eval/out/results.json and prints top-1 / top-5 with and without rerank.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from buythat import grounding, retrieval, screen
from scripts.make_tv_photo import tv_photo

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "out"
DUP_COS = 0.95
QUERIES = ["red ski jacket", "white sneakers", "leather handbag", "aviator sunglasses", "denim jacket",
           "wool beanie", "floor lamp", "over-ear headphones", "running shoes", "trench coat"]


def _same_product_ids(src: Image.Image, ranked) -> set[str]:
    """Listings whose photo is a near-duplicate of the source photo: the same product sold under another id."""
    keep, embs = retrieval._product_embs(list(ranked))
    if embs is None:
        return set()
    sims = (embs @ retrieval._img_embed([src.resize((224, 224))]).T).squeeze(1).tolist()
    return {c.id for c, s in zip(keep, sims) if s >= DUP_COS}


def main(per_query: int = 3):
    from channel3_sdk import Channel3
    OUT.mkdir(parents=True, exist_ok=True)
    client = Channel3(api_key=os.environ["CHANNEL3_API_KEY"])
    random.seed(0)
    rows = []
    for q in QUERIES:
        prods = [p for p in (retrieval._from_c3(x) for x in list(client.products.search(query=q, limit=10).items)) if p]
        for k, p in enumerate(random.sample(prods, min(per_query, len(prods)))):
            src = retrieval._fetch(p.image)
            if src is None:
                continue
            canvas = Image.new("RGB", (1280, 720), (235, 235, 235))
            src.thumbnail((620, 620))
            canvas.paste(src, (random.randint(80, 1200 - src.width), (720 - src.height) // 2))
            raw = OUT / f"{q.replace(' ', '_')}_{k}_src.jpg"
            canvas.save(raw)
            tv = OUT / f"{q.replace(' ', '_')}_{k}_tv.jpg"
            tv_photo(str(raw), str(tv), seed=hash((q, k)) % 1000)
            t = time.perf_counter()
            scr = screen.find_screen(cv2.imread(str(tv)))
            frame = Image.fromarray(cv2.cvtColor(scr.frame, cv2.COLOR_BGR2RGB))
            noun = q.split()[-1]
            box, _ = grounding.ground(frame, noun)
            crop = grounding.crop(frame, box) if box else frame
            cands = retrieval._channel3_search(crop, noun, None, 12)
            ranked = retrieval.rerank(crop, noun, list(cands))
            ids_raw = [c.id for c in cands]
            ids_rr = [c.id for c in ranked]
            same_rr = _same_product_ids(src, ranked)
            rows.append(dict(query=q, product=p.title, id=p.id, screen_found=scr.found, grounded=box is not None,
                             rank_channel3=ids_raw.index(p.id) + 1 if p.id in ids_raw else None,
                             rank_rerank=ids_rr.index(p.id) + 1 if p.id in ids_rr else None,
                             rank_rerank_dedup=next((i + 1 for i, c in enumerate(ranked) if c.id == p.id or c.id in same_rr), None),
                             latency_s=round(time.perf_counter() - t, 2)))
            print(json.dumps(rows[-1]), flush=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))
    n = len(rows)

    def at(key, k):
        return sum(1 for r in rows if r[key] and r[key] <= k) / n
    print(f"\nn={n}  screen_found={sum(r['screen_found'] for r in rows)/n:.2f}  grounded={sum(r['grounded'] for r in rows)/n:.2f}")
    for key in ("rank_channel3", "rank_rerank", "rank_rerank_dedup"):
        print(f"{key:18s} top1={at(key,1):.2f} top5={at(key,5):.2f} top12={at(key,12):.2f}")
    print(f"median latency {np.median([r['latency_s'] for r in rows]):.1f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
