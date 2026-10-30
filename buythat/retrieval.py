"""Product lookup: Channel3 image + text search, re-ranked locally with SigLIP."""
import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import httpx
import torch
from PIL import Image

from .device import DEVICE

SIGLIP_ID = "google/siglip-base-patch16-224"
MOCK_CATALOG = Path(__file__).resolve().parent.parent / "mock" / "catalog.json"


@dataclass
class Product:
    id: str
    title: str
    brand: str
    image: str
    price: float
    currency: str
    retailer: str
    url: str
    score: float = 0.0
    signals: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@lru_cache(maxsize=1)
def _siglip():
    from transformers import AutoModel, AutoProcessor
    return AutoProcessor.from_pretrained(SIGLIP_ID), AutoModel.from_pretrained(SIGLIP_ID).eval().to(DEVICE)


def _img_embed(images: list[Image.Image]) -> torch.Tensor:
    proc, model = _siglip()
    with torch.inference_mode():
        out = model.get_image_features(**proc(images=images, return_tensors="pt").to(DEVICE))
    out = out if isinstance(out, torch.Tensor) else out.pooler_output
    return torch.nn.functional.normalize(out, dim=-1).cpu()


def _txt_embed(texts: list[str]) -> torch.Tensor:
    proc, model = _siglip()
    with torch.inference_mode():
        out = model.get_text_features(**proc(text=texts, padding="max_length", return_tensors="pt").to(DEVICE))
    out = out if isinstance(out, torch.Tensor) else out.pooler_output
    return torch.nn.functional.normalize(out, dim=-1).cpu()


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _from_c3(p) -> Product | None:
    offers = [o for o in (p.offers or []) if o.price is not None]
    if not offers:
        return None
    o = min(offers, key=lambda o: o.price.price)
    imgs = p.images or []
    main = next((i for i in imgs if i.is_main_image), imgs[0] if imgs else None)
    return Product(
        id=p.id, title=p.title, brand=(p.brands[0].name if p.brands else ""),
        image=(main.cleaned_url or main.url) if main else "",
        price=float(o.price.price), currency=o.price.currency or "USD",
        retailer=o.domain or "", url=o.url,
    )


def _c3_query(kw: dict, max_price: float | None, limit: int) -> list[Product]:
    from channel3_sdk import Channel3
    client = Channel3(api_key=os.environ["CHANNEL3_API_KEY"])
    if max_price:
        kw = {**kw, "filters": {"price": {"max_price": max_price}}}
    try:
        items = client.products.search(**kw, limit=limit).items or []
    except Exception as e:  # one path failing should not sink the other
        print("channel3 error:", e)
        return []
    return [p for p in (_from_c3(x) for x in items[:limit]) if p]


def text_search(query: str | None, max_price: float | None = None, limit: int = 12) -> list[Product]:
    if not query or not os.environ.get("CHANNEL3_API_KEY"):
        return []
    return _c3_query({"query": query}, max_price, limit)


def _merge(*lists: list[Product]) -> list[Product]:
    seen, out = set(), []
    for lst in lists:
        for p in lst:
            if p.id not in seen:
                seen.add(p.id)
                out.append(p)
    return out


def _channel3_search(crop: Image.Image, query: str | None, max_price: float | None, limit: int,
                     text_hits: list[Product] | None = None) -> list[Product]:
    if text_hits is None:
        with ThreadPoolExecutor(1) as ex:
            fut = ex.submit(text_search, query, max_price, limit)
            img_hits = _c3_query({"base64image": _b64(crop)}, max_price, limit)
            text_hits = fut.result()
    else:
        img_hits = _c3_query({"base64image": _b64(crop)}, max_price, limit)
    return _merge(img_hits, text_hits)


def _mock_search(query: str | None, max_price: float | None, limit: int) -> list[Product]:
    items = [Product(**d) for d in json.loads(MOCK_CATALOG.read_text())]
    if max_price:
        items = [p for p in items if p.price <= max_price]
    return items[: limit * 2]


def _fetch(url: str) -> Image.Image | None:
    try:
        if url.startswith("http"):
            r = httpx.get(url, timeout=8, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 BuyThat"})
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        path = Path(url) if Path(url).is_absolute() else MOCK_CATALOG.parent / url
        return Image.open(path).convert("RGB")
    except Exception:
        return None


_EMB: dict[str, torch.Tensor] = {}


def _product_embs(products: list[Product]) -> tuple[list[Product], torch.Tensor | None]:
    todo = [p for p in products if p.image not in _EMB]
    with ThreadPoolExecutor(8) as ex:
        thumbs = list(ex.map(lambda p: _fetch(p.image), todo))
    ok = [(p, t) for p, t in zip(todo, thumbs) if t is not None]
    if ok:
        for (p, _), e in zip(ok, _img_embed([t.resize((224, 224)) for _, t in ok])):
            _EMB[p.image] = e
    keep = [p for p in products if p.image in _EMB]
    return keep, (torch.stack([_EMB[p.image] for p in keep]) if keep else None)


def _z(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else x * 0


def rerank(crop: Image.Image, query: str | None, products: list[Product], w_img: float = 0.7,
           fusion: str | None = None) -> list[Product]:
    """Score = w*sim(crop, product photo) + (1-w)*sim(spoken noun, product photo).

    fusion="z" standardizes each signal over the candidate set first; SigLIP text-image cosines live
    on a much smaller scale than image-image ones, so raw sums let the visual term drown the words."""
    fusion = fusion or os.environ.get("BUYTHAT_FUSION", "raw")
    if not products:
        return []
    keep, prod_e = _product_embs(products)
    if not keep:
        return products
    crop_e = _img_embed([crop])
    img_sim = (prod_e @ crop_e.T).squeeze(1)
    if query:
        txt_sim = (prod_e @ _txt_embed([f"a photo of a {query}"]).T).squeeze(1)
    else:
        txt_sim = torch.zeros_like(img_sim)
    wi = w_img if query else 1.0
    if fusion == "z":
        scores = wi * _z(img_sim) + (1 - wi) * _z(txt_sim)
    else:
        scores = wi * img_sim + (1 - wi) * txt_sim
    for p, s, si, st in zip(keep, scores.tolist(), img_sim.tolist(), txt_sim.tolist()):
        p.score, p.signals = round(s, 4), {"visual": round(si, 4), "text": round(st, 4)}
    return sorted(keep, key=lambda p: p.score, reverse=True)


def search(crop: Image.Image, query: str | None, max_price: float | None = None, limit: int = 12,
           text_hits: list[Product] | None = None) -> tuple[list[Product], str]:
    if os.environ.get("CHANNEL3_API_KEY"):
        cands, source = _channel3_search(crop, query, max_price, limit, text_hits), "channel3"
    else:
        cands, source = _mock_search(query, max_price, limit), "mock"
    return rerank(crop, query, cands), source
