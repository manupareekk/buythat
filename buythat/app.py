"""BuyThat server: phone photo of a TV + spoken command -> matched products -> PayPal checkout."""
import io
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import grounding, retrieval, screen, speech
from .paypal import PayPal

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="BuyThat")
paypal = PayPal()
PRODUCTS: dict[str, dict] = {}
POOL = ThreadPoolExecutor(4)


def _jpeg_b64(img: Image.Image, max_side: int = 640) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    return "data:image/jpeg;base64," + retrieval._b64(img)


@app.get("/api/config")
def config():
    return {"paypal_client_id": paypal.client_id if paypal.configured else None,
            "paypal_env": paypal.env,
            "catalog": "channel3" if os.environ.get("CHANNEL3_API_KEY") else "mock"}


@app.post("/api/buy")
async def buy(image: UploadFile = File(...), audio: UploadFile | None = File(None),
              text: str | None = Form(None), rectify: bool = Form(True)):
    timings, t0 = {}, time.perf_counter()

    def lap(name):
        nonlocal t0
        timings[name] = round((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()

    raw = np.frombuffer(await image.read(), np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "could not decode image")
    scr = screen.find_screen(bgr) if rectify else screen.ScreenResult(bgr, None, False)
    frame = Image.fromarray(cv2.cvtColor(scr.frame, cv2.COLOR_BGR2RGB))
    lap("screen")

    transcript = text or ""
    if audio is not None and not text:
        suffix = Path(audio.filename or "a.webm").suffix or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(await audio.read())
            f.flush()
            transcript = speech.transcribe(f.name)
    cmd = speech.parse_command(transcript)
    lap("speech")

    # Text search only needs the spoken noun, so it runs on the network while OWLv2 grounds.
    text_fut = POOL.submit(retrieval.text_search, cmd["target"], cmd["max_price"], 8)
    box, boxes = grounding.ground(frame, cmd["target"])
    lap("grounding")
    if box is None:
        return {"ok": False, "reason": "nothing matching on screen", "command": cmd,
                "screen": {"found": scr.found, "quad": scr.quad}, "frame": _jpeg_b64(frame),
                "timings_ms": timings}
    crop = grounding.crop(frame, box)
    text_hits = text_fut.result() if cmd["target"] else None
    products, source = retrieval.search(crop, cmd["target"] or box.label, cmd["max_price"], limit=8,
                                        text_hits=text_hits)
    lap("retrieval")

    out = []
    for p in products[:8]:
        d = p.to_dict()
        if not d["image"].startswith("http"):
            d["image"] = f"/mock/{d['image']}"
        PRODUCTS[p.id] = d
        out.append(d)
    return {"ok": True, "command": cmd, "frame_id": uuid.uuid4().hex[:12],
            "screen": {"found": scr.found, "quad": scr.quad},
            "frame": _jpeg_b64(frame), "frame_size": frame.size,
            "box": {"label": box.label, "score": round(box.score, 3), "xyxy": box.xyxy},
            "candidates": [{"label": b.label, "score": round(b.score, 3), "xyxy": b.xyxy} for b in boxes[:6]],
            "crop": _jpeg_b64(crop, 320), "source": source, "products": out, "timings_ms": timings}


class OrderReq(BaseModel):
    product_id: str
    frame_id: str | None = None


@app.post("/api/orders")
def create_order(req: OrderReq):
    if not paypal.configured:
        raise HTTPException(503, "PayPal not configured")
    p = PRODUCTS.get(req.product_id)
    if not p:
        raise HTTPException(404, "unknown product; search again")
    order = paypal.create_order(p, req.frame_id)
    return {"id": order["id"], "status": order["status"]}


@app.post("/api/orders/{order_id}/capture")
def capture(order_id: str):
    res = paypal.capture(order_id)
    cap = res["purchase_units"][0]["payments"]["captures"][0]
    return {"id": res["id"], "status": res["status"], "capture_id": cap["id"],
            "amount": cap["amount"], "payer": res.get("payer", {}).get("email_address")}


app.mount("/mock", StaticFiles(directory=Path(__file__).resolve().parent.parent / "mock"), name="mock")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
