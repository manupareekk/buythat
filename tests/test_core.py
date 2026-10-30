import cv2
import numpy as np
import pytest

from buythat import screen, speech
from buythat.paypal import PayPal
from scripts.make_tv_photo import tv_photo


@pytest.mark.parametrize("text,target,price,size", [
    ("buy that jacket", "jacket", None, None),
    ("Buy that red jacket he's wearing.", "red jacket", None, None),
    ("get me those sneakers under $120", "sneakers", 120.0, None),
    ("I want that lamp on the TV", "lamp", None, None),
    ("buy the hoodie in size medium", "hoodie", None, "medium"),
    ("I'm a red jacket under $500.", "red jacket", 500.0, None),
    ("Can you get me that lamp", "lamp", None, None),
])
def test_parse_command(text, target, price, size):
    c = speech.parse_command(text)
    assert (c["target"], c["max_price"], c["size"]) == (target, price, size)


@pytest.mark.parametrize("seed", range(5))
def test_screen_rectify_recovers_corners(tmp_path, seed):
    out = tmp_path / "tv.jpg"
    truth = tv_photo("samples/coco_000000000785.jpg", str(out), seed=seed)
    r = screen.find_screen(cv2.imread(str(out)))
    assert r.found and r.frame.shape[:2] == (screen.OUT_H, screen.OUT_W)
    err = np.linalg.norm(np.array(r.quad) - truth, axis=1).max()
    assert err < 0.05 * 900, err


def test_screen_not_found_on_blank():
    assert not screen.find_screen(np.full((600, 800, 3), 128, np.uint8)).found


def test_paypal_order_body_drops_non_https(monkeypatch):
    sent = {}

    class R:
        status_code = 201
        def json(self): return {"id": "X", "status": "CREATED"}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(json)
        return R()

    pp = PayPal()
    monkeypatch.setattr(pp, "token", lambda: "t")
    monkeypatch.setattr("buythat.paypal.httpx.post", fake_post)
    pp.create_order({"id": "p1", "title": "Jacket", "price": 49.5, "currency": "USD",
                     "url": "http://x", "image": "https://img/x.jpg"})
    item = sent["purchase_units"][0]["items"][0]
    assert sent["intent"] == "CAPTURE" and item["unit_amount"]["value"] == "49.50"
    assert "url" not in item and item["image_url"] == "https://img/x.jpg"
