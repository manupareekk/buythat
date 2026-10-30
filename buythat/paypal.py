"""PayPal Orders v2 (sandbox): create an order for the matched product, capture after approval."""
import os
import time
import uuid

import httpx

BASE = {"sandbox": "https://api-m.sandbox.paypal.com", "live": "https://api-m.paypal.com"}


class PayPal:
    def __init__(self):
        self.client_id = os.environ.get("PAYPAL_CLIENT_ID", "")
        self.secret = os.environ.get("PAYPAL_CLIENT_SECRET", "")
        self.env = os.environ.get("PAYPAL_ENV", "sandbox")
        self.base = BASE[self.env]
        self._token, self._exp = None, 0.0

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.secret)

    def token(self) -> str:
        if self._token and time.time() < self._exp - 60:
            return self._token
        r = httpx.post(f"{self.base}/v1/oauth2/token", auth=(self.client_id, self.secret),
                       data={"grant_type": "client_credentials"}, timeout=20)
        r.raise_for_status()
        j = r.json()
        self._token, self._exp = j["access_token"], time.time() + j["expires_in"]
        return self._token

    def _h(self) -> dict:
        return {"Authorization": f"Bearer {self.token()}", "Content-Type": "application/json",
                "PayPal-Request-Id": str(uuid.uuid4())}

    def create_order(self, product: dict, source_frame_id: str | None = None) -> dict:
        price = f"{float(product['price']):.2f}"
        cur = product.get("currency") or "USD"
        body = {
            "intent": "CAPTURE",
            "purchase_units": [{
                "reference_id": product["id"][:256],
                "description": f"BuyThat: {product['title']}"[:127],
                "custom_id": (source_frame_id or "")[:127],
                "soft_descriptor": "BUYTHAT TV",
                "amount": {"currency_code": cur, "value": price,
                           "breakdown": {"item_total": {"currency_code": cur, "value": price}}},
                "items": [{
                    "name": product["title"][:127], "quantity": "1", "category": "PHYSICAL_GOODS",
                    "unit_amount": {"currency_code": cur, "value": price},
                    "url": product.get("url") or None, "image_url": product.get("image") or None,
                    "sku": product["id"][:127],
                }],
            }],
        }
        for it in body["purchase_units"][0]["items"]:
            for k in ("url", "image_url"):
                if not it[k] or not str(it[k]).startswith("https://"):
                    it.pop(k)
        r = httpx.post(f"{self.base}/v2/checkout/orders", headers=self._h(), json=body, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"create_order {r.status_code}: {r.text}")
        return r.json()

    def capture(self, order_id: str) -> dict:
        r = httpx.post(f"{self.base}/v2/checkout/orders/{order_id}/capture", headers=self._h(), timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"capture {r.status_code}: {r.text}")
        return r.json()
