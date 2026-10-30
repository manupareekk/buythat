"""Build an offline catalog from object crops in samples/ so the app runs without Channel3."""
import json
import random
from pathlib import Path

from PIL import Image

from buythat.grounding import GENERIC, crop, detect

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "mock"
BRANDS = ["Northline", "Atlas & Co", "Kestrel", "Fieldhouse", "Marlow", "Juniper"]
RETAILERS = ["example-outfitters.com", "example-home.com", "example-sport.com"]


def main():
    random.seed(7)
    items = []
    for src in sorted((ROOT / "samples").glob("*.jpg")):
        img = Image.open(src).convert("RGB")
        for i, b in enumerate(detect(img, GENERIC, threshold=0.2)[:4]):
            name = f"{src.stem}_{i}.jpg"
            crop(img, b).save(OUT / "img" / name, quality=90)
            items.append(dict(
                id=f"mock-{src.stem}-{i}", title=f"{random.choice(BRANDS)} {b.label.title()}",
                brand="", image=f"img/{name}", price=round(random.uniform(19, 249), 2),
                currency="USD", retailer=random.choice(RETAILERS), url="https://example.com/",
            ))
    (OUT / "catalog.json").write_text(json.dumps(items, indent=1))
    print(len(items), "items:", sorted({d["title"].split()[-1] for d in items}))


if __name__ == "__main__":
    main()
