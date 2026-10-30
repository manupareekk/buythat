# BuyThat evaluation

All numbers below were measured with `scripts/eval_retrieval.py` against the live Channel3 catalog. They are not estimates.

## Setup

- **Products:** 30 real Channel3 products, 3 each from 10 queries (jackets, sneakers, headphones, trench coats, ...). The set is frozen in `eval/set.json` so every run is comparable.
- **"TV photos":** each product's catalog image is shown on a synthetic TV (`scripts/make_tv_photo.py`) with a random perspective tilt, a bezel, a cluttered room background, sensor noise, and moiré stripes. Then it goes through the same pipeline as the app: screen detect/rectify, OWLv2 grounding of the spoken noun, Channel3 image+text search, and the SigLIP re-rank.
- **Hit:** the exact Channel3 product ID that was on screen appears at rank *k*. The **dedup** rows also count a different listing whose image is a near-copy (SigLIP image cosine > 0.95, e.g. the same jacket sold by two retailers) as a hit.
- **Hardware:** an 8-vCPU cloud VM, CPU only. Apple-silicon Macs use MPS and are faster.

## Results (OWLv2 at 768 px, tight crop; the default config)

| Stage | Metric |
|---|---|
| TV screen found and rectified | 30/30 (100%) |
| Target grounded by OWLv2 | 30/30 (100%) |

| Ranking | top-1 | top-5 | top-12 |
|---|---|---|---|
| Channel3 alone | 17% | 33% | 47% |
| + SigLIP re-rank | 37% | 47% | 47% |
| + SigLIP re-rank, dedup | **53%** | 57% | 57% |

Median end-to-end latency: 12.0 s per query on the CPU VM (grounding is about 5 s of that, and the search plus image fetch is most of the rest).

## What the ablations showed

**Re-ranking more than doubles top-1** (17% → 37% exact). It nearly reaches the recall ceiling: once the right product is among the 12 candidates, SigLIP puts it first about 80% of the time.

**Recall is the bottleneck, and the cause is the capture domain gap.** Same 30 products, Channel3 image-only search, recall@12:

| Query image | recall@12 |
|---|---|
| Clean catalog photo (upper bound) | 87% |
| Rectified TV crop (pipeline) | 40% |
| Rectified crop, no moiré filter | 33% |
| Whole rectified frame, no grounding | 23% |

Grounding and moiré suppression both pay for themselves. The remaining roughly 45-point gap is what you lose by photographing a screen.

**Crop padding:** a tight box (0%) reached 47% recall@12, versus 37% for 8% padding and 30% for 20%. The default is now 0%. Color normalization (gray-world + CLAHE), letterboxing onto white, unsharp masking, and an FFT notch de-moiré did not help (33–43%). A light Gaussian blur scored 50%, but that is within noise.

**OWLv2 input size:** 768 vs the native 960 gave the same top-1 (40% vs 40% exact, 47% vs 50% dedup) and cut median latency from 15.2 s to 13.9 s. It is about 1.8x faster in grounding alone, so 768 is the default (`BUYTHAT_OWL_SIZE`).

## Limitations

- n = 30. One hit is about 3 points, so differences under about 7 points are noise.
- Channel3 results vary slightly between identical calls.
- The TV photos are synthetic. Real phone-of-TV captures add glare, rolling-shutter bands and motion blur. A real-capture set is the next step.
- Products are catalog images, not frames from shows. On real TV content the item may be partly hidden or poorly lit, and it may not be in Channel3's index at all.

## Reproduce

```
export CHANNEL3_API_KEY=...
python -m scripts.eval_retrieval 3
BUYTHAT_OWL_SIZE=960 python -m scripts.eval_retrieval 3
```
