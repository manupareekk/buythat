# BuyThat

Point your phone at the TV, hold the button, say **"buy that jacket"**. BuyThat finds the TV in the
photo, figures out which thing on screen you meant, finds that exact product across retailers, and
checks you out with PayPal.

```
phone photo + voice
  -> screen.py      find the TV quad, perspective-rectify to 1280x720, suppress moire
  -> speech.py      faster-whisper (base.en, int8) -> {"target": "red jacket", "max_price": 500}
  -> grounding.py   OWLv2 open-vocabulary detection of the spoken noun on the rectified frame
  -> retrieval.py   Channel3 image search (crop) + text search (noun), merged,
                    re-ranked locally with SigLIP: 0.7*cos(crop, photo) + 0.3*cos(noun, photo)
  -> paypal.py      PayPal Orders v2 create -> PayPal button approval -> capture (sandbox)
```

All the vision and speech models run on your own machine (CUDA, Apple MPS or CPU). The only
cloud calls are to Channel3, for the product catalog, and to PayPal, for payment.

## Run it

You need Python 3.10+, PayPal sandbox app credentials (developer.paypal.com -> Apps & Credentials
-> Sandbox) and a Channel3 API key (trychannel3.com).

```
git clone https://github.com/manupareekk/buythat.git
cd buythat
export PAYPAL_CLIENT_ID=your_sandbox_client_id
export PAYPAL_CLIENT_SECRET=your_sandbox_secret
export CHANNEL3_API_KEY=your_channel3_key
./run_local.sh
```

Open the printed `https://<your-ip>:8443` on a phone on the same Wi-Fi. Phones only allow the
camera and mic over HTTPS, so the script makes a self-signed certificate; accept the warning
once. On a laptop you can use "use a photo instead" and type the command.

Without `CHANNEL3_API_KEY` the app runs on the offline catalog in `mock/`. Pay with a sandbox
personal account (developer.paypal.com -> Sandbox accounts).

## Tests and eval

```
python -m pytest -q tests
python -m scripts.eval_retrieval 3
```

`eval_retrieval` takes real Channel3 products, renders each one on a synthetic TV photo
(perspective, bezel, moire, sensor noise), runs the full pipeline and reports where the original
product ranks. Results are in [EVAL.md](EVAL.md).

## License

MIT
