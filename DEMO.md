# Demo video script (target 2:40, hard limit 3:00)

| time | shot | voiceover |
|---|---|---|
| 0:00-0:12 | Couch, a show playing on the TV, someone on screen in a red jacket | "You see something on TV you want. Today that means pausing, guessing the brand, and twenty minutes of searching." |
| 0:12-0:30 | Phone raised at the TV at an angle, hold the button: "buy that red jacket under five hundred" | "With BuyThat, you point and say it." |
| 0:30-0:55 | Screen recording of the phone: rectified frame, yellow box on the jacket, trace panel | "The phone finds the TV and straightens the picture. Whisper turns what you said into a target and a price limit, and OWLv2, an open-vocabulary detector, finds the jacket you meant. All of this runs on my laptop." |
| 0:55-1:20 | Results grid with the query crop and score line | "Channel3 searches millions of real products using the crop and the words. Then SigLIP re-ranks them on my machine, mostly by how the item looks, partly by what you said." |
| 1:20-1:50 | Tap the top result, PayPal button, sandbox login, approve, PAID receipt | "One tap on PayPal: we create an order, you approve it in PayPal, and we capture the payment. The receipt keeps the order and capture IDs." |
| 1:50-2:15 | Second take: "buy that lamp", a different scene | "It works on any product it can see. You don't have to say the brand." |
| 2:15-2:40 | EVAL.md numbers on screen, then the architecture line from the README | "We measured this: top-1 and top-5 exact-product recovery from photos of a TV screen. The code is open, MIT licensed, and runs on your own machine." |

## Devpost description (draft)

**BuyThat: point your phone at the TV, say "buy that jacket", pay with PayPal.**

Shopping from TV is still a manual search: you pause, screenshot, guess the brand and scroll.
BuyThat turns an angled phone photo of a screen plus one spoken sentence into a PayPal checkout
for the exact product.

How it works:
- Screen geometry: OpenCV finds the TV quadrilateral, perspective-rectifies the frame and
  suppresses moire.
- Speech: faster-whisper runs locally and pulls out the target, a price cap and a size.
- Grounding: OWLv2 open-vocabulary detection maps "that red jacket" to a box.
- Retrieval: Channel3 image and text search, merged, then re-ranked locally with SigLIP
  (0.7 visual + 0.3 text).
- Commerce: PayPal Orders v2 create -> approve with the PayPal button -> capture, with
  idempotency keys and the source frame ID in `custom_id`.

PayPal + AI: the model's output (which product you meant, at what price limit) becomes a PayPal
order, so the AI decides what gets paid for and PayPal settles it.
