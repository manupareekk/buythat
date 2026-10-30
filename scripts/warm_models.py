"""Download and load all models once (used at build time and before demos)."""
from buythat import grounding, retrieval, speech

if __name__ == "__main__":
    grounding._model()
    retrieval._siglip()
    speech._whisper()
    print("models ready")
