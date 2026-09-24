#!/usr/bin/env python3
"""
Compare checkpoints on text you choose.

Where evaluate.py runs a fixed benchmark, this is for poking at a specific
sentence: give it any text and any number of models, get one wav per model plus
optional transcripts.

    python compare.py "मैं आज office जा रहा हूँ"
    python compare.py "..." --voice ta_hinglish --lang ta
    python compare.py "..." --models base Tharshan/model-a ./local-ckpt
    python compare.py --file sentences.txt --voice te --lang te

"base" resolves to ai4bharat/IndicF5. Every model is built with the same
architecture and vocabulary as the first non-base model, so only the weights
differ.

    pip install torch torchaudio torchdiffeq x-transformers vocos \
                librosa soundfile transformers
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

BASE_MODEL = "ai4bharat/IndicF5"
BASE_PREFIX = "ema_model._orig_mod."
DEFAULT = "Tharshan/indicf5_hindi-english_code_switch"


def load_base(reference, device):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    Cls, Cfg = type(reference), type(reference.config)
    m = Cls(Cfg(), vocab_path=os.path.join(reference.asset_dir, "vocab.txt"),
            asset_dir=reference.asset_dir)
    sd = load_file(hf_hub_download(BASE_MODEL, filename="model.safetensors"))
    sd = {k[len(BASE_PREFIX):]: v for k, v in sd.items()
          if k.startswith(BASE_PREFIX)}
    target = set(m.model.state_dict())
    missing, _ = m.model.load_state_dict(
        {k: v for k, v in sd.items() if k in target}, strict=False)
    loaded = len(target) - len(missing)
    print(f"  base: {loaded}/{len(target)} tensors")
    if loaded < 0.9 * len(target):
        sys.exit("base weights failed to load — the key prefix may have changed")
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--file", help="a file of sentences, one per line")
    ap.add_argument("--models", nargs="+", default=["base", DEFAULT],
                    help="repo ids or local paths; 'base' means ai4bharat/IndicF5")
    ap.add_argument("--voice", default=None, help="bundled voice key")
    ap.add_argument("--ref", help="your own reference wav")
    ap.add_argument("--ref-text", help="exact transcript of --ref")
    ap.add_argument("--lang", help="Whisper language code; omit to skip ASR")
    ap.add_argument("--out-dir", default="compare_out")
    ap.add_argument("--nfe-step", type=int, default=32)
    args = ap.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            texts = [l.strip() for l in f if l.strip()]
    elif args.text:
        texts = [args.text]
    else:
        sys.exit("give some text, or --file")

    if bool(args.ref) != bool(args.ref_text):
        sys.exit("--ref and --ref-text go together")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    from transformers import AutoModel

    # load the non-base models first; base needs one as an architecture template
    models, reference = {}, None
    for spec in args.models:
        if spec == "base":
            continue
        print(f"loading {spec} ...")
        models[spec] = AutoModel.from_pretrained(
            spec, trust_remote_code=True).to(device).eval()
        reference = reference or models[spec]

    if "base" in args.models:
        if reference is None:
            sys.exit("'base' needs at least one other model to borrow "
                     "architecture and vocabulary from")
        print("loading base ...")
        models = {"base": load_base(reference, device), **models}

    voices = reference.voices()
    if args.ref:
        ref_audio, ref_text = args.ref, args.ref_text
    else:
        key = args.voice or next(iter(voices))
        if key not in voices:
            sys.exit(f"unknown voice {key!r}. Available: {list(voices)}")
        ref_audio, ref_text = reference.voice(key)
        print(f"voice: {key}")

    transcribe = None
    if args.lang:
        print("loading whisper ...")
        import librosa
        from transformers import (WhisperForConditionalGeneration,
                                  WhisperProcessor)
        proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        dtype = torch.float16 if device == "cuda" else torch.float32
        asr = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3", torch_dtype=dtype).to(device).eval()

        def transcribe(audio, sr):
            a = audio.mean(axis=1) if audio.ndim > 1 else audio
            if sr != 16000:
                a = librosa.resample(a.astype("float32"), orig_sr=sr,
                                     target_sr=16000)
            feats = proc(a, sampling_rate=16000,
                         return_tensors="pt").input_features
            ids = asr.generate(feats.to(device, dtype), language=args.lang,
                               task="transcribe", max_new_tokens=200)
            return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    os.makedirs(args.out_dir, exist_ok=True)

    for i, text in enumerate(texts):
        print(f"\n[{i}] {text}")
        for name, mdl in models.items():
            audio, sr = mdl.generate(text, ref_audio=ref_audio,
                                     ref_text=ref_text, nfe_step=args.nfe_step)
            slug = name.replace("/", "_").replace(".", "")
            path = os.path.join(args.out_dir, f"{i}_{slug}.wav")
            sf.write(path, np.asarray(audio, np.float32), sr)
            line = f"  {name:45s} {path}"
            if transcribe:
                line += f"\n    -> {transcribe(audio, sr)}"
            print(line)

    print(f"\nwrote {len(texts) * len(models)} files to {args.out_dir}/")


if __name__ == "__main__":
    main()