#!/usr/bin/env python3
"""
Generate speech with the code-switch fine-tune.

    python infer.py "मैं आज office जा रहा हूँ"
    python infer.py "நான் meeting க்கு போறேன்" --voice ta_hinglish
    python infer.py "..." --ref my_voice.wav --ref-text "exact transcript"
    python infer.py --list-voices

Match the reference voice to your text's language — it affects quality more
than any other setting here.

    pip install torch torchaudio torchdiffeq x-transformers vocos \
                librosa soundfile transformers
"""

import argparse
import os
import sys

MODEL = "Tharshan/indicf5_hindi-english_code_switch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="text to speak")
    ap.add_argument("--voice", default=None,
                    help="bundled voice key (default: the first one)")
    ap.add_argument("--ref", help="your own reference wav, 5-15s")
    ap.add_argument("--ref-text", help="exact transcript of --ref")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--nfe-step", type=int, default=32,
                    help="denoising steps; 16 is faster, 64 rarely helps")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--list-voices", action="store_true")
    args = ap.parse_args()

    if bool(args.ref) != bool(args.ref_text):
        sys.exit("--ref and --ref-text go together: the model needs the exact "
                 "transcript of the reference clip, or output degrades badly")
    if args.ref and not os.path.exists(args.ref):
        sys.exit(f"reference audio not found: {args.ref}")
    if not args.text and not args.list_voices:
        sys.exit("give some text, or --list-voices")

    import soundfile as sf
    import torch
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device} ...")
    model = AutoModel.from_pretrained(args.model,
                                      trust_remote_code=True).to(device).eval()

    voices = model.voices()
    if args.list_voices:
        for key, meta in voices.items():
            print(f"  {key:16s} {meta.get('name', '')}")
        return

    if args.ref:
        ref_audio, ref_text = args.ref, args.ref_text
    else:
        key = args.voice or next(iter(voices))
        if key not in voices:
            sys.exit(f"unknown voice {key!r}. Available: {list(voices)}")
        ref_audio, ref_text = model.voice(key)
        print(f"voice: {key}")

    audio, sr = model.generate(args.text, ref_audio=ref_audio,
                               ref_text=ref_text, nfe_step=args.nfe_step,
                               speed=args.speed)
    sf.write(args.out, audio, sr)
    print(f"{args.out}  {len(audio) / sr:.2f}s")


if __name__ == "__main__":
    main()