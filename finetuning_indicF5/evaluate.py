#!/usr/bin/env python3
"""
Evaluate an IndicF5 code-switch fine-tune against base IndicF5.

Generates speech from a fixed test set with both models, transcribes with
Whisper, and writes transcripts, CER and sample wavs.

    python evaluate.py                          # full run
    python evaluate.py --langs hi ta te         # subset
    python evaluate.py --model <repo-or-path>   # evaluate a different checkpoint
    python evaluate.py --no-base                # skip the comparison

Note on CER: it is reported only where the input and the ASR output share a
script. On code-switched text Whisper writes correctly-pronounced English in the
local script (morning -> मॉर्णिंग), so every correct English word counts as
several character errors and the metric ranks a good model below a bad one.
Those rows carry transcripts only, by design.

    pip install torch torchaudio torchdiffeq x-transformers vocos \
                librosa soundfile transformers jiwer
"""

import argparse
import json
import os
import re
import sys
import unicodedata

import numpy as np
import soundfile as sf
import torch

DEFAULT_MODEL = "Tharshan/indicf5_hindi-english_code_switch"
BASE_MODEL = "ai4bharat/IndicF5"
BASE_PREFIX = "ema_model._orig_mod."   # base ships torch.compile'd EMA weights

# code -> (voice key, whisper language, label)
LANGS = {
    "hi": ("ritu_hinglish", "hi", "Hindi"),
    "ta": ("ta_hinglish", "ta", "Tamil"),
    "te": ("te", "te", "Telugu"),
    "bn": ("bn", "bn", "Bengali"),
    "kn": ("kn", "kn", "Kannada"),
    "ml": ("ml", "ml", "Malayalam"),
    "mr": ("mr", "mr", "Marathi"),
    "gu": ("gu", "gu", "Gujarati"),
    "pa": ("pa", "pa", "Punjabi"),
    # Odia has a bundled voice but Whisper can't transcribe it
}

# Each language gets a pure sentence (does the fine-tune cost anything?) and a
# code-switched one (did it help?). `cer` marks whether the metric is valid.
TESTS = {
    "hi": {
        "pure": "नमस्ते दोस्तों, आज का मौसम बहुत अच्छा है और मैं सोच रहा हूँ कि हम सब मिलकर पार्क में घूमने चलें।",
        "mixed": "मैं आज office जा रहा हूँ, morning में एक important meeting है और उसके बाद team के साथ new project discuss करना है।",
    },
    "ta": {
        "pure": "வணக்கம் நண்பர்களே, இன்று வானிலை மிகவும் நன்றாக இருக்கிறது.",
        "mixed": "நான் இன்னைக்கு office போறேன், morning ல ஒரு important meeting இருக்கு.",
    },
    "te": {
        "pure": "నమస్కారం మిత్రులారా, ఈరోజు వాతావరణం చాలా బాగుంది.",
        "mixed": "నేను ఈరోజు office కి వెళ్తున్నాను, morning లో ఒక important meeting ఉంది.",
    },
    "bn": {
        "pure": "নমস্কার বন্ধুরা, আজ আবহাওয়া খুব সুন্দর।",
        "mixed": "আমি আজ office যাচ্ছি, morning এ একটা important meeting আছে।",
    },
    "kn": {
        "pure": "ನಮಸ್ಕಾರ ಸ್ನೇಹಿತರೇ, ಇವತ್ತು ವಾತಾವರಣ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ.",
        "mixed": "ನಾನು ಇವತ್ತು office ಗೆ ಹೋಗುತ್ತಿದ್ದೇನೆ, morning ನಲ್ಲಿ ಒಂದು important meeting ಇದೆ.",
    },
    "ml": {
        "pure": "നമസ്കാരം സുഹൃത്തുക്കളേ, ഇന്ന് കാലാവസ്ഥ വളരെ നല്ലതാണ്.",
        "mixed": "ഞാൻ ഇന്ന് office ലേക്ക് പോകുന്നു, morning ൽ ഒരു important meeting ഉണ്ട്.",
    },
    "mr": {
        "pure": "नमस्कार मित्रांनो, आज हवामान खूप छान आहे.",
        "mixed": "मी आज office ला जातोय, morning मध्ये एक important meeting आहे.",
    },
    "gu": {
        "pure": "નમસ્તે મિત્રો, આજે હવામાન ખૂબ સરસ છે.",
        "mixed": "હું આજે office જઈ રહ્યો છું, morning માં એક important meeting છે.",
    },
    "pa": {
        "pure": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਦੋਸਤੋ, ਅੱਜ ਮੌਸਮ ਬਹੁਤ ਵਧੀਆ ਹੈ।",
        "mixed": "ਮੈਂ ਅੱਜ office ਜਾ ਰਿਹਾ ਹਾਂ, morning ਵਿੱਚ ਇੱਕ important meeting ਹੈ।",
    },
}

# English-only, generated with the Hindi voice
ENGLISH_TEST = ("ritu_hinglish", "en",
                "The weather is quite pleasant today, so we are planning to "
                "visit the park in the evening.")


def load_model(repo, device):
    from transformers import AutoModel
    return AutoModel.from_pretrained(repo, trust_remote_code=True).to(device).eval()


def load_base(reference, device):
    """Build a model matching `reference`'s architecture, load base weights."""
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
    print(f"base: {loaded}/{len(target)} tensors")
    if loaded < 0.9 * len(target):
        sys.exit("base weights failed to load — the key prefix may have changed")
    return m.to(device).eval()


class Transcriber:
    def __init__(self, device, model="openai/whisper-large-v3"):
        from transformers import (WhisperForConditionalGeneration,
                                  WhisperProcessor)
        self.device = device
        self.proc = WhisperProcessor.from_pretrained(model)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model, torch_dtype=dtype).to(device).eval()
        self.dtype = dtype

    def __call__(self, audio, sr, lang):
        import librosa
        a = audio.mean(axis=1) if audio.ndim > 1 else audio
        if sr != 16000:
            a = librosa.resample(a.astype("float32"), orig_sr=sr, target_sr=16000)
        feats = self.proc(a, sampling_rate=16000,
                          return_tensors="pt").input_features
        ids = self.model.generate(feats.to(self.device, self.dtype),
                                  language=lang, task="transcribe",
                                  max_new_tokens=200)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def norm(s):
    s = unicodedata.normalize("NFC", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--langs", nargs="*", default=list(LANGS),
                    help=f"subset of: {' '.join(LANGS)}")
    ap.add_argument("--no-base", action="store_true",
                    help="evaluate the fine-tune alone, no comparison")
    ap.add_argument("--no-english", action="store_true")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--samples-dir", default="samples")
    ap.add_argument("--whisper", default="openai/whisper-large-v3")
    ap.add_argument("--nfe-step", type=int, default=32)
    args = ap.parse_args()

    unknown = [l for l in args.langs if l not in LANGS]
    if unknown:
        sys.exit(f"unknown language code(s): {unknown}. Known: {list(LANGS)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print(f"loading {args.model} ...")
    models = {"ours": load_model(args.model, device)}
    if not args.no_base:
        print("loading base ...")
        models["base"] = load_base(models["ours"], device)

    available = models["ours"].voices()
    print(f"voices: {list(available)}")

    print("loading whisper ...")
    transcribe = Transcriber(device, args.whisper)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)
    rows = []

    def run(tag, voice_key, wlang, text, score_cer):
        if voice_key not in available:
            print(f"  {tag}: voice {voice_key!r} not bundled, skipped")
            return
        ref, ref_text = models["ours"].voice(voice_key)
        entry = {"id": tag, "lang": wlang, "input": text, "cer_valid": score_cer}
        for name, mdl in models.items():
            audio, sr = mdl.generate(text, ref_audio=ref, ref_text=ref_text,
                                     nfe_step=args.nfe_step)
            path = os.path.join(args.samples_dir, f"{tag}_{name}.wav")
            sf.write(path, np.asarray(audio, np.float32), sr)
            hyp = transcribe(audio, sr, wlang)
            entry[name] = hyp
            if score_cer:
                from jiwer import cer
                entry[f"cer_{name}"] = round(cer(norm(text), norm(hyp)), 3)
        rows.append(entry)

        print(f"  {tag}")
        for name in models:
            c = f"  CER {entry[f'cer_{name}']:.3f}" if score_cer else ""
            print(f"    {name:5s}{c} -> {entry[name]}")

    for code in args.langs:
        voice_key, wlang, label = LANGS[code]
        print(f"\n{label}")
        # pure-language: same script in and out, so CER is meaningful
        run(f"{code}_pure", voice_key, wlang, TESTS[code]["pure"], True)
        # code-switched: CER is invalid here, transcripts only
        run(f"{code}_mixed", voice_key, wlang, TESTS[code]["mixed"], False)

    if not args.no_english:
        print("\nEnglish")
        vk, wlang, text = ENGLISH_TEST
        run("en", vk, wlang, text, True)

    # ---- write results ------------------------------------------------
    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    md_path = os.path.join(args.out_dir, "results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        cols = list(models)

        f.write("# Evaluation\n\n")
        f.write(f"Model: `{args.model}`  \n")
        f.write(f"ASR: `{args.whisper}`, nfe_step={args.nfe_step}\n\n")

        f.write("## Code-switched\n\n")
        f.write("Transcripts only — CER ranks these backwards, because Whisper\n"
                "writes correct English in the local script.\n\n")
        f.write("| Test | " + " | ".join(cols) + " |\n")
        f.write("|---" * (len(cols) + 1) + "|\n")
        for r in rows:
            if r["cer_valid"]:
                continue
            f.write(f"| {r['id']} | " + " | ".join(r[c] for c in cols) + " |\n")

        f.write("\n## Pure-language and English\n\n")
        f.write("| Test | " + " | ".join(f"{c} (CER)" for c in cols) + " |\n")
        f.write("|---" * (len(cols) + 1) + "|\n")
        for r in rows:
            if not r["cer_valid"]:
                continue
            cells = [f"{r[c]} ({r[f'cer_{c}']:.3f})" for c in cols]
            f.write(f"| {r['id']} | " + " | ".join(cells) + " |\n")

    print(f"\nwrote {md_path}, {json_path}, and {args.samples_dir}/")


if __name__ == "__main__":
    main()