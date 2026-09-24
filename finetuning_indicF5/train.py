
"""
Fine-tune IndicF5 on Hindi-English code-switched data.

Hardware:
    2x NVIDIA T4 (Kaggle)

Base model:
    ai4bharat/IndicF5

Output:
    Tharshan/indicf5_hindi-english_code_switch

Training:
    Adapter-style fine-tuning
"""

import argparse
import os
import sys

import torch

from f5_tts.model import CFM, DiT
from f5_tts.model.trainer import Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "ai4bharat/IndicF5"

HF_REPO_ID = "Tharshan/indicf5_hindi-english_code_switch"

INDICF5_MODEL_CFG = dict(
    dim=1024,
    depth=22,
    heads=16,
    ff_mult=2,
    text_dim=512,
    conv_layers=4,
)

MEL_SPEC_KWARGS = dict(
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    mel_spec_type="vocos",
)


# ============================================================
# LOAD INDICF5 CHECKPOINT
# ============================================================

def find_indicf5_checkpoint():

    hf_cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--ai4bharat--IndicF5"
    )

    for root, dirs, files in os.walk(hf_cache):

        if "model.safetensors" in files:

            return os.path.join(root, "model.safetensors")

    print("IndicF5 checkpoint not found in cache")
    print("Downloading from Hugging Face...")

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        BASE_MODEL,
        filename="model.safetensors"
    )


def load_pretrained_weights(
    model: CFM,
    checkpoint_path: str,
    device: str = "cpu"
):

    from safetensors.torch import load_file

    print(f"Loading pretrained weights from {checkpoint_path}")

    state_dict = load_file(
        checkpoint_path,
        device=device
    )

    cleaned = {}

    for k, v in state_dict.items():

        key = k

        for prefix in ["ema_model.", "_orig_mod."]:

            while key.startswith(prefix):

                key = key[len(prefix):]

        if key.startswith("vocoder."):

            continue

        cleaned[key] = v

    missing, unexpected = model.load_state_dict(
        cleaned,
        strict=False
    )

    if missing:

        print(f"Missing keys: {len(missing)}")

        for k in missing[:5]:

            print(f"  {k}")

    if unexpected:

        print(f"Unexpected keys: {len(unexpected)}")

        for k in unexpected[:5]:

            print(f"  {k}")

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{total_params / 1e6:.1f}M"
    )

    return model


# ============================================================
# ADAPTER-STYLE TRAINING
# ============================================================

def freeze_backbone(model: CFM):

    """
    Freeze transformer backbone.

    Trainable:
        Text embeddings
        Input/output projections
        Normalization layers
        Mel-related trainable parameters
        Prediction head
    """

    trainable_patterns = [
        "text_embed",
        "proj_in",
        "proj_out",
        "mel_spec",
        "norm",
        "to_pred",
    ]

    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():

        should_train = any(
            pattern in name
            for pattern in trainable_patterns
        )

        if should_train:

            param.requires_grad = True

            trainable_count += param.numel()

        else:

            param.requires_grad = False

            frozen_count += param.numel()

    total = frozen_count + trainable_count

    print("\nTraining configuration:")

    print(
        f"Frozen parameters: "
        f"{frozen_count / 1e6:.1f}M "
        f"({frozen_count / total * 100:.1f}%)"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_count / 1e6:.1f}M "
        f"({trainable_count / total * 100:.1f}%)"
    )


# ============================================================
# HUGGING FACE UPLOAD
# ============================================================

def upload_to_huggingface(checkpoint_dir):

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")

    if not token:

        print("\nHF_TOKEN not found.")
        print("Skipping Hugging Face upload.")

        return

    print(
        f"\nUploading model to {HF_REPO_ID}..."
    )

    api = HfApi(token=token)

    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type="model",
        exist_ok=True
    )

    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Fine-tuned IndicF5 Hindi-English code-switch model"
    )

    print(
        f"Model uploaded successfully:\n"
        f"https://huggingface.co/{HF_REPO_ID}"
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Fine-tune IndicF5 on Hindi-English data"
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to prepared dataset"
    )

    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Checkpoint output directory"
    )

    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Enable adapter-style training"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )

    # --------------------------------------------------------
    # OPTIMIZED TRAINING PARAMETERS FOR 2x NVIDIA T4
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=9600,
        help="Batch size in frames"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=4
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=500
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--wandb-project",
        default="indicf5-hinglish"
    )

    parser.add_argument(
        "--wandb-run",
        default="indicf5-2xt4"
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true"
    )

    parser.add_argument(
        "--log-samples",
        action="store_true"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    ckpt_dir = os.path.abspath(
        args.checkpoint_dir
    )

    os.makedirs(
        ckpt_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)

    print("IndicF5 Hindi-English Fine-Tuning")

    print("=" * 60)

    print(f"Base model:       {BASE_MODEL}")

    print(f"Output repository: {HF_REPO_ID}")

    print(f"Device:           {device}")

    if device == "cuda":

        print(
            f"GPU 0: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"Available GPUs: "
            f"{torch.cuda.device_count()}"
        )

    print(f"Data directory:   {data_dir}")

    print(f"Checkpoint dir:   {ckpt_dir}")

    print(f"Epochs:           {args.epochs}")

    print(f"Learning rate:    {args.lr}")

    print(f"Batch frames:     {args.batch_size}")

    print(f"Max samples:      {args.max_samples}")

    print(f"Grad accumulation: {args.grad_accum}")

    print(f"Warmup steps:     {args.warmup_steps}")

    print(f"Save every:       {args.save_every}")

    print(f"Adapter training: {args.freeze_backbone}")

    print("=" * 60 + "\n")

    # --------------------------------------------------------
    # VOCABULARY
    # --------------------------------------------------------

    vocab_path = os.path.join(
        data_dir,
        "vocab.txt"
    )

    if not os.path.exists(vocab_path):

        print(
            f"ERROR: vocab.txt not found at {vocab_path}"
        )

        sys.exit(1)

    vocab_char_map, vocab_size = get_tokenizer(
        vocab_path,
        tokenizer="custom"
    )

    print(f"Vocabulary size: {vocab_size}")

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    print("Building IndicF5 architecture...")

    backbone = DiT(
        **INDICF5_MODEL_CFG,
        text_num_embeds=vocab_size,
        mel_dim=100,
    )

    model = CFM(
        transformer=backbone,

        mel_spec_kwargs=dict(
            n_mel_channels=100,
            target_sample_rate=24000,
            hop_length=256,
            n_fft=1024,
            win_length=1024,
            mel_spec_type="vocos",
        ),

        odeint_kwargs=dict(
            method="euler"
        ),

        vocab_char_map=vocab_char_map,
    )

    # --------------------------------------------------------
    # LOAD BASE MODEL
    # --------------------------------------------------------

    if not args.resume:

        ckpt_path = find_indicf5_checkpoint()

        model = load_pretrained_weights(
            model,
            ckpt_path,
            device="cpu"
        )

    # --------------------------------------------------------
    # FREEZE BACKBONE
    # --------------------------------------------------------

    if args.freeze_backbone:

        print("\nEnabling adapter-style training...")

        freeze_backbone(model)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print(
        f"\nLoading dataset from {data_dir}..."
    )

    train_dataset = load_dataset(
        dataset_name="openslr104",

        dataset_type="CustomDatasetPath",

        audio_type="raw",

        data_dir=data_dir,

        mel_spec_kwargs=dict(
            n_mel_channels=100,
            target_sample_rate=24000,
            hop_length=256,
            n_fft=1024,
            win_length=1024,
            mel_spec_type="vocos",
        ),
    )

    print(
        f"Dataset utterances: {len(train_dataset)}"
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    print("\nCreating trainer...")

    trainer = Trainer(

        model=model,

        epochs=args.epochs,

        learning_rate=args.lr,

        num_warmup_updates=args.warmup_steps,

        save_per_updates=args.save_every,

        checkpoint_path=ckpt_dir,

        batch_size=args.batch_size,

        batch_size_type="frame",

        max_samples=args.max_samples,

        grad_accumulation_steps=args.grad_accum,

        max_grad_norm=1.0,

        logger=None if args.no_wandb else "wandb",

        wandb_project=args.wandb_project,

        wandb_run_name=args.wandb_run,

        log_samples=args.log_samples,

        last_per_steps=args.save_every,

        mel_spec_type="vocos",

        ema_kwargs=dict(
            beta=0.9999,
            update_after_step=100,
            update_every=10,
        ),
    )

    # --------------------------------------------------------
    # START TRAINING
    # --------------------------------------------------------

    print("\nStarting training...\n")

    trainer.train(

        train_dataset,

        num_workers=args.num_workers,

        resumable_with_seed=args.seed,
    )

    print("\nTraining complete!")

    print(
        f"Checkpoints saved to: {ckpt_dir}"
    )

    # --------------------------------------------------------
    # UPLOAD TO HUGGING FACE
    # --------------------------------------------------------

    upload_to_huggingface(ckpt_dir)


if __name__ == "__main__":

    main()