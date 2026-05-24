#!/usr/bin/env python3
"""
Оптимизированный скрипт для быстрого инференса IndexTTS2 на Tesla V100.

Ключевые оптимизации:
- FP16 (полуточная точность) для ускорения вычислений на тензорных ядрах V100
- torch.compile для компиляции графа модели
- CUDA ядра для BigVGAN
- Оптимизированные параметры батчинга
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from indextts.infer_v2 import IndexTTS2


DEFAULT_CONFIG = Path("checkpoints/config.yaml")
DEFAULT_MODEL_DIR = Path("checkpoints")
DEFAULT_GPT = Path("runs/vietnamese_from_metadata/gpt_finetune_vi/latest.pth")
DEFAULT_TOKENIZER = Path("runs/vietnamese_from_metadata/tokenizer/vi_bpe.model")
DEFAULT_SPEAKER = Path("ref_audio.wav")
DEFAULT_OUTPUT = Path("vi_out.wav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Оптимизированный инференс IndexTTS2 для Vietnamese (Tesla V100)."
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", type=str, help="Text to synthesise.")
    text_group.add_argument(
        "--text-file",
        type=str,
        help="Path to UTF-8 text file containing the synthesis prompt.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=str(DEFAULT_SPEAKER),
        help="Reference speaker audio (default: ref_audio.wav).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Destination wav path (default: vi_out.wav).",
    )
    parser.add_argument(
        "--gpt-checkpoint",
        type=str,
        default=str(DEFAULT_GPT),
        help="Fine-tuned GPT checkpoint to load.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=str(DEFAULT_TOKENIZER),
        help="SentencePiece tokenizer to use.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Base YAML config referencing the other model assets.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Directory that hosts the s2mel / vocoder / semantic assets.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Override device string (e.g. cuda:0). Default: cuda:0 for V100.",
    )
    # Оптимизации производительности
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,  # Включено по умолчанию для V100
        help="Run GPT inference in FP16 (enabled by default for V100).",
    )
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        default=True,  # Включено по умолчанию
        help="Enable torch.compile optimization (enabled by default).",
    )
    parser.add_argument(
        "--use-cuda-kernel",
        action="store_true",
        default=True,  # Включено по умолчанию
        help="Use custom CUDA kernels for BigVGAN (enabled by default).",
    )
    parser.add_argument(
        "--use-deepspeed",
        action="store_true",
        default=False,
        help="Use DeepSpeed for GPT inference.",
    )
    parser.add_argument(
        "--use-accel",
        action="store_true",
        default=False,
        help="Use acceleration engine for GPT2.",
    )
    
    # Параметры качества/скорости
    parser.add_argument(
        "--emo-audio",
        type=str,
        default=None,
        help="Optional emotion reference audio clip.",
    )
    parser.add_argument(
        "--emo-alpha",
        type=float,
        default=1.0,
        help="Blend factor for the emotion reference audio.",
    )
    parser.add_argument(
        "--emo-text",
        type=str,
        default=None,
        help="Optional emotion text prompt when using --use-emo-text.",
    )
    parser.add_argument(
        "--use-emo-text",
        action="store_true",
        help="Derive the emotion vector from text via Qwen emotion model.",
    )
    parser.add_argument(
        "--strip-punctuation",
        action="store_true",
        help="Remove sentence-ending punctuation before synthesis.",
    )
    parser.add_argument(
        "--interval-silence",
        type=int,
        default=200,
        help="Silence duration (milliseconds) inserted between segments.",
    )
    parser.add_argument(
        "--max-text-tokens",
        type=int,
        default=120,
        help="Maximum tokens per segment before splitting. Lower = faster but may affect quality.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Sampling top-k (leave unset for config default).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Sampling top-p (leave unset for config default).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (leave unset for config default).",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=None,
        help="Beam search width (leave unset for config default). Lower = faster.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed inference logs.",
    )
    return parser.parse_args()


def resolve_required(path_str: str, kind: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")
    return path


def load_text_arg(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text.strip()
    text_path = resolve_required(args.text_file, "Text file")
    return text_path.read_text(encoding="utf-8").strip()


def build_generation_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.num_beams is not None:
        kwargs["num_beams"] = args.num_beams
    return kwargs


def main() -> None:
    args = parse_args()
    text = load_text_arg(args)
    if not text:
        raise ValueError("Input text is empty after stripping whitespace.")

    speaker_path = resolve_required(args.speaker, "Speaker reference audio")
    if args.emo_audio:
        resolve_required(args.emo_audio, "Emotion reference audio")
    gpt_path = resolve_required(args.gpt_checkpoint, "GPT checkpoint")
    tokenizer_path = resolve_required(args.tokenizer, "Tokenizer")
    cfg_path = resolve_required(args.config, "Config file")
    model_dir = resolve_required(args.model_dir, "Model directory")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Model directory is not a folder: {model_dir}")

    cfg = OmegaConf.load(cfg_path)
    cfg.gpt_checkpoint = str(gpt_path)
    dataset_cfg = cfg.get("dataset")
    if not dataset_cfg or "bpe_model" not in dataset_cfg:
        raise KeyError("Config is missing dataset.bpe_model, cannot override tokenizer.")
    cfg.dataset["bpe_model"] = str(tokenizer_path)

    tmp_cfg_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp_file:
            OmegaConf.save(cfg, tmp_file.name)
            tmp_cfg_path = Path(tmp_file.name)

        print("=" * 60)
        print("ОПТИМИЗИРОВАННЫЙ ИНФЕРЕНС INDEX-TTS2 ДЛЯ TESLA V100")
        print("=" * 60)
        print(f"FP16: {args.fp16}")
        print(f"torch.compile: {args.use_torch_compile}")
        print(f"CUDA kernels BigVGAN: {args.use_cuda_kernel}")
        print(f"DeepSpeed: {args.use_deepspeed}")
        print(f"Accelerate: {args.use_accel}")
        print(f"Max tokens per segment: {args.max_text_tokens}")
        print("=" * 60)

        engine = IndexTTS2(
            cfg_path=str(tmp_cfg_path),
            model_dir=str(model_dir),
            device=args.device,
            use_fp16=args.fp16,
            use_cuda_kernel=args.use_cuda_kernel,
            use_deepspeed=args.use_deepspeed,
            use_accel=args.use_accel,
            use_torch_compile=args.use_torch_compile,
            strip_sentence_punctuation=args.strip_punctuation,
        )

        generation_kwargs = build_generation_kwargs(args)
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print("\nЗапуск инференса...")
        engine.infer(
            spk_audio_prompt=str(speaker_path),
            text=text,
            output_path=str(output_path),
            emo_audio_prompt=args.emo_audio,
            emo_alpha=args.emo_alpha,
            use_emo_text=args.use_emo_text,
            emo_text=args.emo_text,
            interval_silence=args.interval_silence,
            verbose=args.verbose,
            max_text_tokens_per_segment=args.max_text_tokens,
            **generation_kwargs,
        )
        print(f"\nИнференс завершен. Результат сохранен в {output_path.resolve()}")
    finally:
        if tmp_cfg_path is not None:
            try:
                tmp_cfg_path.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
