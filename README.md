# IndexTTS2 Vietnamese Training & Inference Toolkit

This repository packages the full workflow we use to adapt **IndexTTS2** to Vietnamese (or any low-resource language) — from metadata ingestion, tokenizer training, multiprocess preprocessing, prompt/target generation, GPT fine-tuning, to multiple inference entry points (CLI, Gradio demo, and scripted pipelines).

The repo started as a faithful port of the official [index-tts/index-tts](https://github.com/index-tts/index-tts) codebase and has since been reorganised with an eye on training practicality:

- ✅ **Automated metadata pipeline** that takes CSV metadata and produces ready-to-train manifests, tokenizers, and prompt/target pairs (`run_metadata_pipeline.sh`).
- ✅ **Trainer scripts** that fine-tune GPT on new languages without destroying the base model’s English/Chinese capability (`trainers/train_gpt_v2.py`).
- ✅ **High-quality inference utilities** (`predict.py`, `infer_vi.py`, `gradio_infer_vi.py`, `webui.py`) with support for emotion prompts and mixed precision generation.
- ✅ **SoundFile-based audio saving** so inference does not depend on TorchCodec.

Use this README as the single source of truth for setup, preprocessing, training, and inference.

---

## 1. Repository Layout

```
.
├── indextts/                 # Core model code (GPT, semantic codec, s2mel, vocoder, utils)
├── tools/                    # Data utilities (metadata conversion, preprocessing, tokenizer, pairing)
├── trainers/train_gpt_v2.py  # Primary training/finetuning entry point
├── run_metadata_pipeline.sh  # End-to-end metadata ➜ tokenizer ➜ features ➜ pairs helper
├── predict.py                # Minimal inference script (single shot)
├── infer_vi.py               # Sentence-aware inference helper for Vietnamese
├── gradio_infer_vi.py        # Simple UI for testing
├── train.sh / train.bat      # Example training command line
└── checkpoints/              # Configs, tokenizers, pretrained weights
```

---

## 2. Environment Setup

1. **Install system dependencies**
   - Python 3.10+
   - CUDA toolkit (for GPU training/inference)
   - [git](https://git-scm.com/) + [git-lfs](https://git-lfs.com/)
   - `ffmpeg` and `sox` are recommended for inspecting audio.

2. **Clone the repository & pull LFS assets**

```bash
git clone https://github.com/iamdinhthuan/index-tts-finetune-vietnamese.git
cd index-tts-finetune-vietnamese
git lfs install
git lfs pull
```

> **Automatic weights download:** If `checkpoints/config.yaml` (or other weights) are missing at inference time, the code automatically downloads them from [huggingface.co/dinhthuan/index-tts-2-vietnamese](https://huggingface.co/dinhthuan/index-tts-2-vietnamese).  
> - Override the source repo with `INDEXTTS_HF_REPO=<user/repo>`.  
> - Disable the behaviour by setting `INDEXTTS_DISABLE_AUTO_DOWNLOAD=1` (useful for offline environments).

3. **Create a Python environment** (choose one)

```bash
# uv (recommended)
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# or pip
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -e .
```

4. **Optional: verify GPU visibility**

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')
PY
```

---

## 3. Preparing Data

We expect a metadata CSV with at least the following columns:

| column       | description                                    |
|--------------|------------------------------------------------|
| `audio_path` | Path (absolute or relative) to the WAV/FLAC    |
| `text`       | Normalised text transcription                  |
| `speaker_id` | Speaker label (string or int)                  |
| `language`   | Optional language tag (defaults to `vi`)       |

There are two ways to prepare training inputs.

example of file

id|text

sample_000000|Augsti godātais Valsts prezidenta kungs! Ekselences! Godātie klātesošie! Godātie deputāti!


### 3.1 Manual step-by-step

1. **Convert metadata to manifest**

```bash
python tools/metadata_to_manifest.py \
  --metadata data/metadata.csv \
  --audio-root data/wavs \
  --output runs/vi/manifests/train.jsonl \
  --default-language vi
```

2. **Train (or extend) a tokenizer**

```bash
python tools/tokenizer/train_bpe.py \
  --manifest runs/vi/manifests/train.jsonl \
  --output-prefix runs/vi/tokenizer/vi_bpe \
  --vocab-size 12000 --model-type bpe --byte-fallback
```

3. **Preprocess audio + extract features** (speaker cond, semantic tokens, mels):

```bash
python tools/preprocess_multiproc.py \
  --manifest runs/vi/manifests/train.jsonl \
  --output-dir runs/vi/processed \
  --tokenizer runs/vi/tokenizer/vi_bpe.model \
  --config checkpoints/config.yaml \
  --gpt-checkpoint checkpoints/gpt_old.pth \
  --language vi \
  --val-ratio 0.01 \
  --device cuda --batch-size 4 --workers 4 --num-processes 4 \
  --hf-cache-dir runs/vi/hf_cache
```

4. **Create GPT prompt/target pairs**

```bash
python tools/generate_gpt_pairs.py \
  --dataset runs/vi/processed \
  --pairs-per-target 2 \
  --force
```

This produces `gpt_pairs_train.jsonl` and `gpt_pairs_val.jsonl` inside `runs/vi/processed`.

### 3.2 One-command pipeline

`run_metadata_pipeline.sh` glues the above steps together. Example:

```bash
./run_metadata_pipeline.sh data/metadata.csv data/wavs runs/vietnamese_run
```

Environment variables let you customise tokenizer size, split ratio, GPU device, etc. Inspect the script header for all knobs.

---

## 4. Training / Fine-tuning

1. **Prepare config/checkpoints**
   - `checkpoints/config.yaml` should match the architecture you want to train.
   - `checkpoints/gpt_old.pth` (or any checkpoint) is used as the starting point.

2. **Launch training**

```bash
uv run python trainers/train_gpt_v2.py \
  --train-manifest runs/vi/processed/gpt_pairs_train.jsonl \
  --val-manifest runs/vi/processed/gpt_pairs_val.jsonl \
  --tokenizer runs/vi/tokenizer/vi_bpe.model \
  --config checkpoints/config.yaml \
  --base-checkpoint checkpoints/gpt_old.pth \
  --output-dir runs/vi/finetune_ckpts \
  --batch-size 16 --grad-accumulation 2 \
  --epochs 10 --learning-rate 1e-5 --weight-decay 1e-2 \
  --warmup-steps 1000 --log-interval 10 --val-interval 2000 \
  --grad-clip 1.0 --text-loss-weight 0.2 --mel-loss-weight 0.8 \
  --amp --resume auto
```

Use `train.sh` / `train.bat` as templates. Checkpoints + optimizer states are written to `--output-dir` (best + last). TensorBoard logs go to the same folder.

3. **Pruning / exporting** (optional)

```bash
python tools/prune_gpt_checkpoint.py --checkpoint runs/vi/finetune_ckpts/last.pth --output checkpoints/gpt_vi.pth
```

---

## 5. Inference

### 5.1 Minimal CLI (`predict.py`)

```bash
python predict.py \
  --prompt ref_audio.wav \
  --text "Chiến lược quân sự là nghệ thuật định hướng và sử dụng sức mạnh quân sự nhằm đạt được mục tiêu chính trị, và qua từng thời kỳ, con người đã phát triển nhiều cách tiếp cận khác nhau. Từ thời cổ đại, Tôn Tử đã nhấn mạnh yếu tố mưu lược và coi trọng việc giành thắng lợi bằng trí tuệ, tạo thế và đánh vào tâm lý đối phương hơn là chỉ dựa vào sức mạnh. Trong lịch sử, có những chiến lược như tiêu hao, tức là dùng sức mạnh liên tục để bào mòn lực lượng địch, hay quyết chiến nhanh, tập trung toàn bộ binh lực vào một trận đánh then chốt để xoay chuyển cục diện, như Hannibal ở Cannae hay Võ Nguyên Giáp ở Điện Biên Phủ."
```

The script internally handles sentence splitting and saves `gen.wav`. Edit the file to point to your checkpoint/config.

### 5.2 Batch / streaming (`infer_vi.py` / `indextts.infer_v2.IndexTTS2`)

`infer_vi.py` exposes richer arguments (emotions, streaming, quick tokens). Example:

```bash
python infer_vi.py \
  --cfg checkpoints/config.yaml \
  --model-dir checkpoints \
  --spk-prompt assets/prompt.wav \
  --text-file samples/input.txt \
  --output gen.wav \
  --use-fp16 --strip-punctuation
```

### 5.3 Gradio / Web demo

```bash
python gradio_infer_vi.py   # lightweight demo
python webui.py             # full-featured browser UI
```

Both share the same backend, so any checkpoint/config combination that works for CLI will work here.

### 5.4 Audio saving backend

We patched the repository to use `soundfile` by default (`indextts.utils.audio_io.save_audio`). Install the dependency when running inference:

```bash
pip install soundfile
```

---

## 6. Tips & Troubleshooting

- **Tokenizer drift**: keep the same tokenizer between preprocessing, training, and inference. Mismatches cause `KeyError` on tokens.
- **CUDA OOM**: reduce `--batch-size` in preprocessing and training, or enable gradient accumulation.
- **Long prompts**: `_load_and_cut_audio` trims prompt audio to 15s by default. Update the config if you need longer conditioning.
- **Emotion prompts**: pass `--emo-audio-prompt` or `--emo-text` when using `infer_vi.py`. Text-based emotions require Qwen assets defined in `config.yaml`.
- **HF cache**: set `HF_HUB_CACHE=/custom/path` to reuse downloaded semantic/vocoder weights.

---

## 7. Acknowledgements

This project is built on top of the open-source [IndexTTS](https://github.com/index-tts/index-tts) initiative by the bilibili team. Please respect the upstream license and cite their paper ([arXiv:2506.21619](https://arxiv.org/abs/2506.21619)) when publishing results.

For commercial licensing, contact **indexspeech@bilibili.com**. For this Vietnamese fork feel free to open GitHub issues or reach out via Discord/QQ listed in the upstream repo.
