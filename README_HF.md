# IndexTTS2 Vietnamese (Zero-Shot Voice Cloning)

This model card accompanies the Vietnamese adaptation of **IndexTTS2**. It exposes a ready-to-use checkpoint plus preprocessing recipe so you can fine-tune or run zero-shot inference on your own speakers.

## Model Summary

- **Architecture**: IndexTTS2 (GPT-based semantic generator + S2Mel + BigVGAN vocoder)
- **Languages**: Vietnamese (primary), retains English/Chinese from base checkpoint
- **Capabilities**: zero-shot speaker cloning, optional emotion mixing, sentence-level duration handling, torch.float16 inference
- **Input requirements**: ≤15 s speaker prompt WAV/FLAC (22.05 kHz recommended), UTF-8 text

## Intended Use & Limitations

| ✅ Recommended | ⚠️ Limitations |
|---------------|----------------|
| Voice cloning for assistants, IVR, audiobooks | Commercial use requires permission from IndexTTS authors |
| Research on low-resource TTS | Not robust to noisy transcripts; clean text is required |
| Emotion/style experiments | Emotion text prompts need Qwen assets (see `config.yaml`) |

## How to Use

Install dependencies and load the repo checkpoint. The snippet below assumes the weights/config live under `checkpoints/` inside this repo or your Hugging Face space.

```bash
pip install -U torch torchaudio soundfile sentencepiece modelscope huggingface_hub
```

```python
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,
    use_cuda_kernel=False,
)

result = tts.infer(
    spk_audio_prompt="samples/speaker.wav",
    text="Xin chào bạn nhé. Hôm nay trời đẹp quá!",
    output_path="gen.wav",
    verbose=True,
)
print(result)
```

For batch runs, use `infer_vi.py` or the Gradio demo provided in this repository.

### Automatic download

If `checkpoints/config.yaml` (or any required weights) are missing when you instantiate `IndexTTS2`, the code automatically downloads the full checkpoint bundle from this HuggingFace repo (`dinhthuan/index-tts-2-vietnamese`).  
Set `INDEXTTS_HF_REPO=<user/repo>` to pull from a different repository, or `INDEXTTS_DISABLE_AUTO_DOWNLOAD=1` to skip the download (e.g., on air-gapped machines).

## Data Preparation Recap

1. Convert your metadata CSV to a JSONL manifest: `tools/metadata_to_manifest.py`
2. Train or extend a SentencePiece tokenizer: `tools/tokenizer/train_bpe.py`
3. Run multiprocess preprocessing to extract semantic, acoustic, and emotion features: `tools/preprocess_multiproc.py`
4. Build GPT prompt/target pairs: `tools/generate_gpt_pairs.py`
5. Fine-tune with `trainers/train_gpt_v2.py`

`run_metadata_pipeline.sh` automates steps 1–4.

## Training Procedure

- Base checkpoint: `checkpoints/gpt_old.pth` (IndexTTS2 official release)
- Optimiser: AdamW, LR 1e-5, weight decay 0.01, cosine decay
- Loss mix: text 0.2, mel 0.8
- Mixed precision (AMP) with gradient clipping at 1.0

## Evaluation

- Subjective MOS on internal Vietnamese validation set (~4.3 MOS @ 22 kHz)
- Word Error Rate similar to base IndexTTS2 for English/Chinese prompts (no catastrophic forgetting observed)

## Citation

Please cite the original IndexTTS2 paper if you use this work:

```
@article{IndexTTS2,
  title={IndexTTS2: Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot TTS},
  author={IndexTTS Team},
  journal={arXiv preprint arXiv:2506.21619},
  year={2025}
}
```

For commercial inquiries: **indexspeech@bilibili.com**.
