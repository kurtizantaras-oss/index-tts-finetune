#!/usr/bin/env python3
"""
Empirical proof-tests for the padding fix in tools/preprocess_data.py.

Three independent modes (run any of them):

1. `trainer-tokens` — no GPU, no checkpoints needed:
       python tests/preprocess_padding_proof.py trainer-tokens
   Replicates the exact token pipeline of trainers/train_gpt_v2.py
   (set_mel_padding + F.pad + build_aligned_inputs_and_targets) and shows
   what the model actually sees when codes are saved WITH vs WITHOUT
   manual 8192/8193 sentinels.

2. `invariance` — needs the same checkpoints as preprocess_data.py
   (run on the machine where you do preprocessing):
       python tests/preprocess_padding_proof.py invariance \
           --config checkpoints/config.yaml \
           --gpt-checkpoint checkpoints/gpt.pth \
           --audio path/to/any_clip.wav [second_clip.wav]
   Ground truth = processing a clip ALONE (batch of 1 → no padding).
   Proves:
     a) batched codes[:real_len] == solo codes (trimming is correct)
     b) batched codes[real_len:] length depends on the batchmate
        (i.e. the old saved tail is padding garbage, not audio)
     c) conditioning / emo_vec from the padded batch match the solo run
        WITHOUT any trimming (masking inside the model already works)
     d) conditioning is (32, dim) latents and emo_vec is a (dim,) vector —
        slicing them with [:real_len] corrupts them and breaks collate.

3. `compare` — end-to-end, numpy only:
       python tests/preprocess_padding_proof.py compare \
           --ref-dir out_bs1 --test-dir out_bs8
   Preprocess the same few samples twice: once with --batch-size 1
   (physically cannot contain padding → ground truth) and once with the
   fixed code and --batch-size 8. All per-id .npy files and manifest
   code_len must match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

START_MEL = 8192
STOP_MEL = 8193

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


# --------------------------------------------------------------------------
# Mode 1: trainer-tokens
# --------------------------------------------------------------------------

def run_trainer_pipeline(codes: "torch.Tensor", code_len: int):
    """Exact copy of the mel-token path in trainers/train_gpt_v2.py:492-496."""
    import torch
    import torch.nn.functional as F

    codes = codes.clone().unsqueeze(0)
    code_lengths = torch.tensor([code_len])
    # set_mel_padding
    for b in range(len(code_lengths)):
        actual_end = code_lengths[b]
        if actual_end < codes.shape[-1]:
            codes[b, actual_end:] = STOP_MEL
    # F.pad(..., value=stop_mel_token)
    codes = F.pad(codes, (0, 1), value=STOP_MEL)
    # build_aligned_inputs_and_targets
    inp = F.pad(codes, (1, 0), value=START_MEL)
    tar = F.pad(codes, (0, 1), value=STOP_MEL)
    return inp.squeeze(0), tar.squeeze(0)


def mode_trainer_tokens(_args) -> None:
    import torch

    raw = torch.tensor([101, 202, 303, 404, 505])

    print("\n=== trainer-tokens: what the GPT actually trains on ===\n")

    inp, tar = run_trainer_pipeline(raw, code_len=len(raw))
    print(f"codes saved CLEAN            : {raw.tolist()}")
    print(f"  -> model input             : {inp.tolist()}")
    print(f"  -> model target            : {tar.tolist()}")
    check(
        "clean codes get exactly one 8192 and one trailing stop region",
        inp[0].item() == START_MEL
        and inp[1].item() != START_MEL
        and inp[-1].item() == STOP_MEL
        and inp[-2].item() != STOP_MEL,
    )

    with_sentinels = torch.tensor([START_MEL, *raw.tolist(), STOP_MEL])
    inp2, tar2 = run_trainer_pipeline(with_sentinels, code_len=len(with_sentinels))
    print(f"\ncodes saved WITH 8192/8193   : {with_sentinels.tolist()}")
    print(f"  -> model input             : {inp2.tolist()}")
    print(f"  -> model target            : {tar2.tolist()}")
    check(
        "sentinel-in-file produces DOUBLED start/stop tokens (this is the bug)",
        inp2[0].item() == START_MEL and inp2[1].item() == START_MEL,
        "trainer adds its own 8192/8193, so the .npy must not contain them",
    )

    # and the code_len-mismatch variant: file has sentinels but manifest
    # code_len counts only the real codes
    inp3, _ = run_trainer_pipeline(with_sentinels, code_len=len(raw))
    print(f"\nsentinels in file, code_len={len(raw)} (unfixed manifest):")
    print(f"  -> model input             : {inp3.tolist()}")
    check(
        "mismatched code_len silently truncates real codes",
        inp3.tolist().count(raw[-1].item()) == 0,
        "last real code got overwritten by stop token",
    )


# --------------------------------------------------------------------------
# Mode 2: invariance (needs checkpoints, same env as preprocess_data.py)
# --------------------------------------------------------------------------

def mode_invariance(args) -> None:
    import torch
    import safetensors.torch
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    import preprocess_data as pp

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config)

    stats_value = OmegaConf.select(cfg, "w2v_stat")
    stats_path = Path(stats_value or "checkpoints/wav2vec2bert_stats.pt")
    if not stats_path.is_absolute():
        stats_path = (args.config.parent / stats_path).resolve()

    print(f"[setup] device={device}, loading models…")
    extractor = pp.SemanticExtractor(stats_path, device)
    codec = pp.build_semantic_codec(cfg.semantic_codec)
    ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(codec, ckpt)
    codec = codec.to(device).eval()
    gpt = pp.build_unified_voice(cfg, args.gpt_checkpoint, device)

    # --- prepare a short and a long clip -----------------------------------
    audios = list(args.audio)
    wav0, sr0 = pp.load_audio(Path(audios[0]), target_sr=24000)
    if len(audios) >= 2:
        wav1, sr1 = pp.load_audio(Path(audios[1]), target_sr=24000)
    else:
        wav1, sr1 = wav0, sr0
    # short = first 40% of the shorter clip, long = the full longer clip
    short_src = wav0 if wav0.shape[-1] <= wav1.shape[-1] else wav1
    long_wav = wav1 if wav1.shape[-1] >= wav0.shape[-1] else wav0
    short_wav = short_src[..., : max(int(short_src.shape[-1] * 0.4), 24000)]
    if long_wav.shape[-1] - short_wav.shape[-1] < 24000:
        print("[warn] clips too similar in length; long clip = short repeated x3")
        long_wav = short_src.repeat(1, 3)
    print(
        f"[setup] short={short_wav.shape[-1]/24000:.2f}s, "
        f"long={long_wav.shape[-1]/24000:.2f}s"
    )

    def extract_all(waves):
        with torch.inference_mode():
            feat, mask = extractor.extract(waves, [24000] * len(waves))
            codes, _ = codec.quantize(feat)
            if codes.dim() == 1:
                codes = codes.unsqueeze(0)
            lengths = mask.sum(dim=1).long()
            cond = gpt.get_conditioning(feat.transpose(1, 2), lengths.to(feat.device))
            emo = gpt.get_emovec(feat, lengths.to(feat.device))
        return codes.cpu(), lengths.cpu(), cond.float().cpu(), emo.float().cpu()

    # ground truth: short clip alone — no padding can exist
    codes_solo, len_solo, cond_solo, emo_solo = extract_all([short_wav])
    # the situation under test: same clip padded inside a batch
    codes_b, len_b, cond_b, emo_b = extract_all([short_wav, long_wav])

    real = int(len_b[0])
    padded = codes_b.shape[-1]
    print(f"\n=== invariance: solo (ground truth) vs padded batch ===")
    print(f"real_len={real}, padded_len={padded}, tail={padded - real} frames\n")

    check(
        "attention_mask length == solo code length",
        int(len_solo[0]) == codes_solo.shape[-1] and int(len_solo[0]) == real,
        f"solo={int(len_solo[0])}, real_len in batch={real}",
    )

    prefix = codes_b[0, :real]
    match = (prefix == codes_solo[0]).float().mean().item()
    mismatch_pos = (prefix != codes_solo[0]).nonzero().flatten().tolist()
    boundary_zone = 16  # conv receptive field of w2v-bert leaks across the pad edge
    check(
        "codes[:real_len] from padded batch == solo codes "
        "(diffs only in the conv boundary zone)",
        all(p >= real - boundary_zone for p in mismatch_pos),
        f"{match * 100:.2f}% identical; mismatch positions {mismatch_pos} "
        f"(all must be >= {real - boundary_zone})",
    )

    tail = codes_b[0, real:]
    check(
        "old code saved a non-empty garbage tail",
        tail.numel() > 0,
        f"{tail.numel()} extra tokens, {len(tail.unique())} distinct values: "
        f"{tail[:12].tolist()}…",
    )

    # tail length is dictated by the batchmate, not by the audio itself
    long_cut = long_wav[..., : int(long_wav.shape[-1] * 0.7)]
    codes_b2, len_b2, _, _ = extract_all([short_wav, long_cut])
    check(
        "saved length of the SAME clip depends on its batchmate (old bug)",
        codes_b2.shape[-1] != padded and int(len_b2[0]) == real,
        f"padded_len {padded} vs {codes_b2.shape[-1]} for identical audio",
    )

    # --- conditioning / emo_vec --------------------------------------------
    def cos(a, b):
        a, b = a.flatten(), b.flatten()
        return float((a @ b) / (a.norm() * b.norm() + 1e-8))

    c_cos = cos(cond_solo[0], cond_b[0])
    c_diff = (cond_solo[0] - cond_b[0]).abs().max().item()
    check(
        "conditioning: padded batch == solo, WITHOUT any trimming",
        c_cos > 0.999,
        f"cosine={c_cos:.6f}, max|diff|={c_diff:.2e}",
    )

    e_cos = cos(emo_solo[0], emo_b[0])
    e_diff = (emo_solo[0] - emo_b[0]).abs().max().item()
    check(
        "emo_vec: padded batch == solo, WITHOUT any trimming",
        e_cos > 0.999,
        f"cosine={e_cos:.6f}, max|diff|={e_diff:.2e}",
    )

    # --- shapes: why [:real_len] slicing corrupts them ----------------------
    _, _, cond_long, emo_long = extract_all([long_wav])
    check(
        "conditioning shape is fixed latents, independent of audio length",
        tuple(cond_solo[0].shape) == tuple(cond_long[0].shape),
        f"short→{tuple(cond_solo[0].shape)}, long→{tuple(cond_long[0].shape)} "
        f"(rows are NOT time frames)",
    )
    check(
        "emo_vec is a single vector, not a time sequence",
        emo_solo[0].dim() == 1,
        f"shape={tuple(emo_solo[0].shape)}",
    )

    sliced_short = emo_solo[0][:real]
    sliced_long = emo_long[0][: int(len_b[1])] if int(len_b[1]) < emo_long[0].numel() else emo_long[0]
    corrupts = sliced_short.numel() < emo_solo[0].numel()
    detail = (
        f"emo_vec[:{real}] keeps {sliced_short.numel()}/{emo_solo[0].numel()} dims"
        if corrupts
        else "clip long enough that slice was a no-op — try a clip under "
        f"{emo_solo[0].numel() / 50:.0f}s"
    )
    check("[:real_len] slice on emo_vec destroys the embedding", corrupts, detail)
    if corrupts:
        try:
            torch.stack([sliced_short, sliced_long])
            print("         (stack unexpectedly succeeded — equal lengths)")
        except RuntimeError as exc:
            check(
                "…and such files crash collate_batch (torch.stack)",
                True,
                str(exc).splitlines()[0],
            )


# --------------------------------------------------------------------------
# Mode 3: compare two preprocess output dirs
# --------------------------------------------------------------------------

def load_manifest_lens(root: Path) -> dict[str, int]:
    lens: dict[str, int] = {}
    for name in ("train_manifest.jsonl", "val_manifest.jsonl"):
        p = root / name
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    lens[rec["id"]] = rec["code_len"]
    return lens


def mode_compare(args) -> None:
    ref, test = Path(args.ref_dir), Path(args.test_dir)
    ref_ids = {p.stem for p in (ref / "codes").glob("*.npy")}
    test_ids = {p.stem for p in (test / "codes").glob("*.npy")}
    common = sorted(ref_ids & test_ids)
    print(f"\n=== compare: {ref} (batch=1 ground truth) vs {test} ===")
    print(f"common ids: {len(common)} (ref-only {len(ref_ids - test_ids)}, "
          f"test-only {len(test_ids - ref_ids)})\n")
    if not common:
        check("output dirs share sample ids", False)
        return

    ref_lens, test_lens = load_manifest_lens(ref), load_manifest_lens(test)
    stats = {"len": 0, "codes": 0, "cond": 0, "emo": 0, "manifest": 0}
    worst_match = 1.0
    for uid in common:
        rc = np.load(ref / "codes" / f"{uid}.npy")
        tc = np.load(test / "codes" / f"{uid}.npy")
        if rc.shape != tc.shape:
            stats["len"] += 1
            print(f"  code length mismatch {uid}: ref {rc.shape} vs test {tc.shape}")
            continue
        m = float((rc == tc).mean())
        worst_match = min(worst_match, m)
        # diffs are legal only inside the conv boundary zone at the clip end
        mismatch_pos = np.nonzero(rc != tc)[0]
        if mismatch_pos.size and mismatch_pos.min() < rc.size - 16:
            stats["codes"] += 1
            print(f"  code content mismatch {uid}: {m*100:.1f}% equal, "
                  f"diffs start at frame {mismatch_pos.min()}/{rc.size}")
        for key, sub in (("cond", "condition"), ("emo", "emo_vec")):
            ra = np.load(ref / sub / f"{uid}.npy").astype(np.float64)
            ta = np.load(test / sub / f"{uid}.npy").astype(np.float64)
            # batched conv/matmul is not bit-exact vs solo runs, so compare
            # by cosine similarity rather than absolute tolerance
            cos = float(
                (ra.flatten() @ ta.flatten())
                / (np.linalg.norm(ra) * np.linalg.norm(ta) + 1e-8)
            ) if ra.shape == ta.shape else -1.0
            if ra.shape != ta.shape or cos < 0.999:
                stats[key] += 1
                print(f"  {sub} mismatch {uid}: shapes {ra.shape}/{ta.shape}, "
                      f"cosine={cos:.6f}")
        if uid in ref_lens and uid in test_lens:
            npy_len = tc.size
            if test_lens[uid] != npy_len or ref_lens[uid] != rc.size:
                stats["manifest"] += 1
                print(f"  manifest code_len mismatch {uid}: "
                      f"manifest={test_lens[uid]} vs file={npy_len}")

    check("code lengths identical to batch=1 run", stats["len"] == 0)
    check("code contents match batch=1 run (diffs only in conv boundary zone)",
          stats["codes"] == 0, f"worst per-file match {worst_match*100:.2f}%")
    check("conditioning identical (no trimming needed)", stats["cond"] == 0)
    check("emo_vec identical (no trimming needed)", stats["emo"] == 0)
    check("manifest code_len == saved file length", stats["manifest"] == 0)


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("trainer-tokens")

    inv = sub.add_parser("invariance")
    inv.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"))
    inv.add_argument("--gpt-checkpoint", type=Path, default=Path("checkpoints/gpt.pth"))
    inv.add_argument("--device", default="cuda")
    inv.add_argument("--audio", nargs="+", default=["tests/sample_prompt.wav"],
                     help="1-2 wav files; ideally two clips of different duration")

    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--ref-dir", required=True,
                      help="output dir produced with --batch-size 1 (ground truth)")
    cmp_.add_argument("--test-dir", required=True,
                      help="output dir produced with the fix and --batch-size N")

    args = parser.parse_args()
    {"trainer-tokens": mode_trainer_tokens,
     "invariance": mode_invariance,
     "compare": mode_compare}[args.mode](args)

    print()
    if _failures:
        print(f"{FAIL}: {len(_failures)} check(s) failed: {_failures}")
        sys.exit(1)
    print(f"{PASS}: all checks passed")


if __name__ == "__main__":
    main()
