#!/usr/bin/env python3
"""Convert a metadata CSV into the manifest format expected by ``preprocess_data.py``.

This helper makes it easy to bootstrap datasets that only provide a
``metadata.csv`` (e.g. LJ Speech style ``id|text|text_norm`` files) plus a
directory of audio files. The script emits a JSONL manifest where each row has
the ``id``, ``text`` and ``audio`` keys consumed by the preprocessing pipeline
and optionally ``speaker`` / ``language`` / ``duration`` hints.

Example usage (LJ Speech layout)::

    python tools/metadata_to_manifest.py \
        --metadata /data/lj/metadata.csv \
        --audio-root /data/lj/wavs \
        --text-column 1 --id-column 0 \
        --audio-pattern "{id}.wav" \
        --output /data/lj/lj_manifest.jsonl

After producing the manifest you can run ``tools/preprocess_data.py`` with
``--manifest /data/lj/lj_manifest.jsonl`` as usual.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert metadata CSV to preprocessing manifest.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata CSV/TSV file.")
    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Directory containing the waveform files referenced by the metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metadata_manifest.jsonl"),
        help="Destination JSONL manifest path (default: metadata_manifest.jsonl).",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="|",
        help="Field delimiter inside metadata (default: '|', pass ',' for standard CSV).",
    )
    parser.add_argument(
        "--quotechar",
        type=str,
        default="\"",
        help="Quote character used in the metadata file (default: double quote).",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Encoding used when reading the metadata file (default: utf-8).",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Treat the metadata as headerless (use column indices instead of names).",
    )

    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Column name or zero-based index that contains a unique utterance ID.",
    )
    parser.add_argument(
        "--text-column",
        dest="text_columns",
        action="append",
        default=None,
        help="Column name/index for text. Repeat to provide fallbacks (first non-empty wins).",
    )
    parser.add_argument(
        "--audio-column",
        type=str,
        default=None,
        help="Column name/index pointing to the audio filename. Optional when --audio-pattern is set.",
    )
    parser.add_argument(
        "--audio-pattern",
        type=str,
        default="{id}.wav",
        help=(
            "Format string used to build audio filenames when --audio-column is missing. "
            "Receives every column as a key plus 'id'."
        ),
    )
    parser.add_argument(
        "--speaker-column",
        type=str,
        default='speaker',
        help="Column name/index for speaker IDs (optional).",
    )
    parser.add_argument(
        "--language-column",
        type=str,
        default='language',
        help="Column name/index for per-utterance language code (optional).",
    )
    parser.add_argument(
        "--duration-column",
        type=str,
        default=None,
        help="Column name/index for durations in seconds (optional).",
    )
    parser.add_argument(
        "--default-language",
        type=str,
        default=None,
        help="Language code to use when the row is missing language metadata (optional).",
    )
    parser.add_argument(
        "--default-speaker",
        type=str,
        default=None,
        help="Speaker ID applied whenever the metadata row lacks a speaker field (optional).",
    )
    parser.add_argument(
        "--store-relative",
        action="store_true",
        help="Store audio paths relative to --audio-root when possible (default: absolute paths).",
    )
    parser.add_argument(
        "--skip-missing-audio",
        action="store_true",
        help="Skip rows whose audio file does not exist instead of raising an error.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on the number of rows to convert (0 = use every row).",
    )
    return parser.parse_args()


RowType = Union[Dict[str, str], List[str]]


def is_dict_row(row: RowType) -> bool:
    return isinstance(row, dict)


def get_fieldnames(reader: csv.reader) -> Optional[List[str]]:
    return getattr(reader, "fieldnames", None)


def parse_index(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fetch_value(
    row: RowType,
    field: Optional[str],
    fieldnames: Optional[Sequence[str]],
) -> Optional[str]:
    if field is None:
        return None
    if is_dict_row(row):
        value = row.get(field)
        if value is not None:
            return value
        idx = parse_index(field)
        if idx is not None and fieldnames and 0 <= idx < len(fieldnames):
            return row.get(fieldnames[idx])
        return None
    idx = parse_index(field)
    if idx is None:
        raise ValueError("Column references by name require a header. Re-run without --no-header or use indices.")
    if 0 <= idx < len(row):
        return row[idx]
    return None


def first_non_empty(
    row: RowType,
    columns: Optional[Sequence[str]],
    fieldnames: Optional[Sequence[str]],
) -> Optional[str]:
    if not columns:
        return None
    for column in columns:
        raw = fetch_value(row, column, fieldnames)
        if raw is None:
            continue
        stripped = raw.strip()
        if stripped:
            return stripped
    return None


def ensure_audio_reference(
    raw_value: str,
    audio_root: Path,
    *,
    store_relative: bool,
) -> tuple[str, Path]:
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = (audio_root / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if store_relative:
        try:
            rel = candidate.relative_to(audio_root.resolve())
            audio_field = rel.as_posix()
        except ValueError:
            audio_field = candidate.as_posix()
    else:
        audio_field = candidate.as_posix()
    return audio_field, candidate


def build_audio_value(
    row: RowType,
    *,
    args: argparse.Namespace,
    fieldnames: Optional[Sequence[str]],
    context: Dict[str, str],
) -> str:
    if args.audio_column:
        value = fetch_value(row, args.audio_column, fieldnames)
        if value is None or not value.strip():
            raise ValueError("Missing audio filename in column '%s'" % args.audio_column)
        return value.strip()

    try:
        return args.audio_pattern.format(**context)
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(
            f"Audio pattern references unknown key '{missing}'. Available keys: {sorted(context.keys())}"
        ) from exc


def normalize_id(raw_id: Optional[str], audio_path: Path, *, used_ids: Dict[str, int]) -> str:
    base = (raw_id or audio_path.stem).strip()
    if not base:
        base = audio_path.stem or "sample"
    base = base.replace(" ", "_")
    count = used_ids.get(base, 0)
    used_ids[base] = count + 1
    if count == 0:
        return base
    return f"{base}_{count+1}"


def load_rows(args: argparse.Namespace) -> Iterable[RowType]:
    open_kwargs = {"newline": "", "encoding": args.encoding}
    with args.metadata.open("r", **open_kwargs) as handle:
        if args.no_header:
            reader = csv.reader(handle, delimiter=args.delimiter, quotechar=args.quotechar)
            yield from reader
        else:
            reader = csv.DictReader(handle, delimiter=args.delimiter, quotechar=args.quotechar)
            for row in reader:
                yield row


def row_context(row: RowType, fieldnames: Optional[Sequence[str]]) -> Dict[str, str]:
    if is_dict_row(row):
        return {k: (v or "") for k, v in row.items()}
    return {f"col{i}": (value or "") for i, value in enumerate(row)}


def main() -> None:
    args = parse_args()
    if args.audio_column is None and not args.audio_pattern:
        print("[Error] Provide either --audio-column or --audio-pattern to locate audio files.", file=sys.stderr)
        sys.exit(2)

    if args.text_columns is None:
        # Default to LJ Speech style: first non-ID column.
        args.text_columns = ["1", "text", "normalized_text"]

    audio_root = args.audio_root.expanduser().resolve()
    if not audio_root.exists() or not audio_root.is_dir():
        print(f"[Error] Audio root does not exist or is not a directory: {audio_root}", file=sys.stderr)
        sys.exit(2)

    args.metadata = args.metadata.expanduser().resolve()
    if not args.metadata.exists():
        print(f"[Error] Metadata file not found: {args.metadata}", file=sys.stderr)
        sys.exit(2)

    total = 0
    written = 0
    skipped_missing_audio = 0
    duplicate_ids = 0
    used_ids: Dict[str, int] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    open_kwargs = {"mode": "w", "encoding": "utf-8"}
    with args.output.open(**open_kwargs) as sink:
        rows = load_rows(args)
        fieldnames = None

        # Peek for DictReader fieldnames when headers are present
        if not args.no_header:
            with args.metadata.open("r", newline="", encoding=args.encoding) as handle:
                reader = csv.DictReader(handle, delimiter=args.delimiter, quotechar=args.quotechar)
                fieldnames = reader.fieldnames

        for row in rows:
            if args.max_samples and written >= args.max_samples:
                break

            # Skip empty rows
            if is_dict_row(row):
                values = list(row.values())
            else:
                values = list(row)
            if not any(value and value.strip() for value in values):
                continue

            total += 1
            context = row_context(row, fieldnames)

            raw_text = first_non_empty(row, args.text_columns, fieldnames)
            if not raw_text:
                print(f"[Warn] Row {total} missing text; skipping.", file=sys.stderr)
                continue

            raw_id = fetch_value(row, args.id_column, fieldnames) if args.id_column else None
            if raw_id:
                context.setdefault("id", raw_id.strip())

            try:
                audio_value = build_audio_value(row, args=args, fieldnames=fieldnames, context=context)
            except ValueError as exc:
                print(f"[Warn] Row {total} skipped: {exc}", file=sys.stderr)
                continue

            audio_field, audio_path = ensure_audio_reference(
                audio_value,
                audio_root,
                store_relative=args.store_relative,
            )

            if not audio_path.exists():
                message = f"[Warn] Audio file not found for row {total}: {audio_path}"
                if args.skip_missing_audio:
                    skipped_missing_audio += 1
                    print(message + " (skipped)", file=sys.stderr)
                    continue
                raise FileNotFoundError(message)

            sample_id = normalize_id(raw_id, audio_path, used_ids=used_ids)
            if sample_id != (raw_id or audio_path.stem):
                duplicate_ids += 1

            speaker = fetch_value(row, args.speaker_column, fieldnames)
            language = fetch_value(row, args.language_column, fieldnames)
            duration = fetch_value(row, args.duration_column, fieldnames)

            record = {
                "id": sample_id,
                "audio": audio_field,
                "text": raw_text,
            }
            if args.default_speaker and not (speaker and speaker.strip()):
                speaker = args.default_speaker
            if speaker:
                record["speaker"] = speaker.strip()

            if args.default_language and not (language and language.strip()):
                language = args.default_language
            if language:
                record["language"] = language.strip()

            if duration:
                try:
                    record["duration"] = float(duration)
                except ValueError:
                    pass

            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"[Done] Processed {total} rows -> wrote {written} manifest entries "
        f"(missing-audio skipped: {skipped_missing_audio}, deduped IDs: {duplicate_ids})."
    )


if __name__ == "__main__":
    main()
