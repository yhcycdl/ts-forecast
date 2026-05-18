#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np


@dataclass
class SignalRecord:
    dataset: str
    record_id: str
    signal_name: str
    fs: float
    values: np.ndarray
    source_path: str


def _parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "signal"


def _moving_average(values: np.ndarray, window: int, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if window <= 1:
        return values.copy()
    window = int(window)
    cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    if mode == "causal":
        ends = np.arange(1, values.size + 1, dtype=np.int64)
        starts = np.maximum(0, ends - window)
    elif mode == "centered":
        half_left = window // 2
        half_right = window - half_left - 1
        centers = np.arange(values.size, dtype=np.int64)
        starts = np.maximum(0, centers - half_left)
        ends = np.minimum(values.size, centers + half_right + 1)
    else:
        raise ValueError(f"Unsupported smoothing mode: {mode}")
    counts = ends - starts
    return (cumsum[ends] - cumsum[starts]) / counts


def _moving_rms(values: np.ndarray, window: int, mode: str) -> np.ndarray:
    return np.sqrt(np.maximum(_moving_average(np.square(values), window, mode), 0.0))


def _infer_fs(time: np.ndarray) -> float | None:
    if time.size < 2:
        return None
    diff = np.diff(time)
    diff = diff[np.isfinite(diff) & (diff > 0)]
    if diff.size == 0:
        return None
    return float(1.0 / np.median(diff))


def _resample(values: np.ndarray, fs: float, target_fs: float | None) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if target_fs is None or target_fs <= 0 or abs(target_fs - fs) / max(fs, 1e-12) < 1e-6:
        return values, float(fs)
    old_t = np.arange(values.size, dtype=np.float64) / float(fs)
    new_len = int(math.floor(old_t[-1] * float(target_fs))) + 1
    if new_len < 2:
        raise ValueError("Resampling leaves fewer than 2 samples.")
    new_t = np.arange(new_len, dtype=np.float64) / float(target_fs)
    return np.interp(new_t, old_t, values), float(target_fs)


def _crop(values: np.ndarray, fs: float, start_sec: float | None, end_sec: float | None, max_duration_sec: float | None) -> np.ndarray:
    start = 0 if start_sec is None else max(0, int(round(start_sec * fs)))
    end = values.size if end_sec is None else min(values.size, int(round(end_sec * fs)))
    if max_duration_sec is not None and max_duration_sec > 0:
        end = min(end, start + int(round(max_duration_sec * fs)))
    if end <= start:
        raise ValueError("Crop range produced no samples.")
    return values[start:end]


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("Signal has no finite samples.")
    fill = float(np.nanmedian(values[finite]))
    values = np.where(finite, values, fill)
    return values


def _transform(values: np.ndarray, transform: str, rms_window: int, smooth_mode: str) -> np.ndarray:
    if transform == "none":
        return values
    if transform == "abs":
        return np.abs(values)
    if transform == "square":
        return np.square(values)
    if transform == "rms":
        return _moving_rms(values, max(1, rms_window), smooth_mode)
    raise ValueError(f"Unsupported transform: {transform}")


def _match_columns(columns: list[str], wanted: list[str]) -> list[str]:
    if not wanted:
        return []
    lower_map = {c.lower(): c for c in columns}
    selected: list[str] = []
    for item in wanted:
        key = item.lower()
        if key in lower_map:
            selected.append(lower_map[key])
            continue
        matches = [c for c in columns if key in c.lower()]
        if not matches:
            raise ValueError(f"Could not find signal '{item}' in columns/signals: {columns}")
        selected.append(matches[0])
    return selected


def _read_bidmc_csv_signals(root: Path, wanted: list[str]) -> list[SignalRecord]:
    records: list[SignalRecord] = []
    signal_files = sorted(root.rglob("*_Signals.csv"))
    if not signal_files:
        signal_files = sorted(root.rglob("*Signals.csv"))
    for path in signal_files:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            fieldnames = list(reader.fieldnames)
            time_col = None
            for name in fieldnames:
                if "time" in name.lower():
                    time_col = name
                    break
            numeric_cols = [c for c in fieldnames if c != time_col]
            selected = _match_columns(numeric_cols, wanted) if wanted else []
            if not selected:
                defaults = ["RESP", "resp", "Resp", "PLETH", "ppg", "II"]
                selected = _match_columns(numeric_cols, defaults[:1]) if any(c.lower() == "resp" for c in numeric_cols) else [numeric_cols[0]]

            buffers = {name: [] for name in selected}
            time_buf: list[float] = []
            for row in reader:
                if time_col is not None:
                    try:
                        time_buf.append(float(row[time_col]))
                    except Exception:
                        time_buf.append(float("nan"))
                for name in selected:
                    try:
                        buffers[name].append(float(row[name]))
                    except Exception:
                        buffers[name].append(float("nan"))
        fs = _infer_fs(np.asarray(time_buf, dtype=np.float64)) if time_buf else None
        if fs is None:
            fs = 125.0
        record_id = path.stem.replace("_Signals", "").replace("Signals", "")
        for name in selected:
            records.append(
                SignalRecord(
                    dataset="bidmc",
                    record_id=record_id,
                    signal_name=name,
                    fs=float(fs),
                    values=np.asarray(buffers[name], dtype=np.float64),
                    source_path=str(path),
                )
            )
    return records


def _read_wfdb_records(root: Path, dataset: str, wanted: list[str]) -> list[SignalRecord]:
    try:
        import wfdb  # type: ignore
    except Exception as exc:
        raise RuntimeError("Reading WFDB datasets requires: pip install wfdb") from exc

    records: list[SignalRecord] = []
    for header in sorted(root.rglob("*.hea")):
        stem = header.stem
        if dataset == "bidmc" and stem.endswith("n"):
            continue
        record_path = str(header.with_suffix(""))
        try:
            signals, fields = wfdb.rdsamp(record_path)
        except Exception:
            continue
        sig_names = [str(name) for name in fields.get("sig_name", [])]
        fs = float(fields.get("fs", 1.0))
        if not sig_names:
            sig_names = [f"ch{i}" for i in range(signals.shape[1])]
        selected_names = _match_columns(sig_names, wanted) if wanted else [sig_names[0]]
        for name in selected_names:
            idx = sig_names.index(name)
            records.append(
                SignalRecord(
                    dataset=dataset,
                    record_id=stem,
                    signal_name=name,
                    fs=fs,
                    values=np.asarray(signals[:, idx], dtype=np.float64),
                    source_path=str(header),
                )
            )
    return records


def _numeric_mat_arrays(obj, prefix: str = "") -> Iterable[tuple[str, np.ndarray]]:
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            for name in obj.dtype.names:
                yield from _numeric_mat_arrays(obj[name], f"{prefix}.{name}" if prefix else name)
            return
        if obj.dtype == object:
            for i, item in enumerate(obj.reshape(-1)):
                yield from _numeric_mat_arrays(item, f"{prefix}_{i}" if prefix else str(i))
            return
        if np.issubdtype(obj.dtype, np.number):
            arr = np.asarray(obj).squeeze()
            if arr.ndim == 1 and arr.size > 8:
                yield prefix, arr.astype(np.float64)


def _read_mat_records(root: Path, dataset: str, wanted: list[str]) -> list[SignalRecord]:
    try:
        from scipy.io import loadmat  # type: ignore
    except Exception as exc:
        raise RuntimeError("Reading MATLAB datasets requires: pip install scipy") from exc

    records: list[SignalRecord] = []
    for path in sorted(root.rglob("*.mat")):
        raw = loadmat(path, squeeze_me=True, struct_as_record=False)
        arrays: list[tuple[str, np.ndarray]] = []
        for key, value in raw.items():
            if key.startswith("__"):
                continue
            arrays.extend(_numeric_mat_arrays(value, key))
        if wanted:
            names = [name for name, _ in arrays]
            selected_names = _match_columns(names, wanted)
            arrays = [(name, arr) for name, arr in arrays if name in selected_names]
        elif dataset == "cwru":
            preferred = [(name, arr) for name, arr in arrays if "DE_time".lower() in name.lower()]
            arrays = preferred or arrays[:1]
        else:
            arrays = arrays[:1]
        for name, values in arrays:
            records.append(
                SignalRecord(
                    dataset=dataset,
                    record_id=path.stem,
                    signal_name=name,
                    fs=float("nan"),
                    values=np.asarray(values, dtype=np.float64),
                    source_path=str(path),
                )
            )
    return records


def _ensure_extracted(source: Path, extract_root: Path | None, temp_root: Path) -> Path:
    if source.is_dir():
        return source
    if source.suffix.lower() != ".zip":
        return source.parent
    if extract_root is None:
        target = temp_root / source.stem
    else:
        target = extract_root / source.stem
    marker = target / ".extract_complete"
    if marker.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source, "r") as zf:
        zf.extractall(target)
    marker.write_text("ok\n", encoding="utf-8")
    return target


def _load_records(args: argparse.Namespace, temp_root: Path) -> list[SignalRecord]:
    dataset = args.dataset.lower()
    wanted = _parse_csv_list(args.signal_names)
    extract_root = Path(args.extract_dir) if args.extract_dir else None
    all_records: list[SignalRecord] = []
    for source_arg in args.sources:
        source = Path(source_arg)
        root = _ensure_extracted(source, extract_root, temp_root)
        if dataset == "bidmc":
            records = _read_bidmc_csv_signals(root, wanted)
            if not records:
                records = _read_wfdb_records(root, dataset, wanted)
        elif dataset in {"fantasia", "mitdb"}:
            records = _read_wfdb_records(root, dataset, wanted)
        elif dataset in {"cwru", "mat"}:
            records = _read_mat_records(root, dataset, wanted)
        elif dataset == "generic_csv":
            records = _read_generic_csv_records(root, wanted, args.sample_rate)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        all_records.extend(records)
    if args.record_limit > 0:
        all_records = all_records[: args.record_limit]
    return all_records


def _read_generic_csv_records(root: Path, wanted: list[str], sample_rate: float | None) -> list[SignalRecord]:
    records: list[SignalRecord] = []
    for path in sorted(root.rglob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            fieldnames = list(reader.fieldnames)
            time_col = next((c for c in fieldnames if "time" in c.lower()), None)
            numeric_cols = [c for c in fieldnames if c != time_col]
            selected = _match_columns(numeric_cols, wanted) if wanted else numeric_cols[:1]
            buffers = {name: [] for name in selected}
            time_buf: list[float] = []
            for row in reader:
                if time_col is not None:
                    try:
                        time_buf.append(float(row[time_col]))
                    except Exception:
                        time_buf.append(float("nan"))
                for name in selected:
                    try:
                        buffers[name].append(float(row[name]))
                    except Exception:
                        buffers[name].append(float("nan"))
        fs = sample_rate or (_infer_fs(np.asarray(time_buf, dtype=np.float64)) if time_buf else None)
        if fs is None:
            raise ValueError(f"Cannot infer sample rate for {path}; pass --sample-rate.")
        for name in selected:
            records.append(
                SignalRecord(
                    dataset="generic_csv",
                    record_id=path.stem,
                    signal_name=name,
                    fs=float(fs),
                    values=np.asarray(buffers[name], dtype=np.float64),
                    source_path=str(path),
                )
            )
    return records


def _split_for_segment(index: int, count: int, length: int, args: argparse.Namespace) -> np.ndarray:
    policy = args.split_policy
    if policy == "chronological":
        train_end = int(length * args.train_ratio)
        val_end = int(length * (args.train_ratio + args.val_ratio))
        if train_end <= 0 or val_end <= train_end or val_end >= length:
            raise ValueError("Invalid chronological split ratios.")
        split = np.full(length, "test", dtype=object)
        split[:train_end] = "train"
        split[train_end:val_end] = "val"
        return split

    train_end = int(count * args.train_ratio)
    val_end = int(count * (args.train_ratio + args.val_ratio))
    if index < train_end:
        label = "train"
    elif index < val_end:
        label = "val"
    else:
        label = "test"
    return np.full(length, label, dtype=object)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare quasi-periodic signal datasets for smooth long-horizon waveform forecasting.")
    parser.add_argument("--dataset", required=True, choices=["bidmc", "fantasia", "mitdb", "cwru", "mat", "generic_csv"])
    parser.add_argument("--sources", nargs="+", required=True, help="Input ZIP files or extracted directories.")
    parser.add_argument("--signal-names", default=None, help="Comma-separated signal names or substrings, e.g. RESP,PLETH or MLII.")
    parser.add_argument("--output", required=True, help="Output long-format CSV path.")
    parser.add_argument("--extract-dir", default=None, help="Optional persistent extraction directory for ZIP sources.")
    parser.add_argument("--list-signals", action="store_true", help="Only list discovered records/signals; do not write CSV.")
    parser.add_argument("--record-limit", type=int, default=0, help="Use only the first N discovered records/signals; 0 means all.")
    parser.add_argument("--sample-rate", type=float, default=None, help="Override sample rate; required for some MAT/generic CSV files.")
    parser.add_argument("--resample-to", type=float, default=None, help="Optional target sample rate in Hz.")
    parser.add_argument("--time-start-sec", type=float, default=None)
    parser.add_argument("--time-end-sec", type=float, default=None)
    parser.add_argument("--max-duration-sec", type=float, default=None)
    parser.add_argument("--transform", choices=["none", "abs", "square", "rms"], default="none")
    parser.add_argument("--rms-window-sec", type=float, default=0.05)
    parser.add_argument("--input-smooth-sec", type=float, default=2.0)
    parser.add_argument("--input-smooth-mode", choices=["causal", "centered"], default="causal")
    parser.add_argument("--target-smooth-sec", type=float, default=2.0)
    parser.add_argument("--target-smooth-mode", choices=["causal", "centered"], default="centered")
    parser.add_argument("--split-policy", choices=["chronological", "by_record"], default="by_record")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_path = Path(args.output)
    with TemporaryDirectory() as tmp:
        records = _load_records(args, Path(tmp))

    if not records:
        raise ValueError("No records/signals were discovered.")

    if args.list_signals:
        for i, record in enumerate(records):
            fs = args.sample_rate if args.sample_rate is not None else record.fs
            print(
                f"{i:04d} dataset={record.dataset} record={record.record_id} "
                f"signal={record.signal_name} fs={fs:g} n={record.values.size} source={record.source_path}"
            )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    segments = []
    fieldnames = ["time", "raw", "input_smooth", "target_smooth", "split", "segment_id", "record_id", "signal_name", "dataset", "fs"]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, record in enumerate(records):
            fs = float(args.sample_rate) if args.sample_rate is not None else float(record.fs)
            if not np.isfinite(fs) or fs <= 0:
                raise ValueError(f"Sample rate is unknown for {record.source_path}; pass --sample-rate.")
            values = _standardize(record.values)
            values = _crop(values, fs, args.time_start_sec, args.time_end_sec, args.max_duration_sec)
            values, fs = _resample(values, fs, args.resample_to)
            rms_window = max(1, int(round(args.rms_window_sec * fs)))
            values = _transform(values, args.transform, rms_window, args.input_smooth_mode)
            input_window = max(1, int(round(args.input_smooth_sec * fs)))
            target_window = max(1, int(round(args.target_smooth_sec * fs)))
            input_smooth = _moving_average(values, input_window, args.input_smooth_mode)
            target_smooth = _moving_average(values, target_window, args.target_smooth_mode)
            split = _split_for_segment(idx, len(records), values.size, args)
            segment_id = f"{_safe_name(record.dataset)}_{_safe_name(record.record_id)}_{_safe_name(record.signal_name)}"
            time = np.arange(values.size, dtype=np.float64) / fs
            for i in range(values.size):
                writer.writerow(
                    {
                        "time": float(time[i]),
                        "raw": float(values[i]),
                        "input_smooth": float(input_smooth[i]),
                        "target_smooth": float(target_smooth[i]),
                        "split": str(split[i]),
                        "segment_id": segment_id,
                        "record_id": record.record_id,
                        "signal_name": record.signal_name,
                        "dataset": record.dataset,
                        "fs": float(fs),
                    }
                )
            rows_written += int(values.size)
            segments.append(
                {
                    "segment_id": segment_id,
                    "dataset": record.dataset,
                    "record_id": record.record_id,
                    "signal_name": record.signal_name,
                    "source_path": record.source_path,
                    "fs": float(fs),
                    "rows": int(values.size),
                    "duration_sec": float(values.size / fs),
                    "train_rows": int(np.sum(split == "train")),
                    "val_rows": int(np.sum(split == "val")),
                    "test_rows": int(np.sum(split == "test")),
                }
            )

    config = {
        "dataset": args.dataset,
        "sources": [str(Path(p).resolve()) for p in args.sources],
        "signal_names": _parse_csv_list(args.signal_names),
        "split_policy": args.split_policy,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "sample_rate": args.sample_rate,
        "resample_to": args.resample_to,
        "transform": args.transform,
        "input_smooth_sec": args.input_smooth_sec,
        "input_smooth_mode": args.input_smooth_mode,
        "target_smooth_sec": args.target_smooth_sec,
        "target_smooth_mode": args.target_smooth_mode,
        "output_csv": str(output_path.resolve()),
        "rows_written": rows_written,
        "num_segments": len(segments),
        "segments": segments,
    }
    config_path = output_path.with_suffix(".config.json")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared quasi-periodic waveform CSV: {output_path}")
    print(f"Config JSON: {config_path}")
    print(f"Segments: {len(segments)}")
    print(f"Rows written: {rows_written}")
    print("Input column: input_smooth")
    print("Target column: target_smooth")
    print("Raw plot column: raw")


if __name__ == "__main__":
    main()
