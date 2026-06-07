#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    keywords: tuple[str, ...]
    signal_names: str
    resample_to: float
    input_smooth_sec: float
    target_smooth_sec: float
    output_subdir: str
    output_csv: str
    min_duration_sec: float


DATASETS: dict[str, DatasetSpec] = {
    "bidmc": DatasetSpec(
        name="bidmc",
        keywords=("bidmc",),
        signal_names="RESP",
        resample_to=25.0,
        input_smooth_sec=2.0,
        target_smooth_sec=2.0,
        output_subdir="quasi_bidmc_resp_ma2s",
        output_csv="bidmc_resp_ma2s.csv",
        min_duration_sec=120.0,
    ),
    "fantasia": DatasetSpec(
        name="fantasia",
        keywords=("fantasia",),
        signal_names="RESP",
        resample_to=25.0,
        input_smooth_sec=2.0,
        target_smooth_sec=2.0,
        output_subdir="quasi_fantasia_resp_ma2s",
        output_csv="fantasia_resp_ma2s.csv",
        min_duration_sec=300.0,
    ),
    "mitdb": DatasetSpec(
        name="mitdb",
        keywords=("mitdb", "mit-bih", "mit_bih", "arrhythmia"),
        signal_names="MLII",
        resample_to=125.0,
        input_smooth_sec=0.08,
        target_smooth_sec=0.08,
        output_subdir="quasi_mitdb_mlii_ma008s",
        output_csv="mitdb_mlii_ma008s.csv",
        min_duration_sec=300.0,
    ),
}


ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")


def _is_archive(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _looks_extracted(path: Path, dataset: str) -> bool:
    if not path.is_dir():
        return False
    if dataset == "bidmc" and any(path.rglob("*_Signals.csv")):
        return True
    return any(path.rglob("*.hea"))


def _find_source(spec: DatasetSpec, data_root: Path, extract_root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Explicit {spec.name} source does not exist: {path}")
        return path

    extracted_dataset = extract_root / spec.name
    if _looks_extracted(extracted_dataset, spec.name):
        return extracted_dataset

    candidates: list[Path] = []
    for path in sorted(data_root.rglob("*")):
        lower = path.name.lower()
        if not any(keyword in lower for keyword in spec.keywords):
            continue
        if path.is_dir() and _looks_extracted(path, spec.name):
            candidates.append(path)
        elif path.is_file() and _is_archive(path):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a source for {spec.name} under {data_root}. "
            f"Pass --{spec.name}-source explicitly."
        )
    candidates.sort(key=lambda p: (0 if p.is_dir() else 1, len(str(p)), str(p)))
    return candidates[0]


def _run(cmd: list[str], dry_run: bool) -> None:
    print(shlex.join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _prepare_command(
    spec: DatasetSpec,
    source: Path,
    args: argparse.Namespace,
    repo_root: Path,
) -> tuple[list[str], Path]:
    output_dir = Path(args.output_root) / spec.output_subdir
    output_csv = output_dir / spec.output_csv
    quality_csv = output_csv.with_suffix(".quality.csv")
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "prepare_quasiperiodic_wave_dataset.py"),
        "--dataset",
        spec.name,
        "--sources",
        str(source),
        "--extract-dir",
        str(Path(args.extract_root) / spec.name),
        "--signal-names",
        spec.signal_names,
        "--resample-to",
        str(spec.resample_to),
        "--input-smooth-sec",
        str(spec.input_smooth_sec),
        "--input-smooth-mode",
        "causal",
        "--target-smooth-sec",
        str(spec.target_smooth_sec),
        "--target-smooth-mode",
        "centered",
        "--split-policy",
        "by_record",
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--min-finite-ratio",
        str(args.min_finite_ratio),
        "--min-std",
        str(args.min_std),
        "--min-duration-sec",
        str(spec.min_duration_sec),
        "--bad-record-policy",
        args.bad_record_policy,
        "--robust-clip-z",
        str(args.robust_clip_z),
        "--quality-output",
        str(quality_csv),
        "--output",
        str(output_csv),
    ]
    if args.record_limit > 0:
        cmd.extend(["--record-limit", str(args.record_limit)])
    if args.clip_quantile_low is not None:
        cmd.extend(["--clip-quantile-low", str(args.clip_quantile_low)])
    if args.clip_quantile_high is not None:
        cmd.extend(["--clip-quantile-high", str(args.clip_quantile_high)])
    return cmd, output_csv


def _profile_command(output_csv: Path, repo_root: Path) -> list[str]:
    profile_dir = output_csv.parent / "profile"
    return [
        sys.executable,
        str(repo_root / "scripts" / "analyze_quasiperiodic_profile.py"),
        "--csv",
        str(output_csv),
        "--signal-cols",
        "target_smooth",
        "--time-col",
        "time",
        "--fs-col",
        "fs",
        "--segment-col",
        "segment_id",
        "--split-col",
        "split",
        "--split-values",
        "train",
        "--output-dir",
        str(profile_dir),
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and clean the public quasi-periodic datasets used in the runbook.")
    parser.add_argument("--data-root", default="./data", help="Root containing downloaded archives or extracted dataset folders.")
    parser.add_argument("--extract-root", default="./data/extracted", help="Persistent extraction root.")
    parser.add_argument("--output-root", default="./outputs", help="Prepared CSV output root.")
    parser.add_argument("--datasets", default="bidmc,fantasia,mitdb", help="Comma-separated subset: bidmc,fantasia,mitdb.")
    parser.add_argument("--bidmc-source", default=None)
    parser.add_argument("--fantasia-source", default=None)
    parser.add_argument("--mitdb-source", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-finite-ratio", type=float, default=0.995)
    parser.add_argument("--min-std", type=float, default=1e-10)
    parser.add_argument("--bad-record-policy", choices=["error", "skip"], default="error")
    parser.add_argument("--robust-clip-z", type=float, default=12.0)
    parser.add_argument("--clip-quantile-low", type=float, default=None)
    parser.add_argument("--clip-quantile-high", type=float, default=None)
    parser.add_argument("--record-limit", type=int, default=0, help="Debug only: keep first N discovered records per dataset.")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root)
    extract_root = Path(args.extract_root)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    names = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]
    for name in names:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset '{name}'. Choose from: {', '.join(DATASETS)}")
        spec = DATASETS[name]
        explicit = getattr(args, f"{name}_source")
        source = _find_source(spec, data_root, extract_root, explicit)
        print(f"\n# {name}: source={source}")
        prepare_cmd, output_csv = _prepare_command(spec, source, args, repo_root)
        _run(prepare_cmd, args.dry_run)
        if not args.skip_profile:
            _run(_profile_command(output_csv, repo_root), args.dry_run)


if __name__ == "__main__":
    main()
