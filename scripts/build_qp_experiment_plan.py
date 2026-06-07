#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path


def _safe_name(value) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "unknown"


def _q(value) -> str:
    return shlex.quote(str(value))


def _format_command(parts: list[str]) -> str:
    return " \\\n  ".join(parts)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _profile_command(csv_path: str, profile_dir: Path, args: argparse.Namespace) -> str:
    parts = [
        "python scripts/analyze_quasiperiodic_profile.py",
        f"--csv {_q(csv_path)}",
        f"--signal-cols {_q(args.profile_signal_cols)}",
        f"--time-col {_q(args.time_col)}",
        f"--fs-col {_q(args.fs_col)}",
        f"--segment-col {_q(args.segment_col)}",
        f"--split-col {_q(args.split_col)}",
        f"--split-values {_q(args.profile_split_values)}",
        f"--output-dir {_q(profile_dir)}",
    ]
    if args.profile_max_samples_per_segment > 0:
        parts.append(f"--max-samples-per-segment {args.profile_max_samples_per_segment}")
    return _format_command(parts)


def _augment_command(csv_path: str, profile_csv: str, augmented_csv: Path, args: argparse.Namespace) -> str:
    parts = [
        "python scripts/augment_quasiperiodic_dataset.py",
        f"--csv {_q(csv_path)}",
        f"--profile-csv {_q(profile_csv)}",
        f"--profile-signal-col {_q(args.profile_signal_cols.split(',')[0].strip())}",
        f"--output {_q(augmented_csv)}",
        f"--raw-col {_q(args.raw_col)}",
        f"--time-col {_q(args.time_col)}",
        f"--fs-col {_q(args.fs_col)}",
        f"--segment-col {_q(args.segment_col)}",
        f"--split-col {_q(args.split_col)}",
        f"--feature-mode {_q(args.feature_mode)}",
        f"--input-smooth-mode {_q(args.augment_input_smooth_mode)}",
        f"--target-smooth-mode {_q(args.augment_target_smooth_mode)}",
    ]
    if args.augment_modules:
        parts.append(f"--modules {_q(args.augment_modules)}")
    return _format_command(parts)


def _recommend_command(
    csv_path: str,
    profile_csv: str,
    augmented_csv: Path,
    recommend_dir: Path,
    signal_type: str,
    args: argparse.Namespace,
) -> str:
    prefix = f"{args.model_id_prefix}_{_safe_name(signal_type)}" if args.model_id_prefix else _safe_name(signal_type)
    parts = [
        "python scripts/recommend_qp_config.py",
        f"--profile-csv {_q(profile_csv)}",
        f"--prepared-csv {_q(csv_path)}",
        f"--enhanced-csv {_q(augmented_csv)}",
        f"--output-dir {_q(recommend_dir)}",
        f"--model-id-prefix {_q(prefix)}",
        f"--input-col {_q(args.input_col)}",
        f"--output-col {_q(args.output_col)}",
        f"--raw-col {_q(args.raw_col)}",
        f"--split-col {_q(args.split_col)}",
        f"--segment-col {_q(args.segment_col)}",
        f"--batch-size {args.batch_size}",
        f"--learning-rate {args.learning_rate}",
        f"--train-epochs {args.train_epochs}",
        f"--gpu {args.gpu}",
    ]
    if args.include_smoothpec:
        parts.append("--include-smoothpec")
    if args.include_legacy_baselines:
        parts.append("--include-legacy-baselines")
    if args.input_cycles > 0:
        parts.append(f"--input-cycles {args.input_cycles}")
    if args.output_cycles > 0:
        parts.append(f"--output-cycles {args.output_cycles}")
    return _format_command(parts)


def _write_shell(path: Path, commands: list[tuple[str, str]]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for title, command in commands:
        lines.append(f"# {title}")
        lines.append(command)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build profile/augment/recommend scripts for type-split quasi-periodic experiments."
    )
    parser.add_argument("--split-metadata", required=True, help="split_by_type_metadata.json from split_quasiperiodic_dataset_by_type.py.")
    parser.add_argument("--output-dir", required=True, help="Experiment-plan output directory.")
    parser.add_argument("--include-types", default=None, help="Optional comma-separated type labels to include.")
    parser.add_argument("--profile-signal-cols", default="target_smooth")
    parser.add_argument("--profile-split-values", default="train")
    parser.add_argument("--profile-max-samples-per-segment", type=int, default=200_000)
    parser.add_argument("--input-col", default="input_smooth")
    parser.add_argument("--output-col", default="target_smooth")
    parser.add_argument("--raw-col", default="raw")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--fs-col", default="fs")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--segment-col", default="segment_id")
    parser.add_argument("--feature-mode", choices=["causal", "offline"], default="causal")
    parser.add_argument("--augment-modules", default="all")
    parser.add_argument("--augment-input-smooth-mode", choices=["causal", "centered"], default="causal")
    parser.add_argument("--augment-target-smooth-mode", choices=["causal", "centered"], default="centered")
    parser.add_argument("--model-id-prefix", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-epochs", type=int, default=40)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--input-cycles", type=int, default=0, help="Optional override passed to recommend_qp_config.py.")
    parser.add_argument("--output-cycles", type=int, default=0, help="Optional override passed to recommend_qp_config.py.")
    parser.add_argument("--include-smoothpec", action="store_true")
    parser.add_argument(
        "--include-legacy-baselines",
        action="store_true",
        help="Pass through restored GRU/CNNLSTM/CRNN/InceptionTime/FastTCN/SpectralCNN/TimeMixer baselines.",
    )
    parser.add_argument("--reprofile", action="store_true", help="Ignore profile subsets in split metadata and recompute profiles per type.")
    args = parser.parse_args()

    split_metadata = _read_json(Path(args.split_metadata))
    outputs = list(split_metadata.get("outputs", []))
    if not outputs:
        raise ValueError("split metadata contains no outputs.")

    include = set(_parse_list(args.include_types))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_commands: list[tuple[str, str]] = []
    train_commands: list[tuple[str, str]] = []
    manifest_items = []
    for item in outputs:
        signal_type = str(item.get("signal_type", "unknown"))
        if include and signal_type not in include:
            continue
        csv_path = item.get("csv")
        if not csv_path:
            continue

        type_dir = output_dir / _safe_name(signal_type)
        profile_dir = type_dir / "profile"
        recommend_dir = type_dir / "recommend"
        augmented_csv = type_dir / f"{Path(str(csv_path)).stem}_aug.csv"
        type_dir.mkdir(parents=True, exist_ok=True)

        profile_csv = item.get("profile_csv")
        if args.reprofile or not profile_csv:
            profile_csv = str(profile_dir / "profile_by_segment.csv")
            prepare_commands.append(
                (
                    f"profile {signal_type}",
                    _profile_command(str(csv_path), profile_dir, args),
                )
            )
        prepare_commands.append(
            (
                f"augment {signal_type}",
                _augment_command(str(csv_path), str(profile_csv), augmented_csv, args),
            )
        )
        prepare_commands.append(
            (
                f"recommend {signal_type}",
                _recommend_command(str(csv_path), str(profile_csv), augmented_csv, recommend_dir, signal_type, args),
            )
        )
        train_commands.append((f"train {signal_type}", f"bash {_q(recommend_dir / 'recommended_train_commands.sh')}"))
        manifest_items.append(
            {
                "signal_type": signal_type,
                "csv": str(csv_path),
                "profile_csv": str(profile_csv),
                "augmented_csv": str(augmented_csv),
                "recommend_dir": str(recommend_dir),
                "train_script": str(recommend_dir / "recommended_train_commands.sh"),
            }
        )

    if not manifest_items:
        raise ValueError("No experiment items selected.")

    prepare_path = output_dir / "prepare_qp_experiments.sh"
    train_path = output_dir / "run_all_train_commands.sh"
    manifest_path = output_dir / "qp_experiment_plan.json"
    _write_shell(prepare_path, prepare_commands)
    _write_shell(train_path, train_commands)
    manifest_path.write_text(
        json.dumps(
            {
                "split_metadata": str(Path(args.split_metadata)),
                "output_dir": str(output_dir),
                "items": manifest_items,
                "prepare_script": str(prepare_path),
                "train_script": str(train_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved prepare script: {prepare_path}")
    print(f"Saved train script: {train_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Experiment types: {', '.join(item['signal_type'] for item in manifest_items)}")


if __name__ == "__main__":
    main()
