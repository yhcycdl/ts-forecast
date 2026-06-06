# data_provider/data_factory.py
import os
import numpy as np
from torch.utils.data import DataLoader

from data_provider.processing import (
    load_dataframe,
    select_feature_values,
    time_split,
    DataScaler,
    GlobalScaler,
)
from data_provider.data_custom import WindowDataset


def _parse_csv_cols(value):
    if value is None:
        return None
    cols = [c.strip() for c in str(value).split(",") if c.strip()]
    return cols if cols else None

def _infer_columns(args, df_cols):
    if getattr(args, "input_cols", None) is not None or getattr(args, "output_cols", None) is not None:
        input_cols = _parse_csv_cols(getattr(args, "input_cols", None)) or []
        output_cols = _parse_csv_cols(getattr(args, "output_cols", None)) or []
        merged = []
        seen = set()
        for col in input_cols + output_cols:
            if col not in seen:
                merged.append(col)
                seen.add(col)
        return merged if merged else None
    # 你也可以更复杂：支持 MS/M 特征组合
    if args.col_names is not None:
        return [c.strip() for c in args.col_names.split(",") if c.strip() != ""]
    # 默认：S 用 target 一列；M 用所有数值列（在 processing.load_csv 里已经是 all numeric）
    if args.features == "S":
        return [args.target]
    return None  # None 表示用所有数值列

def _load_values(args):
    csv_path = os.path.join(args.root_path, args.data_path)
    df = load_dataframe(csv_path, max_rows=args.max_rows)
    col_names = None
    if args.col_names is not None or args.features == "S":
        col_names = _infer_columns(args, None)

    if col_names is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        feature_cols = list(col_names)
    values = select_feature_values(df, col_names=feature_cols)
    return df, values, feature_cols


def _load_sample_weights(args, df):
    weight_col = getattr(args, "sample_weight_col", None)
    if weight_col is None:
        return None
    if weight_col not in df.columns:
        raise ValueError(f"sample_weight_col '{weight_col}' not found in input CSV.")
    weights = df[weight_col].to_numpy(dtype=np.float32).reshape(-1)
    weights = np.maximum(weights, 0.0)
    scale = float(getattr(args, "sample_weight_scale", 1.0))
    bias = float(getattr(args, "sample_weight_bias", 1.0))
    return bias + scale * weights


def _resolve_forecast_io_indices(args, feature_cols):
    input_cols = _parse_csv_cols(getattr(args, "input_cols", None))
    output_cols = _parse_csv_cols(getattr(args, "output_cols", None))
    if input_cols is not None or output_cols is not None:
        if not input_cols:
            raise ValueError("Explicit forecast IO requires --input_cols.")
        if not output_cols:
            raise ValueError("Explicit forecast IO requires --output_cols.")
        missing_inputs = [col for col in input_cols if col not in feature_cols]
        missing_outputs = [col for col in output_cols if col not in feature_cols]
        if missing_inputs or missing_outputs:
            raise ValueError(
                f"Explicit IO columns not found in feature columns. "
                f"missing_inputs={missing_inputs}, missing_outputs={missing_outputs}, feature_cols={feature_cols}"
            )
        return (
            np.array([feature_cols.index(col) for col in input_cols], dtype=np.int64),
            np.array([feature_cols.index(col) for col in output_cols], dtype=np.int64),
        )

    feature_mode = str(getattr(args, "features", "S")).upper()
    num_features = len(feature_cols)
    if feature_mode == "S":
        return np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)
    if feature_mode == "M":
        full = np.arange(num_features, dtype=np.int64)
        return full, full
    if feature_mode == "MS":
        target = getattr(args, "target", None)
        if target is None:
            raise ValueError("features='MS' requires --target.")
        if target not in feature_cols:
            raise ValueError(f"target '{target}' not found in feature columns: {feature_cols}")
        return np.arange(num_features, dtype=np.int64), np.array([feature_cols.index(target)], dtype=np.int64)
    raise ValueError(f"Unknown features mode: {feature_mode}")


def _build_scaler(args):
    return DataScaler() if args.scaler == "channel" else GlobalScaler()


def _build_forecast_loader(args, ds, flag):
    if flag == "train":
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
            drop_last=bool(args.drop_last),
        )

    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=bool(args.pin_memory),
        drop_last=False,
    )


def _require_non_empty_windows(ds, args, flag):
    if len(ds) > 0:
        return
    target_shift = int(getattr(args, "target_shift", 0))
    need = int(getattr(args, "seq_len", 0)) + target_shift + int(getattr(args, "pred_len", 0))
    raise ValueError(
        f"{flag} split produced 0 forecast windows. "
        f"Need at least seq_len + target_shift + pred_len = {need} contiguous rows per segment. "
        "Reduce seq_len/pred_len/target_shift, increase the split size, or check split_col/segment_col."
    )


def _count_window_starts(length: int, args, target_shift: int, window_mode: str, stride: int) -> int:
    if window_mode == "center":
        n = (int(length) - int(args.seq_len)) // int(stride) + 1
    else:
        n = (int(length) - int(args.seq_len) - int(target_shift) - int(args.pred_len)) // int(stride) + 1
    return max(0, int(n))


def _split_column_arrays(args, values, split_labels, segment_labels, flag, sample_weights=None):
    flag = str(flag).lower()
    split_labels = np.asarray(split_labels, dtype=object)
    segment_labels = np.asarray(segment_labels, dtype=object)
    mask = np.char.lower(split_labels.astype(str)) == flag
    target_shift = int(getattr(args, "target_shift", 0))
    window_mode = str(getattr(args, "window_mode", "past")).lower()
    if flag == "train":
        start_stride = int(getattr(args, "stride", 1))
    else:
        eval_stride = int(getattr(args, "eval_stride", -1))
        start_stride = int(getattr(args, "stride", 1)) if eval_stride == 0 else eval_stride
        if start_stride < 1:
            start_stride = int(getattr(args, "pred_len", 1))

    chunks = []
    weight_chunks = []
    start_indices = []
    offset = 0
    i = 0
    total = len(values)
    while i < total:
        if not mask[i]:
            i += 1
            continue
        seg = segment_labels[i]
        j = i + 1
        while j < total and mask[j] and segment_labels[j] == seg:
            j += 1
        chunk = values[i:j]
        chunks.append(chunk)
        if sample_weights is not None:
            weight_chunks.append(sample_weights[i:j])
        n_starts = _count_window_starts(len(chunk), args, target_shift, window_mode, start_stride)
        if n_starts > 0:
            start_indices.append(offset + np.arange(n_starts, dtype=np.int64) * start_stride)
        offset += len(chunk)
        i = j

    if not chunks:
        return None, None, np.asarray([], dtype=np.int64)

    split_values = np.concatenate(chunks, axis=0)
    split_weights = np.concatenate(weight_chunks, axis=0) if sample_weights is not None else None
    if start_indices:
        starts = np.concatenate(start_indices, axis=0)
    else:
        starts = np.asarray([], dtype=np.int64)
    return split_values, split_weights, starts


def _forecast_data_provider(args, flag):
    df, values, feature_cols = _load_values(args)
    in_indices, out_indices = _resolve_forecast_io_indices(args, feature_cols)
    target_shift = int(getattr(args, "target_shift", 0))
    window_mode = str(getattr(args, "window_mode", "past")).lower()
    center_left = getattr(args, "center_left", None)
    center_left = None if center_left is None or int(center_left) < 0 else int(center_left)
    sample_weights = _load_sample_weights(args, df)
    split_col = getattr(args, "split_col", None)
    segment_col = getattr(args, "segment_col", None)

    if split_col is not None:
        if split_col not in df.columns:
            raise ValueError(f"split_col '{split_col}' not found in input CSV.")
        if segment_col is None:
            segment_labels = np.zeros(len(df), dtype=object)
        else:
            if segment_col not in df.columns:
                raise ValueError(f"segment_col '{segment_col}' not found in input CSV.")
            segment_labels = df[segment_col].to_numpy(dtype=object)
        split_labels = df[split_col].to_numpy(dtype=object)

        train_raw, train_w, train_starts = _split_column_arrays(args, values, split_labels, segment_labels, "train", sample_weights)
        val_raw, val_w, val_starts = _split_column_arrays(args, values, split_labels, segment_labels, "val", sample_weights)
        test_raw, test_w, test_starts = _split_column_arrays(args, values, split_labels, segment_labels, "test", sample_weights)
        if train_raw is None or test_raw is None:
            raise ValueError("split_col mode requires non-empty train and test splits.")

        scaler = _build_scaler(args)
        scaler.fit(train_raw)
        train_norm = scaler.transform(train_raw)
        val_norm = scaler.transform(val_raw) if val_raw is not None else None
        test_norm = scaler.transform(test_raw)

        if flag == "train":
            ds = WindowDataset(train_norm, args.seq_len, args.pred_len, stride=args.stride,
                               in_indices=in_indices, out_indices=out_indices, target_shift=target_shift, sample_weights=train_w,
                               window_mode=window_mode, center_left=center_left, start_indices=train_starts)
            _require_non_empty_windows(ds, args, flag)
            return ds, _build_forecast_loader(args, ds, flag), scaler

        if flag == "val":
            if val_norm is None:
                raise ValueError("No val split rows found.")
            ds = WindowDataset(val_norm, args.seq_len, args.pred_len, stride=args.stride,
                               in_indices=in_indices, out_indices=out_indices, target_shift=target_shift, sample_weights=val_w,
                               window_mode=window_mode, center_left=center_left, start_indices=val_starts)
            _require_non_empty_windows(ds, args, flag)
            return ds, _build_forecast_loader(args, ds, flag), scaler

        if flag == "test":
            ds = WindowDataset(test_norm, args.seq_len, args.pred_len, stride=args.stride,
                               in_indices=in_indices, out_indices=out_indices, target_shift=target_shift, sample_weights=test_w,
                               window_mode=window_mode, center_left=center_left, start_indices=test_starts)
            _require_non_empty_windows(ds, args, flag)
            return ds, _build_forecast_loader(args, ds, flag), scaler

        raise ValueError(f"Unknown flag: {flag}")

    train_raw, val_raw, test_raw = time_split(
        values,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_mode=getattr(args, "split_mode", "total"),
    )
    if sample_weights is not None:
        train_w, val_w, test_w = time_split(
            sample_weights,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            split_mode=getattr(args, "split_mode", "total"),
        )
    else:
        train_w = val_w = test_w = None

    scaler = _build_scaler(args)
    scaler.fit(train_raw)
    train_norm = scaler.transform(train_raw)
    val_norm = scaler.transform(val_raw) if val_raw is not None else None
    test_norm = scaler.transform(test_raw)

    if flag == "train":
        ds = WindowDataset(train_norm, args.seq_len, args.pred_len, stride=args.stride,
                           in_indices=in_indices, out_indices=out_indices, target_shift=target_shift, sample_weights=train_w,
                           window_mode=window_mode, center_left=center_left)
        _require_non_empty_windows(ds, args, flag)
        return ds, _build_forecast_loader(args, ds, flag), scaler

    if flag == "val":
        if val_norm is None:
            raise ValueError("val_ratio=0, no val set.")
        ds = WindowDataset(val_norm, args.seq_len, args.pred_len, stride=args.pred_len,
                           in_indices=in_indices, out_indices=out_indices, target_shift=target_shift, sample_weights=val_w,
                           window_mode=window_mode, center_left=center_left)
        _require_non_empty_windows(ds, args, flag)
        return ds, _build_forecast_loader(args, ds, flag), scaler

    if flag == "test":
        ds = WindowDataset(test_norm, args.seq_len, args.pred_len, stride=args.pred_len,
                           in_indices=in_indices, out_indices=out_indices, target_shift=target_shift, sample_weights=test_w,
                           window_mode=window_mode, center_left=center_left)
        _require_non_empty_windows(ds, args, flag)
        return ds, _build_forecast_loader(args, ds, flag), scaler

    raise ValueError(f"Unknown flag: {flag}")


def data_provider(args, flag):
    return _forecast_data_provider(args, flag)
