# data_provider/data_factory.py
import os
import numpy as np
from torch.utils.data import DataLoader

from data_provider.processing import (
    load_dataframe,
    select_feature_values,
    time_split,
    time_split_bounds,
    DataScaler,
    GlobalScaler,
)
from data_provider.risk_labels import (
    fit_risk_label_config,
    compute_risk_window_stats,
    assign_risk_labels,
    compute_class_weights,
    derive_window_labels_from_point_labels,
    load_label_file,
    split_window_records_by_segment,
)
from data_provider.data_custom import WindowDataset, RiskWindowDataset


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
    label_mode = str(getattr(args, "label_mode", "generated")).lower()
    label_col = getattr(args, "label_col", None)

    if label_mode == "column" and label_col is not None:
        if col_names is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            col_names = [c for c in numeric_cols if c != label_col]
        elif label_col in col_names:
            raise ValueError("label_col must not be included in model input feature columns.")

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


def _build_risk_loader(args, ds, flag):
    return DataLoader(
        ds,
        batch_size=args.batch_size if flag == "train" else max(1, args.batch_size),
        shuffle=(flag == "train"),
        num_workers=args.num_workers if flag == "train" else 0,
        pin_memory=bool(args.pin_memory),
        drop_last=bool(args.drop_last) if flag == "train" else False,
    )


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
            return ds, _build_forecast_loader(args, ds, flag), scaler

        if flag == "val":
            if val_norm is None:
                raise ValueError("No val split rows found.")
            ds = WindowDataset(val_norm, args.seq_len, args.pred_len, stride=args.stride,
                               in_indices=in_indices, out_indices=out_indices, target_shift=target_shift, sample_weights=val_w,
                               window_mode=window_mode, center_left=center_left, start_indices=val_starts)
            return ds, _build_forecast_loader(args, ds, flag), scaler

        if flag == "test":
            ds = WindowDataset(test_norm, args.seq_len, args.pred_len, stride=args.stride,
                               in_indices=in_indices, out_indices=out_indices, target_shift=target_shift, sample_weights=test_w,
                               window_mode=window_mode, center_left=center_left, start_indices=test_starts)
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
        return ds, _build_forecast_loader(args, ds, flag), scaler

    if flag == "val":
        if val_norm is None:
            raise ValueError("val_ratio=0, no val set.")
        ds = WindowDataset(val_norm, args.seq_len, args.pred_len, stride=args.pred_len,
                           in_indices=in_indices, out_indices=out_indices, target_shift=target_shift, sample_weights=val_w,
                           window_mode=window_mode, center_left=center_left)
        return ds, _build_forecast_loader(args, ds, flag), scaler

    if flag == "test":
        ds = WindowDataset(test_norm, args.seq_len, args.pred_len, stride=args.pred_len,
                           in_indices=in_indices, out_indices=out_indices, target_shift=target_shift, sample_weights=test_w,
                           window_mode=window_mode, center_left=center_left)
        return ds, _build_forecast_loader(args, ds, flag), scaler

    raise ValueError(f"Unknown flag: {flag}")


def _risk_data_provider(args, flag):
    """
    return: dataset, dataloader, scaler
    """
    df, values, _ = _load_values(args)
    total_len = len(values)
    train_raw, val_raw, test_raw = time_split(
        values,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_mode=getattr(args, "split_mode", "total"),
    )
    (tr_s, tr_e), (va_s, va_e), (te_s, te_e) = time_split_bounds(
        total_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_mode=getattr(args, "split_mode", "total"),
    )

    scaler = _build_scaler(args)
    scaler.fit(train_raw)
    train_norm = scaler.transform(train_raw)
    val_norm = scaler.transform(val_raw) if val_raw is not None else None
    test_norm = scaler.transform(test_raw)

    label_mode = str(getattr(args, "label_mode", "generated")).lower()

    if label_mode == "generated":
        label_config, train_stats = fit_risk_label_config(train_raw, args)
        val_stats = compute_risk_window_stats(
            val_raw,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            stride=args.stride,
            target_channel=label_config["target_channel"],
            sample_rate=label_config["sample_rate"],
            band_low=label_config["band_low"],
            band_high=label_config["band_high"],
        ) if val_raw is not None else None
        test_stats = compute_risk_window_stats(
            test_raw,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            stride=args.stride,
            target_channel=label_config["target_channel"],
            sample_rate=label_config["sample_rate"],
            band_low=label_config["band_low"],
            band_high=label_config["band_high"],
        )

        train_start_indices = np.asarray(train_stats["start_indices"], dtype=np.int64)
        val_start_indices = np.asarray(val_stats["start_indices"], dtype=np.int64) if val_stats is not None else None
        test_start_indices = np.asarray(test_stats["start_indices"], dtype=np.int64)
        train_labels = assign_risk_labels(train_stats, label_config)
        val_labels = assign_risk_labels(val_stats, label_config) if val_stats is not None else None
        test_labels = assign_risk_labels(test_stats, label_config)

    elif label_mode == "column":
        label_col = getattr(args, "label_col", None)
        if label_col is None:
            raise ValueError("label_mode='column' requires --label_col.")
        if label_col not in df.columns:
            raise ValueError(f"label_col '{label_col}' not found in input CSV.")
        full_labels = df[label_col].to_numpy(dtype=np.int64).reshape(-1)
        strategy = getattr(args, "window_label_strategy", "last")
        train_start_indices, train_labels = derive_window_labels_from_point_labels(
            full_labels[tr_s:tr_e], args.seq_len, args.pred_len, args.stride, strategy=strategy, num_classes=args.num_classes
        )
        if val_raw is not None:
            val_start_indices, val_labels = derive_window_labels_from_point_labels(
                full_labels[va_s:va_e], args.seq_len, args.pred_len, args.stride, strategy=strategy, num_classes=args.num_classes
            )
        else:
            val_start_indices, val_labels = None, None
        test_start_indices, test_labels = derive_window_labels_from_point_labels(
            full_labels[te_s:te_e], args.seq_len, args.pred_len, args.stride, strategy=strategy, num_classes=args.num_classes
        )
        train_stats = {}
        val_stats = {}
        test_stats = {}
        label_config = {
            "source": "column",
            "label_col": label_col,
            "num_classes": int(args.num_classes),
            "window_label_strategy": strategy,
        }

    elif label_mode == "file":
        label_path = getattr(args, "label_path", None)
        if label_path is None:
            raise ValueError("label_mode='file' requires --label_path.")
        if not os.path.isabs(label_path):
            label_path = os.path.join(args.root_path, label_path)
        loaded = load_label_file(
            label_path,
            value_col=getattr(args, "label_file_col", None),
            start_col=getattr(args, "label_start_col", "start_idx"),
        )
        granularity_arg = str(getattr(args, "label_granularity", "auto")).lower()
        granularity = loaded.get("granularity", "point") if granularity_arg == "auto" else granularity_arg
        train_stats = {}
        val_stats = {}
        test_stats = {}
        label_config = {
            "source": "file",
            "label_path": label_path,
            "label_granularity": granularity,
            "num_classes": int(args.num_classes),
        }

        if granularity == "point":
            full_labels = np.asarray(loaded["labels"], dtype=np.int64).reshape(-1)
            if full_labels.shape[0] != total_len:
                raise ValueError(
                    f"Point labels must have the same length as the raw series. Got {full_labels.shape[0]} vs {total_len}."
                )
            strategy = getattr(args, "window_label_strategy", "last")
            label_config["window_label_strategy"] = strategy
            train_start_indices, train_labels = derive_window_labels_from_point_labels(
                full_labels[tr_s:tr_e], args.seq_len, args.pred_len, args.stride, strategy=strategy, num_classes=args.num_classes
            )
            if val_raw is not None:
                val_start_indices, val_labels = derive_window_labels_from_point_labels(
                    full_labels[va_s:va_e], args.seq_len, args.pred_len, args.stride, strategy=strategy, num_classes=args.num_classes
                )
            else:
                val_start_indices, val_labels = None, None
            test_start_indices, test_labels = derive_window_labels_from_point_labels(
                full_labels[te_s:te_e], args.seq_len, args.pred_len, args.stride, strategy=strategy, num_classes=args.num_classes
            )
        elif granularity == "window":
            if "start_indices" not in loaded:
                raise ValueError("Window labels require start_indices in the label file.")
            global_starts = np.asarray(loaded["start_indices"], dtype=np.int64).reshape(-1)
            global_labels = np.asarray(loaded["labels"], dtype=np.int64).reshape(-1)
            train_start_indices, train_labels = split_window_records_by_segment(
                global_starts, global_labels, tr_s, tr_e - tr_s, args.seq_len, args.pred_len
            )
            if val_raw is not None:
                val_start_indices, val_labels = split_window_records_by_segment(
                    global_starts, global_labels, va_s, va_e - va_s, args.seq_len, args.pred_len
                )
            else:
                val_start_indices, val_labels = None, None
            test_start_indices, test_labels = split_window_records_by_segment(
                global_starts, global_labels, te_s, te_e - te_s, args.seq_len, args.pred_len
            )
        else:
            raise ValueError(f"Unknown label_granularity: {granularity}")
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    class_weights, train_counts = compute_class_weights(
        train_labels,
        num_classes=int(args.num_classes),
        power=float(getattr(args, "class_weight_power", 1.0)),
    )
    label_config["train_class_counts"] = train_counts.tolist()

    if flag == "train":
        ds = RiskWindowDataset(
            train_norm,
            train_labels,
            args.seq_len,
            args.pred_len,
            stride=args.stride,
            start_indices=train_start_indices,
            label_config=label_config,
            window_stats=train_stats,
        )
        ds.class_weights = class_weights
        ds.class_counts = train_counts
        return ds, _build_risk_loader(args, ds, flag), scaler

    if flag == "val":
        if val_raw is None or val_norm is None:
            raise ValueError("val_ratio=0, no val set.")
        ds = RiskWindowDataset(
            val_norm,
            val_labels,
            args.seq_len,
            args.pred_len,
            stride=args.stride,
            start_indices=val_start_indices,
            label_config=label_config,
            window_stats=val_stats,
        )
        ds.class_weights = class_weights
        ds.class_counts = np.bincount(val_labels, minlength=int(args.num_classes))
        return ds, _build_risk_loader(args, ds, flag), scaler

    if flag == "test":
        ds = RiskWindowDataset(
            test_norm,
            test_labels,
            args.seq_len,
            args.pred_len,
            stride=args.stride,
            start_indices=test_start_indices,
            label_config=label_config,
            window_stats=test_stats,
        )
        ds.class_weights = class_weights
        ds.class_counts = np.bincount(test_labels, minlength=int(args.num_classes))
        return ds, _build_risk_loader(args, ds, flag), scaler

    raise ValueError(f"Unknown flag: {flag}")


def data_provider(args, flag):#拼接数据路径，读取变量列，划分数据集，归一化，根据任务类型进行数据加载
    if getattr(args, "task_name", "long_term_forecast") == "risk_classification":
        return _risk_data_provider(args, flag)
    return _forecast_data_provider(args, flag)
