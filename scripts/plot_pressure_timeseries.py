import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def _parse_columns(value: str | None):
    if value is None:
        return None
    cols = [item.strip() for item in value.split(",") if item.strip()]
    return cols if cols else None


def _default_pressure_columns(df: pd.DataFrame):
    return [col for col in df.columns if col.lower().startswith("p")]


def main():
    parser = argparse.ArgumentParser(description="绘制整段压力时序图")
    parser.add_argument("--csv", type=str, required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出图片路径")
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="要绘制的压力列，逗号分隔；默认绘制所有压力列",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20000,
        help="最多绘制多少个时间点；超过时自动均匀抽样",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="图标题；默认使用文件名",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="16,8",
        help="画布大小，格式为 宽,高",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "time" not in df.columns:
        raise ValueError("CSV 中缺少 time 列，无法绘制时间轴。")

    pressure_cols = _parse_columns(args.columns)
    if pressure_cols is None:
        pressure_cols = _default_pressure_columns(df)

    if not pressure_cols:
        raise ValueError("未找到可绘制的压力列。")

    missing = [col for col in pressure_cols if col not in df.columns]
    if missing:
        raise ValueError(f"以下列在 CSV 中不存在: {missing}")

    plot_df = df[["time"] + pressure_cols].copy()
    total_points = len(plot_df)
    if args.max_points > 0 and total_points > args.max_points:
        step = max(1, total_points // args.max_points)
        plot_df = plot_df.iloc[::step].reset_index(drop=True)
        print(f"原始点数: {total_points}，抽样后点数: {len(plot_df)}，抽样步长: {step}")
    else:
        print(f"点数: {total_points}，未做抽样。")

    width, height = [float(item.strip()) for item in args.figsize.split(",")]
    plt.figure(figsize=(width, height))

    for col in pressure_cols:
        plt.plot(plot_df["time"], plot_df[col], linewidth=1.0, label=col)

    plt.xlabel("时间 / 秒")
    plt.ylabel("压力")
    plt.title(args.title or os.path.basename(args.csv))
    plt.grid(True, alpha=0.3)
    if len(pressure_cols) <= 12:
        plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=200)
    plt.close()
    print(f"已保存图片: {args.output}")


if __name__ == "__main__":
    main()
