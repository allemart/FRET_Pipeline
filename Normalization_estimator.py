"""
Normalize FL1 (HeadP_Ab) traces using FL2 (R18) loading AUC as reference.

Input expectation:
- `folder_str` points to a parent directory with subfolders.
- Each subfolder contains `Loading_Baseline/` CSVs produced by `FCS_splitter.py`:
  `loading_fl1.csv`, `loading_fl2.csv`, `initial_fl1.csv`, `baseline_fl1.csv`.

Pipeline summary:
1. Import loading FL1/FL2 traces from all subfolders.
2. Normalize FL1 to each trace's own pre-R18 initial level (`initial_fl1` mean = 100%).
3. Build an FL1 trace that includes loading + baseline phases.
4. Estimate FL2 loading AUC for each trace (first `AUC_WINDOW` points).
5. Select traces whose normalized FL1 baseline falls in `[low, high]`.
6. Use median FL2 AUC of selected traces as `AUC_std` (gold-standard AUC).
7. Scale FL1 traces by `(trace_FL2_AUC / AUC_std)`.
8. Export summary tables and plot raw FL2, pre-AUC FL1, and post-AUC FL1.

CLI example:
`python Normalization_estimator.py --folder "/path/to/parent" --low 55 --high 65`
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AUC_WINDOW = 300
PLOT_WINDOW = 400


def _read_phase_file(folder: Path, phase: str, channel: str) -> pd.DataFrame:
    """Read one phase/channel CSV from a processed experiment subfolder."""
    return pd.read_csv(folder / f"Loading_Baseline/{phase}_{channel}.csv")


def _normalize_fl1_to_initial(fl1_df: pd.DataFrame, initial_fl1_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize FL1 columns to their own initial (pre-R18) mean, expressed in %."""
    fl1_initial_mean = initial_fl1_df.mean(axis=0)
    return fl1_df.divide(fl1_initial_mean, axis=1) * 100


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence of duplicate column names."""
    return df.loc[:, ~df.columns.duplicated()].copy()


def _auc_first_window(series: pd.Series, window: int = AUC_WINDOW) -> float:
    """Compute trapezoidal AUC for the first `window` points."""
    return float(np.trapz(series.iloc[:window]))


def acquire_data(folders, phase, channels):
    """Load and prepare FL1/FL2 traces from all experiment subfolders.

    Args:
        folders (list[Path]): Experiment subfolders.
        phase (str): Phase prefix for loading tables (typically `loading`).
        channels (list[str]): `['fl1', 'fl2']`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
            trace_fl1: loading+baseline FL1 (normalized to initial)
            loading_fl2: loading FL2 (raw)
            baseline_fl1_mean: mean normalized baseline FL1 per trace
            loading_fl1: loading FL1 (normalized to initial)
    """
    fl1_channel, fl2_channel = channels

    loading_fl2_parts = []
    loading_fl1_parts = []
    trace_fl1_parts = []
    baseline_mean_parts = []

    for folder in folders:
        loading_fl2 = _read_phase_file(folder, phase, fl2_channel)
        loading_fl1_raw = _read_phase_file(folder, phase, fl1_channel)
        initial_fl1 = _read_phase_file(folder, "initial", fl1_channel)
        baseline_fl1_raw = _read_phase_file(folder, "baseline", fl1_channel)

        # Per-file FL1 normalization to its own pre-R18 mean.
        loading_fl1_norm = _normalize_fl1_to_initial(loading_fl1_raw, initial_fl1)
        baseline_fl1_norm = _normalize_fl1_to_initial(baseline_fl1_raw, initial_fl1)

        # Full FL1 trace used for AUC-based correction (loading + baseline).
        trace_fl1 = pd.concat([loading_fl1_norm, baseline_fl1_norm], axis=0, ignore_index=True)

        loading_fl2_parts.append(loading_fl2)
        loading_fl1_parts.append(loading_fl1_norm)
        trace_fl1_parts.append(trace_fl1)
        baseline_mean_parts.append(baseline_fl1_norm.mean(axis=0))

    loading_fl2_merged = pd.concat(loading_fl2_parts, axis=1)
    loading_fl1_merged = pd.concat(loading_fl1_parts, axis=1)
    trace_fl1_merged = pd.concat(trace_fl1_parts, axis=1)
    baseline_fl1_mean = pd.concat(baseline_mean_parts)

    return trace_fl1_merged, loading_fl2_merged, baseline_fl1_mean, loading_fl1_merged


def auc_gold(loading_fl2: pd.DataFrame, good_mask: pd.Series, window: int = AUC_WINDOW) -> float:
    """Estimate gold-standard FL2 AUC as median AUC among baseline-qualified traces."""
    auc_golds = []
    for col in loading_fl2.columns:
        if bool(good_mask.get(col, False)):
            auc_golds.append(_auc_first_window(loading_fl2[col], window=window))

    if not auc_golds:
        raise ValueError("No traces passed baseline filter; cannot estimate AUC_std.")

    auc_std = float(np.median(auc_golds))
    print(f"AUC_std (median of good traces): {auc_std:.4f}")
    return auc_std


def normalize_fl1(fl1_original: pd.DataFrame, fl2_original: pd.DataFrame,
                  auc_std: float, window: int = AUC_WINDOW) -> pd.DataFrame:
    """Apply AUC-based normalization factor to each FL1 trace.

    For each trace:
    - compute FL2 AUC over the loading window,
    - compute `norm_coeff = trace_auc / auc_std`,
    - multiply FL1 by `norm_coeff`.
    """
    fl1_norm = _drop_duplicate_columns(fl1_original)
    fl2_clean = _drop_duplicate_columns(fl2_original)

    common_cols = [col for col in fl1_norm.columns if col in fl2_clean.columns]
    for col in common_cols:
        auc_tmp = _auc_first_window(fl2_clean[col], window=window)
        norm_coeff = auc_tmp / auc_std
        fl1_norm[col] = fl1_norm[col].multiply(norm_coeff)

    return fl1_norm


def plot_all(fl1_to_plot: pd.DataFrame, fl2_to_plot: pd.DataFrame,
             normalized: pd.DataFrame, export_xlsx: str,
             auc_window: int = AUC_WINDOW, plot_window: int = PLOT_WINDOW,
             show_plot: bool = True):
    """Plot pre/post normalization traces and export summary spreadsheets.

    Exports:
    - `<export_xlsx>`: mean normalized baseline (post-loading) per trace.
    - `<export_xlsx.replace('.xlsx', '_AUCplot.xlsx')>`: AUC and FL1 amplitudes.
    - `<export_xlsx.replace('.xlsx', '_traces.png')>`: panel figure with all traces.
    """
    fl1_clean = _drop_duplicate_columns(fl1_to_plot)
    fl2_clean = _drop_duplicate_columns(fl2_to_plot)
    normalized_clean = _drop_duplicate_columns(normalized)

    auc_by_trace = {}
    fl1_original_delta = {}
    fl1_normalized_delta = {}
    baseline_export = {}

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    for col in fl2_clean.columns:
        # Metrics over loading interval.
        auc_by_trace[col] = _auc_first_window(fl2_clean[col], window=auc_window)

        fl1_loading = fl1_clean[col].iloc[:auc_window]
        fl1_original_delta[col] = float(fl1_loading.max() - fl1_loading.min())

        fl1_norm_loading = normalized_clean[col].iloc[:auc_window]
        if fl1_norm_loading.max() < 90:
            fl1_normalized_delta[col] = float(90 - fl1_norm_loading.min())
        else:
            fl1_normalized_delta[col] = float(fl1_norm_loading.max() - fl1_norm_loading.min())

        baseline_export[col] = normalized_clean[col].iloc[auc_window:]

        color = "black"
        alpha = 0.5
        axes[0, 0].plot(fl2_clean[col].iloc[:plot_window].values, color=color, alpha=alpha)
        axes[0, 0].set_title("R18 Fluor, FL2")

        axes[0, 1].plot(fl1_clean[col].iloc[:plot_window].values, color=color, alpha=alpha)
        axes[0, 1].set_title("HeadP_Ab Fluor, FL1 (pre-AUC normalization)")

        axes[1, 1].plot(normalized_clean[col].iloc[:plot_window].values, color=color, alpha=alpha)
        axes[1, 1].set_title("HeadP_Ab Fluor, FL1 (post-AUC normalization)")

    axes[1, 0].axis("off")

    auc_summary = pd.concat(
        [
            pd.Series(auc_by_trace, name="AUC"),
            pd.Series(fl1_original_delta, name="Original FL1"),
            pd.Series(fl1_normalized_delta, name="Normalized FL1"),
        ],
        axis=1,
    )

    auc_summary.to_excel(export_xlsx.replace(".xlsx", "_AUCplot.xlsx"))

    baseline_export_df = pd.DataFrame(baseline_export).mean(axis=0)
    baseline_export_df.to_excel(export_xlsx)

    fig.tight_layout()
    fig.savefig(export_xlsx.replace(".xlsx", "_traces.png"), dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _build_parser():
    """Create CLI parser for FL1 AUC-based normalization workflow."""
    parser = argparse.ArgumentParser(
        description="Normalize HeadP_Ab FL1 traces using FL2 loading AUC."
    )
    parser.add_argument(
        "--folder",
        default="/Users/alexeymartyanov/Desktop/FCS Exports/RASA3",
        help="Parent folder containing experiment subfolders with Loading_Baseline exports.",
    )
    parser.add_argument(
        "--phase",
        default="loading",
        help="Phase prefix used for loading CSV files (default: loading).",
    )
    parser.add_argument(
        "--fl1-channel",
        default="fl1",
        help="FL1 channel suffix in exported filenames (default: fl1).",
    )
    parser.add_argument(
        "--fl2-channel",
        default="fl2",
        help="FL2 channel suffix in exported filenames (default: fl2).",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=55.0,
        help="Lower bound for normalized FL1 baseline filter (default: 55).",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=65.0,
        help="Upper bound for normalized FL1 baseline filter (default: 65).",
    )
    parser.add_argument(
        "--auc-window",
        type=int,
        default=AUC_WINDOW,
        help=f"Number of points for AUC calculation (default: {AUC_WINDOW}).",
    )
    parser.add_argument(
        "--plot-window",
        type=int,
        default=PLOT_WINDOW,
        help=f"Number of points shown in traces figure (default: {PLOT_WINDOW}).",
    )
    parser.add_argument(
        "--export-xlsx",
        default=None,
        help="Optional output xlsx path. Defaults to '<folder>.xlsx'.",
    )
    parser.add_argument(
        "--show-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display interactive matplotlib window (default: true).",
    )
    return parser


def main():
    """CLI entry point."""
    args = _build_parser().parse_args()

    folder_path = Path(args.folder)
    sub_folders = sorted(p for p in folder_path.iterdir() if p.is_dir())
    if not sub_folders:
        raise ValueError(f"No subfolders found under: {folder_path}")

    trace_fl1, loading_fl2, base_fl1, loading_fl1 = acquire_data(
        sub_folders,
        args.phase,
        [args.fl1_channel, args.fl2_channel],
    )

    # Baseline filter (normalized FL1, post-loading segment).
    good_mask = base_fl1.between(args.low, args.high)

    auc_std = auc_gold(loading_fl2, good_mask, window=args.auc_window)
    wt_norm = normalize_fl1(trace_fl1, loading_fl2, auc_std, window=args.auc_window)

    export_path = args.export_xlsx or f"{args.folder}.xlsx"
    plot_all(
        trace_fl1,
        loading_fl2,
        wt_norm,
        export_path,
        auc_window=args.auc_window,
        plot_window=args.plot_window,
        show_plot=args.show_plot,
    )


if __name__ == "__main__":
    main()
