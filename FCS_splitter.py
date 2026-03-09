"""
Split FlowJo-exported FACS traces into experiment phases.

Expected input:
- A folder containing `.csv` files exported from `.fcs` files in FlowJo.
- Each CSV contains at least `Time`, `Comp-FL1-H`, and `Comp-FL2-H` columns.

Processing overview for each CSV:
1. Keep the first 4000 s (matching experiment duration used in this analysis).
2. Average signal in 1-second bins.
3. Apply a rolling mean (window=10 points) to smooth noise.
4. Show FL1 trace and let user click 5 landmarks (sorted left-to-right):
   point 0, R18 addition, BSA addition, Activator addition, final point.
5. Slice intervals between landmarks and aggregate them across files.

Output:
- A `Loading_Baseline` subfolder with phase-specific CSVs:
  `initial_fl1.csv`, `loading_fl1.csv`, `loading_fl2.csv`,
  `baseline_fl1.csv`, `baseline_fl2.csv`, `activated_fl1.csv`.

CLI example:
`python FCS_splitter.py --parent-folder "/path/to/exports" --process-subfolders`
"""

import argparse
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt


def pick_times(ax):
    """Collect five manual time landmarks from the displayed trace.

    The user clicks any 5 positions on the x-axis. Click order is not important:
    coordinates are sorted so returned times are always chronological.

    Returns:
        tuple[float, float, float, float, float]:
            (t1, t2, t3, t4, t5) corresponding to:
            start, R18 addition, BSA addition, activator addition, end.
    """
    pts = plt.ginput(5, timeout=-1) 
    xs = [p[0] for p in pts]
    xs_sorted = sorted(xs)

    # Draw vertical lines where you clicked
    for x in xs_sorted:
        ax.axvline(x=x, color='red', linestyle='--', linewidth=1)

    # Force redraw so you see the lines immediately
    ax.figure.canvas.draw()

    return xs_sorted[0], xs_sorted[1], xs_sorted[2], xs_sorted[3], xs_sorted[4]

def slice_interval(df, t_col, y_col, t_start, t_end):
    """
    Return values from one channel in a half-open time interval.

    Interval convention is [t_start, t_end), i.e. inclusive start, exclusive end.
    Returned index is reset so segments from different files can be aligned by row.
    """
    m = (df[t_col] >= t_start) & (df[t_col] < t_end)
    return df.loc[m, y_col].reset_index(drop=True)

def identify_files(path_str, max_time=4000.0, dt=1.0, roll_window=10):
    """Process all CSV files in one experiment folder and export split traces.

    Args:
        path_str (str | Path): Folder containing FlowJo-exported CSV files.
        max_time (float): Keep only rows where `Time < max_time`.
        dt (float): Time bin size in seconds.
        roll_window (int): Rolling average window size in binned points.
    """
    folder = Path(path_str)
    csv_files = list(folder.glob('*.csv'))

    # Dictionaries accumulate one column per input file.
    # Key = file name, value = sliced 1D signal for that phase.
    loading_fl1_cols = {}
    loading_fl2_cols = {}
    
    initial_fl1_cols = {}
    
    baseline_fl1_cols = {}
    baseline_fl2_cols = {}

    activated_fl1_cols = {}

    for file in csv_files:

        # Read one FlowJo export and keep experiment-relevant time range.
        raw_data = pd.read_csv(file)
        raw_data = raw_data[raw_data['Time'] < max_time]

        # Step 1: average events in 1-second bins to create a regular time trace.
        time_averaged = raw_data.assign(
            time_bin=(raw_data['Time'] // dt) * dt).groupby(
                'time_bin', as_index=False).mean()
        
        # Step 2: smooth binned trace with a running average (10 seconds here).
        rolled = time_averaged.copy()
        rolled = rolled.sort_values('time_bin').set_index('time_bin')
        rolled = rolled.rolling(window=roll_window).mean().reset_index()

        # Interactive landmark picking is performed on FL1 view.
        fig, ax = plt.subplots()

        ax.scatter(rolled['time_bin'], rolled['Comp-FL1-H'], label=f"{file.parent.name}_{file.name}")
        ax.legend()

        # t1..t5 correspond to:
        # point 0, R18 addition, BSA addition, activator addition, final point.
        t1, t2, t3, t4, t5 = pick_times(ax)

        plt.close(fig)

        # Phase splitting:
        # initial:   point0 -> R18
        # loading:   R18 -> BSA (both FL1 and FL2)
        # baseline:  BSA -> activator (both FL1 and FL2)
        # activated: activator -> final (FL1)
        initial_fl1_cols[file.name] = slice_interval(rolled, 'time_bin',
                                                     'Comp-FL1-H', t1, t2)
        loading_fl1_cols[file.name]  = slice_interval(rolled, 'time_bin', 
                                                      'Comp-FL1-H', t2, t3)
        loading_fl2_cols[file.name]  = slice_interval(rolled, 'time_bin', 
                                                      'Comp-FL2-H', t2, t3)
        baseline_fl1_cols[file.name] = slice_interval(rolled, 'time_bin',
                                                      'Comp-FL1-H', t3, t4)
        baseline_fl2_cols[file.name] = slice_interval(rolled, 'time_bin',
                                                      'Comp-FL2-H', t3, t4)
        
        activated_fl1_cols[file.name] = slice_interval(rolled, 'time_bin',
                                                      'Comp-FL1-H', t4, t5)

    # Build output tables:
    # each column is one source file; rows are time-indexed samples inside a phase.
    # Different segment lengths are padded with NaN by pandas.
    
    loading_fl1  = pd.DataFrame(loading_fl1_cols)
    loading_fl2  = pd.DataFrame(loading_fl2_cols)

    initial_fl1 = pd.DataFrame(initial_fl1_cols)

    baseline_fl1 = pd.DataFrame(baseline_fl1_cols)
    baseline_fl2 = pd.DataFrame(baseline_fl2_cols)
    
    # Note: variable name is reused here (dict -> DataFrame), preserving script behavior.
    activated_fl1_cols = pd.DataFrame(activated_fl1_cols)

    # Export all split phases to a dedicated subfolder.
    output_folder = folder / 'Loading_Baseline'
    output_folder.mkdir(parents=True, exist_ok=True)

    loading_fl1.to_csv(output_folder / "loading_fl1.csv", index=False)
    loading_fl2.to_csv(output_folder / "loading_fl2.csv", index=False)

    initial_fl1.to_csv(output_folder / "initial_fl1.csv", index=False)

    baseline_fl1.to_csv(output_folder / "baseline_fl1.csv", index=False)
    baseline_fl2.to_csv(output_folder / "baseline_fl2.csv", index=False)

    activated_fl1_cols.to_csv(output_folder / "activated_fl1.csv", index=False)


def _build_parser():
    """Create CLI parser for batch splitting of FlowJo CSV exports."""
    parser = argparse.ArgumentParser(
        description="Split FlowJo-exported CSV traces into loading/baseline/activation phases."
    )
    parser.add_argument(
        "--parent-folder",
        default="/Users/alexeymartyanov/Desktop/FCS Exports/JAK2",
        help="Folder containing experiment subfolders (or CSVs if --no-process-subfolders).",
    )
    parser.add_argument(
        "--process-subfolders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process each immediate subfolder independently (default: true).",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=4000.0,
        help="Keep input rows where Time < max_time (default: 4000).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time bin width in seconds (default: 1).",
    )
    parser.add_argument(
        "--roll-window",
        type=int,
        default=10,
        help="Running-average window size in binned points (default: 10).",
    )
    return parser


def main():
    """CLI entry point."""
    args = _build_parser().parse_args()
    parent_path = Path(args.parent_folder)

    if args.process_subfolders:
        target_folders = sorted(p for p in parent_path.iterdir() if p.is_dir())
    else:
        target_folders = [parent_path]

    if not target_folders:
        raise ValueError(f"No target folders found under: {parent_path}")

    for folder in target_folders:
        identify_files(folder, max_time=args.max_time, dt=args.dt, roll_window=args.roll_window)
        print(f"{folder.name} Done")


if __name__ == "__main__":
    main()
