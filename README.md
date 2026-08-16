# R18 / HeadP_Ab Analysis Pipeline

This folder contains the publication pipeline for processing FlowJo-exported FACS traces from gated platelet populations.

## Files
- `FCS_splitter.py`
- `Normalization_estimator.py`
- `Normalization_estimator_normalized_FL2.py`
- `R18_Kinet_routine_publ.ipynb`
- `R18_Kinet_routine_publ_normalized_FL2_AUC.ipynb`

## Experimental Input Context
AUC standardization (`AUCSTD`) was calibrated using 3 replicates from 25 WT mice.

Data preparation before this pipeline:
1. Acquire flow cytometry files (`.fcs`) and gate platelets in FSC-SSC.
2. Gate events positive for headpiece antibody.
3. Export gated data from FlowJo as `.csv` files containing `Time`, `FL2` (R18), and `FL1` (HeadP_Ab-compatible column names used by scripts).

## Recommended Workflow
1. Run `FCS_splitter.py` to split each trace into phase-specific CSV files.
2. Run `Normalization_estimator.py` to estimate `AUCSTD` from calibration datasets.
3. After `AUCSTD` is calibrated, use `R18_Kinet_routine_publ.ipynb` for routine dataset analysis.

Optional alternative:
- Use `Normalization_estimator_normalized_FL2.py` and `R18_Kinet_routine_publ_normalized_FL2_AUC.ipynb` when the correction should depend on the shape of the R18 loading trace rather than its absolute FL2 intensity.

## Step 1: Split FlowJo CSV Traces
`FCS_splitter.py` performs:
1. 1-second binning of time trace.
2. Running-average smoothing.
3. Manual selection of 5 landmarks:
`point 0`, `R18 addition`, `BSA addition`, `Activator addition`, `final point`.
4. Export of split tables to `Loading_Baseline/`:
`initial_fl1.csv`, `loading_fl1.csv`, `loading_fl2.csv`, `baseline_fl1.csv`, `baseline_fl2.csv`, `activated_fl1.csv`.

Example:
```bash
python FCS_splitter.py \
  --parent-folder <flowjo_export_folder> \
  --process-subfolders
```

## Step 2: Calibrate AUCSTD
`Normalization_estimator.py`:
1. Loads split files from each subfolder.
2. Normalizes FL1 to pre-R18 initial level.
3. Computes FL2 loading AUC per trace.
4. Selects traces whose normalized FL1 baseline falls within `[low, high]`.
5. Uses median FL2 AUC of selected traces as `AUCSTD`.
6. Exports summary spreadsheets and trace plots.

Example:
```bash
python Normalization_estimator.py \
  --folder <processed_parent_folder> \
  --output-dir ../results \
  --low 55 \
  --high 65
```

Use the printed `AUC_std` value as the fixed calibration constant for routine analysis.

## Step 3: Routine Analysis with Calibrated AUCSTD
Open `R18_Kinet_routine_publ.ipynb` and:
1. Set `AUCSTD` to the calibrated value from Step 2.
2. Set `master_folder` to the directory containing processed subfolders.
3. Run all cells.

Notebook outputs:
- Per-subfolder, in `../results`: `<subfolder>_Analyzed.xlsx` (assembled normalized FL1 traces).
- Master-level, in `../results`: `result.xlsx` with `Resting` and `Activated` values.

## Optional: Normalized FL2 AUC Calibration
`Normalization_estimator_normalized_FL2.py` follows the same input and baseline filtering steps as `Normalization_estimator.py`, but computes AUC after each FL2 loading trace is divided by its own loading-window maximum:

```text
normalized_FL2_AUC = AUC(loading_FL2 / max(loading_FL2))
```

This produces a normalized-FL2 `AUC_std` that can be used as an alternative when absolute FL2 intensity is less desirable as the correction reference. In this version, the correction is based on the shape and timing of the R18 loading curve after each trace has been normalized to its own maximum. It therefore reduces dependence on absolute FL2 amplitude, which may vary with acquisition settings, staining/loading magnitude, or session-level intensity differences.

This option was added for cases where R18 loading appears excessive relative to the range observed in the WT calibration traces. In those cases, FL2 fluorescence can be substantially above the usual range and can skew absolute-FL2-AUC normalization, even when the Leo.H4/HeadP_Ab quenching dynamics and apparent availability of FRET acceptors are not otherwise changed. The normalized-FL2 pathway is therefore useful as an assay-debugging and robustness workflow for reducing sensitivity to unusually high R18 amplitude.

This option should be interpreted as an alternative or sensitivity analysis, not as a claim that normalized-FL2 AUC is universally superior to absolute-FL2 AUC. For primary analysis, use a single prespecified pathway and report clearly whether absolute-FL2 AUC or normalized-FL2 AUC was used.

The per-trace correction coefficient is:

```text
correction_coefficient = normalized_FL2_AUC / normalized_FL2_AUCSTD
```

Example:
```bash
python3 Normalization_estimator_normalized_FL2.py \
  --folder <processed_parent_folder> \
  --output-dir ../results \
  --low 55 \
  --high 65
```

Then open `R18_Kinet_routine_publ_normalized_FL2_AUC.ipynb` and:
1. Set `AUCSTD` to the normalized-FL2 `AUC_std` printed by the calibration script.
2. Set `master_folder` to the processed dataset directory.
3. Run all cells.

Notebook outputs:
- Per-subfolder, in `../results`: `<subfolder>_Analyzed_normalized_FL2_AUC.xlsx`.
- Master-level, in `../results`: `result_normalized_FL2_AUC.xlsx` with `Resting` and `Activated` values.


## Notes
- The scripts assume split outputs are under `Loading_Baseline/` in each experiment subfolder.
- Keep channel naming consistent with script expectations (`fl1`, `fl2` file suffixes).
- If your FlowJo export columns differ from expected names (for example `Comp-FL1-H`, `Comp-FL2-H`), align column names before running.

## Code Ocean Notes
- Place input data under `../data`, with one processed dataset folder per experiment group. Each experiment subfolder should contain the `Loading_Baseline/` files listed above.
- The estimator scripts write spreadsheets and figures to `../results` by default. Use `--output-dir ../results` explicitly when running in Code Ocean.
- `FCS_splitter.py` uses manual landmark selection and is intended for interactive preprocessing before headless Code Ocean execution. For Code Ocean runs, upload the already split `Loading_Baseline/` CSV files to `../data`.
- Install runtime dependencies through the Code Ocean environment editor rather than inside the scripts. Required Python packages are `numpy`, `pandas`, `matplotlib`, and `openpyxl`.
- The repository contains source code and notebooks only; raw data and generated result files should remain outside the `../code` folder.
