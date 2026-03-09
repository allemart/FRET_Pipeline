# R18 / HeadP_Ab Analysis Pipeline

This folder contains the publication pipeline for processing FlowJo-exported FACS traces from gated platelet populations.

## Files
- `FCS_splitter.py`
- `Normalization_estimator.py`
- `R18_Kinet_routine_publ.ipynb`

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
python FACSScripts/Paper/FCS_splitter.py \
  --parent-folder "/path/to/FCS Exports/JAK2" \
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
python FACSScripts/Paper/Normalization_estimator.py \
  --folder "/path/to/FCS Exports/RASA3" \
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
- Per-subfolder: `Analyzed.xlsx` (assembled normalized FL1 traces).
- Master-level: `result.xlsx` with `Resting` and `Activated` values.

## Notes
- The scripts assume split outputs are under `Loading_Baseline/` in each experiment subfolder.
- Keep channel naming consistent with script expectations (`fl1`, `fl2` file suffixes).
- If your FlowJo export columns differ from expected names (for example `Comp-FL1-H`, `Comp-FL2-H`), align column names before running.
