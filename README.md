# R18 / HeadP_Ab Analysis Pipeline

This folder contains the publication pipeline for processing FlowJo-exported FACS traces from gated platelet populations.

## Files
- `FCS_splitter.py`
- `Normalization_estimator.py`
- `Normalization_estimator_normalized_FL2.py`
- `R18_Kinet_routine_publ.ipynb`
- `R18_Kinet_routine_publ_normalized_FL2_AUC.ipynb`
- `join_two_csvs.py`

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

## Optional: Normalized FL2 AUC Calibration
`Normalization_estimator_normalized_FL2.py` follows the same input and baseline filtering steps as `Normalization_estimator.py`, but computes AUC after each FL2 loading trace is divided by its own loading-window maximum:

```text
normalized_FL2_AUC = AUC(loading_FL2 / max(loading_FL2))
```

This produces a normalized-FL2 `AUC_std` that can be used as an alternative when absolute FL2 intensity is less desirable as the correction reference. In this version, the correction is based on the shape and timing of the R18 loading curve after each trace has been scaled to its own maximum. It therefore reduces dependence on absolute FL2 amplitude, which may vary with acquisition settings, staining/loading magnitude, or session-level intensity differences.

This option should be interpreted as an alternative or sensitivity analysis, not as a claim that normalized-FL2 AUC is universally superior to absolute-FL2 AUC. For publication, report clearly which AUC standardization was used for the main analysis. If both versions are used, keep the absolute-FL2 and normalized-FL2 results separate and describe the normalized-FL2 workflow as a robustness/alternative processing route.

No additional damping, exponent, gamma, or empirical scaling factor is applied in this workflow. The per-trace correction coefficient is:

```text
correction_coefficient = normalized_FL2_AUC / normalized_FL2_AUCSTD
```

Example:
```bash
python3 FACSScripts/Paper/Normalization_estimator_normalized_FL2.py \
  --folder "/path/to/FCS Exports/WT" \
  --low 55 \
  --high 65
```

Then open `R18_Kinet_routine_publ_normalized_FL2_AUC.ipynb` and:
1. Set `AUCSTD` to the normalized-FL2 `AUC_std` printed by the calibration script.
2. Set `master_folder` to the processed dataset directory.
3. Run all cells.

Notebook outputs:
- Per-subfolder: `Analyzed_normalized_FL2_AUC.xlsx`.
- Master-level: `result_normalized_FL2_AUC.xlsx` with `Resting` and `Activated` values.

## Repository Archiving
For publication, archive the final GitHub release in a permanent repository such as Zenodo and cite the resulting DOI in the manuscript. The archived version should include this `Paper/` folder, the analysis notebooks, scripts, and enough usage notes for a reader to reproduce the reported processing steps.

## Notes
- The scripts assume split outputs are under `Loading_Baseline/` in each experiment subfolder.
- Keep channel naming consistent with script expectations (`fl1`, `fl2` file suffixes).
- If your FlowJo export columns differ from expected names (for example `Comp-FL1-H`, `Comp-FL2-H`), align column names before running.
