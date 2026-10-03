# Re-running after the October 2026 review

The review fixed bugs and corrected claims, but it could not execute the notebooks: the dataset server
was unreachable from the review environment, and the checkpoints and cached results live in Google
Drive. The stored notebook outputs, `results/*.json`, `figures/` and the numbers in the README and the
write-up therefore still come from the original run. This file lists what to run, in which order, and
what to update afterwards.

## What changed

| notebook | change | affects results? |
|---|---|---|
| all | Helpers moved to `src/airfrans_surrogate` (tested in `tests/`); notebooks clone the repo to import them, print package versions, and run outside Colab with `USE_DRIVE = False` | no |
| 00 | Removed the unused normalisation-statistics section (it was computed on all 200 cases, validation included) | no |
| 02 | Force evaluation in fp32 instead of fp16 autocast; the earlier CSV is kept as `test_forces_fp16.csv` | **yes**, all force metrics |
| 02b | Ensemble hyperparameters selected by validation MAE instead of MSE (one separated validation case dominated the MSE); coverage row in the summary corrected (CL 0.79, not 0.55) | **yes**, possibly the selected ensemble |
| 03 | Differentiable descriptor now matches `airfrans` at P = 0 (it returned camber m(1 − x²) there instead of zero, and NaN gradients); multi-start accepts only converged, feasible starts; evaluation counts include constraint calls; bounds and distances use the 160 training cases only | **yes** |
| 04 | Same optimizer fix; position check adds full input-space distance, bound detection and the GP-uncertainty percentile; failure case searched among unseen cases only; summary corrected (93.3 ± 1.9 → 91.9 ± 4.2) | **yes** |
| 05 | New: precision comparison, seed variation, refit on 200 cases, ~600 further unseen cases, measured separation, cross-evaluation of optima | new results |

## Order

1. **00, 01, 01b**: no need to re-run. To estimate run-to-run variation of the field models, re-run
   01 and 01b with another `CFG['seed']` and a different `CKPT_DIR`. Note that the seed also draws the
   validation split, and 01b checks that its split matches Phase 1's.
2. **02** (GPU): recomputes all test forces in fp32. On first run it moves the old CSV to
   `phase2/test_forces_fp16.csv`. About 15 minutes plus data loading.
3. **02b**: retrains the direct surrogates with MAE-based selection. Delete nothing; the training-force
   CSV cache is reused.
4. **03**, then **04**: re-optimise and redo the trust study with the fixed descriptor and optimizer.
5. **05**: the new robustness checks, about 30–40 minutes on CPU.

## Afterwards

* Copy the refreshed `phase*_summary.json` files from Drive into `results/` (add `phase5_summary.json`)
  and the regenerated figures into `figures/`.
* Update the numbers in `README.md` and `writeup/writeup.md`, in particular:
  * Phase 2 tables (field-model drag error, rank correlations), and the sentence on fp16 once
    notebook 05 section 4 shows whether precision mattered;
  * the Phase 2b table, with the seed spread from notebook 05 section 5 as error bars;
  * the Phase 3 optimization table (evaluation counts change for problem 2; designs may change);
  * all Phase 4 figures, including the "on a bound" and input-space distance results;
  * the failure-case paragraph: keep, strengthen or soften it according to notebook 05 sections 6–7
    (is the case measurably separated; are other confidently wrong cases thin sections at negative
    incidence; is the "139×" case isolated);
  * the optimizer-exploitation sentence, according to notebook 05 section 8 (is the ensemble–GP
    disagreement at the gradient optimum larger than the typical disagreement over the design box?).
* Replace the "Status" note at the top of the README.
* Add the AirfRANS paper's table numbers to the baseline values in notebook 02 (section 8) and the README;
  they could not be looked up from the review environment.
* Optional: commit the small artefacts from Drive (`phase2b/force_ensemble.pt`, `phase2b/force_gp.joblib`,
  `phase2b/train_forces.csv`, `phase2/test_forces.csv`, `phase3/optima.npz`, together well under 2 MB), so
  that the optimization and trust phases can be inspected without the 9 GB dataset. The joblib file only
  loads with the scikit-learn version that wrote it.
