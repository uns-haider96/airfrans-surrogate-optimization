# A Data-Driven Surrogate for Airfoil Shape Optimization

Neural-network surrogates trained on the [AirfRANS](https://airfrans.readthedocs.io) benchmark, used to drive airfoil shape optimization, with an explicit study of when the resulting optimum can be trusted.

**Author:** Muhammad Uns Haider Shah

---

## Problem statement

AirfRANS is published as a flow-field prediction benchmark. Field error, however, is not what decides whether a surrogate is useful for design. What matters inside an optimization loop is whether the surrogate **orders candidate designs correctly** and whether the optimum it proposes is real.

This project therefore asks three questions:

1. What must a surrogate be given in order to rank airfoil designs, rather than merely fit flow fields?
2. Which aerodynamic quantities can a surrogate of this kind rank reliably, and which can it not?
3. When the surrogate proposes an optimum, how far can that optimum be trusted without new high-fidelity simulation?

## Dataset

The AirfRANS *scarce* task: 200 two-dimensional, steady, incompressible RANS simulations over NACA 4- and 5-digit airfoils, with 200 held-out test cases shared with the *full* task.

* Reynolds number 2–6 million, angle of attack −5° to 15°, k-ω SST turbulence model, solved in OpenFOAM.
* About 180,000 mesh nodes per case (not the ~32,000 sometimes quoted; the smaller figure is the number of nodes the original baselines subsampled during training).
* Inputs per node: position, inlet velocity vector, signed distance to the airfoil, surface normal. Targets: velocity, kinematic pressure, turbulent viscosity. Angle of attack is encoded in the inlet velocity direction; the geometry is never rotated.
* Splits: 200 training cases divided 160 / 40 into training and validation. The test set is untouched until final evaluation.

## Method

| Phase | Notebook | Content |
|---|---|---|
| 0 | `00_data_pipeline.ipynb` | Data pipeline, flow-field and Cp visualisation, pressure-force integration validated against the library's reference implementation |
| 1 | `01_baseline_mlp.ipynb` | Pointwise MLP predicting the four flow fields at each mesh node |
| 1b | `01b_shape_aware_mlp.ipynb` | Identical model plus a 13-number global shape descriptor per node |
| 2 | `02_aerodynamic_evaluation.ipynb` | Evaluation in aerodynamic terms: Cp, lift and drag including skin friction, rank correlation, pairwise ranking, boundary layers, comparison with published baselines |
| 2b | `02b_direct_force_models.ipynb` | Direct force surrogates (neural ensemble and Gaussian process) mapping shape and operating condition to lift and drag |
| 3 | `03_shape_optimization.ipynb` | Shape optimization: Bayesian optimization, random-search control, and gradient-based search using automatic differentiation through the network and the parameterisation |
| 4 | `04_trust_and_extrapolation.ipynb` | Trust study: position relative to the data, stability under resampling, controlled extrapolation, documented failure regime |

All notebooks run end to end on Google Colab. Phases 1 and 1b use a GPU; the rest run on CPU. The dataset is cached in Google Drive after first download, and long computations checkpoint to Drive so an interrupted session resumes.

## Results

### Shape information is what allows a surrogate to rank designs

Both models are trained identically on the same 160 cases; the only difference is that the second receives a description of the whole airfoil at each node.

| test set, 200 cases | pointwise MLP | shape-aware MLP |
|---|---|---|
| volume field MSE (mean, normalised) | 0.252 | **0.056** |
| surface pressure MSE (normalised) | 0.967 | **0.063** |
| pressure lift, RMSE | 0.115 | **0.040** |
| lift rank correlation | 0.984 | **0.998** |
| lift ordering, pairs at comparable conditions | 0.902 | **0.971** |

A pointwise model sees each node in isolation, but surface pressure depends on the entire airfoil. Without global shape information the model memorises the training airfoils: its validation error rises after epoch ~350 while training error keeps falling.

### Field-based surrogates predict lift well, and cannot rank drag

Predicted fields are integrated to forces using the benchmark's own post-processing.

| shape-aware MLP, 200 test cases | value |
|---|---|
| lift relative error (median) | 2.6 % |
| lift rank correlation | 0.998 |
| drag relative error (median) | 1715 % |
| drag rank correlation | 0.075 |
| drag ordering, comparable pairs | 0.63 (chance = 0.5) |

The cause is physical, not a coding error. Viscous drag is 68 % of total drag and is computed from the velocity gradient across a first cell about 2 µm thick; a smooth network cannot resolve that layer. Enforcing the exact no-slip condition at the wall changed nothing, which locates the error in the nodes just above the wall. Pressure drag fails for a different reason: it is a small residual of large, nearly cancelling pressure forces, and still carries a 60 % median error even where surface pressure is accurate. Every model in the published benchmark fails on drag in the same way (rank correlations of −0.12 to −0.14).

Against the published baselines (single training run here, five-run means in the paper):

| | surface p MSE | CD rel. err | CL rel. err | ρ_D | ρ_L |
|---|---|---|---|---|---|
| AirfRANS MLP (full task, 800 cases) | 0.113 | 4.29 | 0.767 | −0.117 | 0.913 |
| AirfRANS GraphSAGE (scarce task) | 0.195 | 3.50 | 0.385 | −0.139 | 0.981 |
| this work, pointwise MLP (scarce) | 0.967 | 23.3 | 0.296 | −0.019 | 0.985 |
| this work, shape-aware MLP (scarce) | **0.063** | 18.2 | **0.096** | 0.075 | **0.998** |

The shape-aware model is given the NACA parameters, which the benchmark models are not: they learn from the mesh alone and apply to arbitrary geometries. The comparison is therefore favourable to this work by construction and is reported for context, not as a like-for-like ranking.

### Direct force surrogates rank both lift and drag

Regressing lift and log-drag directly from shape, incidence and Reynolds number bypasses the near-wall problem entirely.

| 200 test cases | field model | NN ensemble | Gaussian process |
|---|---|---|---|
| drag relative error (median) | 1715 % | 0.8 % | **0.3 %** |
| drag rank correlation | 0.075 | 0.998 | **0.999** |
| L/D rank correlation | 0.883 | 0.997 | **0.999** |
| drag ordering, comparable & distinguishable pairs | 0.66 | 0.995 | **1.000** |
| within ±2σ of predicted uncertainty | – | 55 % | **91 %** |

The Gaussian process is both more accurate and better calibrated; the ensemble spread underestimates its own error. Learned length scales identify incidence and thickness as the dominant drivers of drag, then camber near 20 % and 80 % chord, with Reynolds number only weakly influential over this range.

### Both search strategies reach the surrogate's optimum; random search does not

Maximising L/D at Re = 4 × 10⁶ and α = 4°, with thickness constrained to ≥ 12 %:

| method | surrogate evaluations | best L/D | vs. 200,000-point reference |
|---|---|---|---|
| Bayesian optimization | 50 | 91.73 | −0.03 % |
| random search (same budget) | 50 | 88.89 | +3.07 % |
| gradient-based, 20 starts | 176 (8.8 per start) | 92.01 | −0.33 % |

Lift-constrained drag minimisation behaved the same way, with the constraint active at exactly CL = 0.800. A single gradient-based search is about six times cheaper than the Bayesian run, but one of the 20 starts converged to L/D = 76 instead of 92, so restarts are necessary. Gradients come from automatic differentiation through the network and through the camber-line formula; no adjoint solver and no finite differences are involved. The sensitivities agree with central finite differences to seven decimal places.

### The optimum's performance is robust; its location is not

Retraining the surrogate on eight random 80 % subsets and repeating the optimization:

* L/D at the optimum: 93.8 ± 1.6; lift-constrained drag: 0.0099 ± 0.0001.
* The optimal design itself moves substantially: camber from 4.8 to 7.0 and camber position from 4.2 to 6.9, roughly 14–20 % of the design range.
* Thickness sat on the 12 % lower bound in every single run.
* Two surrogates trained on the same data disagree by about 4 % on the value of a given optimum, while the optimizers compete over differences of 0.3 %.

The surrogate identifies a **family** of near-equivalent designs rather than a unique optimum, and the binding constraint is structural, not aerodynamic.

### Uncertainty tracks distance from the data, not changes of flow regime

Narrowing the training band deliberately and testing outside it:

| band narrowed in | drag error inside | near outside | far outside | error / predicted σ, far |
|---|---|---|---|---|
| Reynolds number | 0.8 % | 1.3 % | 2.4 % | 0.89 |
| angle of attack | 0.3 % | 0.5 % | 4.0 % | 1.60 |

Predicted uncertainty grows along with the error, so it is a usable stopping signal. The exception is decisive: the worst-predicted case in the dataset is a 5.2 %-thick airfoil at −4.4° incidence whose flow separates along the lower surface. Its drag is 4.1× the training median and 93 % pressure drag (typically 33 %), the surrogate under-predicts it by 80 %, and the error is **139× the predicted uncertainty**. Only 7 of 200 training cases lie in that region of the design space.

A surrogate's confidence bounds its interpolation error. It cannot see a change of flow regime, because the inputs look ordinary.

## Conclusion

A cheap data-driven surrogate narrows a design space quickly: both optimization problems were solved in seconds to within a fraction of a percent of the surrogate's own optimum. It cannot certify the result. Surrogate-to-surrogate disagreement and the movement of the optimum under resampling both exceed the gains the optimizer is chasing, and confidence collapses where the flow physics changes rather than where the inputs become unusual.

The defensible use is mixed-fidelity: surrogate-based exploration to identify a family of candidate designs, followed by high-fidelity verification. Where gradient-based refinement is wanted at high fidelity, an adjoint formulation is the appropriate tool, since its cost is essentially independent of the number of design variables, which is precisely the regime where surrogate-based optimization stops being viable.

## Limitations

* **Flow regime.** AirfRANS is incompressible and subsonic at Reynolds numbers of a few million. Much benchmark shape-optimization work is transonic, where shock waves and wave drag dominate. The methodology transfers; the flow regime does not.
* **Dimensionality.** Three design variables. Surrogate-based optimization is viable here and degrades as the design space grows, because the sample count required rises sharply with dimension. Adjoint methods exist for the opposite regime.
* **Parametric family.** The direct force surrogates apply only to NACA 4- and 5-digit airfoils at the trained conditions. They are not field predictors and do not generalise to arbitrary geometries, unlike the benchmark's published models.
* **No high-fidelity verification.** No new CFD was run. The optima are supported by resampling stability, surrogate agreement, uncertainty estimates and the nearest simulations in the dataset, none of which substitutes for a verification run.
* **Single training runs.** Published baselines report means over five runs. Numbers here come from single runs and carry corresponding run-to-run uncertainty.
* **Leakage in the benchmark splits.** The `reynolds` and `aoa` splits are drawn from the same 1000 simulations as the `full` task; 88 of 496 cases belonged to this model's training set and were removed before evaluation.

## Reproducing

```
pip install airfrans "pyvista==0.48.0" "vtk<9.7"
```

Run the notebooks in order (`00` → `04`) on Google Colab. Phase 0 downloads the dataset (~9 GB) once and caches it in Google Drive under `MyDrive/airfrans/`; later notebooks restore it from there and read the checkpoints and result files written by earlier phases. Each notebook states which earlier outputs it needs.

## References

* Bonnet, Mazari, Cinnella, Gallinari. *AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions.* NeurIPS 2022 Datasets and Benchmarks.
* Dataset library: <https://github.com/Extrality/airfrans_lib>
