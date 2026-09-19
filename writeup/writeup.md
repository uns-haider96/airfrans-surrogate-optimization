# When Can a Data-Driven Surrogate Be Trusted Inside a Shape-Optimization Loop?

**Muhammad Uns Haider Shah**

## Abstract

Neural-network surrogates trained on RANS data are usually assessed by flow-field error. For design optimization the relevant question is different: whether the surrogate orders candidate designs correctly, and whether the optimum it proposes survives scrutiny. Using the AirfRANS benchmark (200 training simulations), we show three things. First, a pointwise surrogate cannot rank designs unless it is given global shape information; supplying a thirteen-number shape descriptor reduces surface-pressure error fifteenfold and raises lift rank correlation from 0.984 to 0.998. Second, field-based surrogates predict lift well (2.6 % median error) but cannot rank drag at all (rank correlation 0.075), because 68 % of drag is viscous and is computed from velocity gradients across a 2 µm near-wall cell; surrogates that regress forces directly reach 0.3 % drag error and 0.999 rank correlation. Third, gradient-free and gradient-based searches both reach the surrogate's own optimum, but retraining on data subsets moves the optimal design by 14–20 % of the design range while its performance varies by only 2 %, and two surrogates trained on the same data disagree by 4 % where the optimizers compete over 0.3 %. Predicted uncertainty grows with distance from the training envelope and is a usable stopping signal there, but fails completely at a change of flow regime: a separated case is mispredicted by 80 %, at 139 times the predicted uncertainty. The conclusion is a mixed-fidelity one: surrogates identify a family of candidate designs cheaply; certifying a design requires high-fidelity verification.

## 1. Introduction

Surrogate-based optimization replaces expensive simulations with a cheap regression model. Its value depends not on the surrogate's average accuracy but on two narrower properties: whether it ranks designs consistently with the truth, and whether its optimum is a property of the physics rather than of the training sample.

The AirfRANS benchmark provides 1000 RANS simulations over NACA 4- and 5-digit airfoils, published with baseline field-prediction scores. This work reframes it as an optimization problem: train surrogates, optimise airfoil shape with them, and interrogate the result. The headline metrics are rank correlation and pairwise ordering accuracy, not field mean squared error.

## 2. Data and evaluation protocol

The *scarce* task is used throughout: 200 training simulations (split 160/40 for training and validation) and 200 test simulations shared with the *full* task. Reynolds number spans 2–6 million and incidence −5° to 15°; the solver is OpenFOAM with the k-ω SST model; meshes carry about 180,000 nodes after cropping.

Forces are obtained by the benchmark's own post-processing: predicted fields are written back into the simulation object and integrated over the airfoil surface, including the viscous contribution derived from the velocity gradient at the wall. An independent pressure-integration routine, written directly on the raw point data, agrees with the reference implementation to within 1 × 10⁻⁴ in lift across twenty cases, establishing the post-processing error floor.

Two ranking measures are reported. Spearman correlation over the whole test set is the conventional choice, but lift there is dominated by incidence, which every model receives as an input. The stricter measure is **pairwise ordering accuracy at comparable operating conditions**: over all test-case pairs whose incidence differs by less than 1° and whose Reynolds number differs by less than 0.5 million (479 pairs), the fraction the model orders correctly. This is the question an optimizer asks when comparing two shapes.

## 3. Shape information governs rankability

A pointwise multilayer perceptron predicting velocity, pressure and turbulent viscosity at each node from local features (position, wall distance, surface normal, inlet velocity) overfits severely: training loss falls monotonically while validation loss turns upward after roughly 350 epochs. Its surface-pressure error is 0.967 in normalised units, close to the variance of the field itself.

The cause is structural. Pressure at a point on the surface depends on the entire airfoil; a node cannot observe it. Appending a global descriptor to every node's input, consisting of the thickness ratio and the camber-line height at twelve chord stations, leaves everything else unchanged and produces:

| | pointwise | shape-aware |
|---|---|---|
| volume field MSE | 0.252 | 0.056 |
| surface pressure MSE | 0.967 | 0.063 |
| lift rank correlation | 0.984 | 0.998 |
| lift ordering, comparable pairs | 0.902 | 0.971 |

The gap between the two ranking measures is informative. The pointwise model appears competent when incidence varies freely (0.984, 0.95 pairwise) and degrades when it cannot rely on incidence (0.902). Aggregate rank correlation over a test set spanning a wide range of operating conditions therefore overstates a surrogate's usefulness for shape design.

## 4. Why field-based surrogates cannot rank drag

Integrating the shape-aware model's predicted fields gives a median lift error of 2.6 % and rank correlation 0.998, but a drag error of 1715 % and rank correlation 0.075. Three diagnostics locate the failure.

1. **Viscous drag dominates.** It accounts for 68 % of total drag (median over the test set) and carries a median error of 2700 %.
2. **The error is not at the wall node.** Imposing the exact no-slip condition before integration changes the result by less than 0.2 %, so the error lies in the nodes immediately above the wall, where the first cell is about 2 µm thick. Wall shear is effectively a velocity difference divided by that spacing, so a velocity error of 0.1 m/s produces an order-of-magnitude shear error. A smooth network cannot represent a layer that thin.
3. **Pressure drag fails independently.** It retains a 60 % median error even where surface pressure is accurate, because it is a small residual of large, nearly cancelling forces, some fifty times smaller than lift.

This is consistent with the published baselines, where every model attains drag rank correlations between −0.12 and −0.14. The failure is a property of the formulation, not of a particular architecture.

Regressing forces directly from the design variables removes the near-wall problem by construction. Two models are trained on the same 160 cases, mapping the shape descriptor, incidence and Reynolds number to lift and log-drag: an ensemble of ten small networks, and a Gaussian process with a Matérn-5/2 kernel and one length scale per input.

| 200 test cases | field model | NN ensemble | Gaussian process |
|---|---|---|---|
| drag error (median) | 1715 % | 0.8 % | 0.3 % |
| drag rank correlation | 0.075 | 0.998 | 0.999 |
| L/D rank correlation | 0.883 | 0.997 | 0.999 |
| drag ordering, comparable and distinguishable pairs | 0.66 | 0.995 | 1.000 |
| fraction within ±2σ | – | 0.55 | 0.91 |

The trade is explicit: the direct models sacrifice the flow field and generality across arbitrary geometries in exchange for force accuracy within a parametric family. For parametric design optimization that is the right trade; for a field predictor it is not. The Gaussian process is additionally well calibrated, which the ensemble is not, and its learned length scales identify incidence and thickness as the dominant drag drivers, then camber near 20 % and 80 % chord.

## 5. Optimization: gradient-free against differentiated surrogate

Two problems are solved over NACA 4-digit designs (camber, camber position, thickness), with bounds taken from the training cases and thickness constrained to at least 12 % as a structural stand-in: maximise L/D at Re = 4 × 10⁶ and α = 4°, and minimise drag subject to CL ≥ 0.8 with incidence free.

Both are solved on the *same* objective, the network ensemble, by three routes: Bayesian optimization with expected improvement (constrained by probability of feasibility in the second problem), a random search at equal budget, and multi-start L-BFGS-B/SLSQP using gradients obtained by automatic differentiation through the network and through the analytic camber-line formula. A 200,000-point sample of the objective provides the surrogate's own optimum, separating search error from surrogate error.

| method | evaluations | best L/D | gap to reference |
|---|---|---|---|
| Bayesian optimization | 50 | 91.73 | −0.03 % |
| random search | 50 | 88.89 | +3.07 % |
| gradient-based, 20 starts | 176 (8.8/start) | 92.01 | −0.33 % |

Sensitivities match central finite differences to seven decimal places. A single gradient-based search costs roughly a sixth of the Bayesian run, but one start in twenty converged to L/D = 76 rather than 92, so restarts are not optional. The practical reading: where the surrogate is differentiable, gradients make each local search almost free and the remaining cost is global coverage, which is exactly the cost Bayesian optimization is designed to pay. Neither route requires an adjoint solver; but neither, equally, provides the high-fidelity sensitivities an adjoint would.

## 6. Can the optimum be trusted?

Four checks, none requiring new simulation.

**Position.** The optimum's nearest training design is 0.130 away in normalised design-variable units, closer than the median spacing between training designs (0.158). It is interpolation. The Gaussian process gives its drag as 0.01256 with a ±2σ interval of ±2.3 %.

**Stability under resampling.** Retraining on eight random 80 % subsets and re-optimising yields L/D = 93.8 ± 1.6 and lift-constrained drag 0.0099 ± 0.0001. The designs themselves move much more: camber from 4.8 to 7.0 and camber position from 4.2 to 6.9, some 14–20 % of the design range. Thickness sits on the 12 % bound in every run. The objective has a flat ridge: the surrogate identifies a family of near-equivalent designs, and which member appears best is a property of the sample. Since two surrogates trained on the same data disagree by about 4 % on a given design's L/D, while the optimizers compete over 0.3 %, surrogate accuracy, not search quality, is the binding limitation.

**Controlled extrapolation.** Narrowing the training band to the central 60 % of a variable and testing outside it:

| band narrowed in | inside | near outside | far outside | error/σ far outside |
|---|---|---|---|---|
| Reynolds number | 0.8 % | 1.3 % | 2.4 % | 0.89 |
| angle of attack | 0.3 % | 0.5 % | 4.0 % | 1.60 |

Predicted uncertainty grows roughly in step with actual error, so it can legitimately be used to stop an optimizer at the edge of the data.

**A failure that uncertainty cannot see.** The worst-predicted case in the dataset is a 5.2 %-thick airfoil at −4.4° incidence. Its flow separates along the lower surface: drag is 4.1× the training median and 93 % of it is pressure drag, against 33 % typically. The surrogate under-predicts by 80 %, at **139× its predicted uncertainty**. Only 7 of 200 training cases lie in that region.

The distinction this draws is the central result. Uncertainty estimates measure distance from the data in input space and behave well there. A change of flow regime is invisible to them, because the inputs look unremarkable: nothing about "5 % thickness at −4°" is far from the training set, yet the physics is different. Predicted confidence bounds interpolation error, not physics error.

Note also that an optimizer minimising drag is drawn towards thin sections, which is precisely the direction of this failure. The minimum-thickness constraint that makes the problem physically sensible also, and not coincidentally, keeps the surrogate inside its valid regime.

## 7. Limitations

AirfRANS is incompressible and subsonic; much benchmark shape-optimization work is transonic, where shocks and wave drag dominate, and the flow regime does not transfer even though the methodology does. The design space has three variables, the regime in which surrogates are viable; sample requirements rise sharply with dimension, which is why adjoint methods, whose cost is essentially independent of the number of design variables, exist for the opposite regime. The direct force models apply only within the NACA families at the trained conditions and are not field predictors, so comparisons with the benchmark's published models, which learn from the mesh alone, favour this work by construction. Results are single training runs against published five-run means. No high-fidelity verification of the optima was performed. Finally, 88 of 496 cases in the benchmark's `reynolds` and `aoa` splits belong to this model's training set, since those splits are drawn from the same 1000 simulations as the `full` task; they were removed before evaluation.

## 8. Conclusion

A surrogate trained on 200 RANS simulations solves a three-variable airfoil optimization problem in seconds, to within a fraction of a percent of its own optimum, and orders designs by drag essentially perfectly when forces are regressed directly rather than integrated from predicted fields. What it cannot do is certify its answer. The optimum's location is sample-dependent even where its performance is stable; two surrogates fitted to identical data disagree by more than the improvement being sought; and confidence estimates, reliable against distance from the data, are blind to a change of flow regime.

The appropriate role is therefore exploratory. A surrogate narrows a large design space to a family of candidates cheaply and identifies which constraints bind. Verification, and any refinement demanding trustworthy sensitivities, belongs at high fidelity, where for many design variables the adjoint formulation is the tool built for the job.

## Future work

Three extensions follow naturally. Retraining on the *full* task (800 cases) would separate sample-limited error from model-limited error, since the optimum's instability under resampling is a data-quantity symptom. Active learning would place the next expensive simulation where it most reduces uncertainty along the flat ridge, rather than uniformly. An operator-learning architecture, with one branch encoding geometry and operating condition and another taking query coordinates, would restore field prediction while retaining the global shape conditioning that proved decisive here.
