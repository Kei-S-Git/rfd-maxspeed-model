# Analysis code for "A finite rate of force development as a candidate constraint on maximal running speed"

This is the code and data availability package for the manuscript of the same name.
It is deliberately venue-neutral: the analysis, the figures, and the provenance of every
number are the same whichever journal the manuscript is under.

## What this is

A minimal impulse model of the limit on maximal running speed, its calibration, an
algebraic audit of what that calibration can and cannot test, and an out-of-sample test
against four independent laboratories.

**No new measurements were made.** Every empirical value is transcribed from a published
table. There are no external data files: each value appears in the source code next to
its citation, its sample size, and its role (calibration or test), so the provenance of
any number in the paper can be traced by reading one file.

- Calibration values (Weyand et al. 2010, Table 1): `src/calibrate_parameters.py`,
  class `Measured`.
- Extrapolation values (Weyand et al. 2000): `src/elite_recalibration.py`, module
  constants.
- Out-of-sample cohorts (Paradisis et al. 2019; Kuitunen et al. 2002):
  `src/external_validation.py`, list `COHORTS`.

## Reproducing everything

```
pip install -r requirements.txt
python scripts/reproduce_paper1.py
```

Runs in well under a minute on a laptop and regenerates every number and all five
figures into `output/` (PNG at 300 dpi plus vector PDF). The versions used in the
manuscript are shipped in `figures/` for comparison; a fresh run should reproduce them.

```
python scripts/reproduce_paper1.py --check
```

runs only the machine-checked claims, without drawing figures.

## Modules

| File | What it does |
|---|---|
| `src/rfd_maxspeed_model.py` | The model: available impulse, peak force, the closed-form `v_max(R)`, a numerical root of the same balance for cross-checking, and a `sympy` re-derivation. Its `main()` also draws an **illustrative** figure with the class defaults; that figure is written as `rfd_maxspeed_scaling_illustrative.png` and is *not* the manuscript figure. |
| `src/calibrate_parameters.py` | Calibration against Weyand et al. (2010) and the force budget. Produces Fig. 2. |
| `src/elite_recalibration.py` | Extrapolation over the Weyand et al. (2000) speed range and the required rate. Produces Fig. 3. |
| `src/degeneracy_audit.py` | **The algebraic audit.** Proves symbolically that the calibration makes `beta` and `R` cancel from the impulse balance, that the two customary within-dataset comparisons are one statement, that the "non-circularity" ratio reduces to `kappa/(2(kappa-1))`, and that the elite-extrapolation result is a quadratic in `v` by construction. Every claim is an `assert`, so the module fails loudly if any of them stops holding. |
| `src/external_validation.py` | The out-of-sample test: the four independent cohorts, the transfer of the structural assumptions, the refutation of a universal capacity parameter, and the `A ~ v^2.15` scaling. Produces Fig. 4. |
| `src/robustness_analyses.py` | Sensitivity to contact length and to waveform asymmetry. Produces Fig. 5. |
| `src/figure_scaling_law.py` | **Fig. 1**, drawn from the calibrated parameters. |
| `src/figstyle.py` | Shared figure style: ≤180 mm wide, 300 dpi plus vector, 8 pt sans labels, 12 pt bold uppercase panel letters. These are the settings the submitted figures were produced with; they satisfy the stricter of the limits considered and are not specific to one journal. |
| `scripts/reproduce_paper1.py` | Runs all of the above in order and verifies that every figure was produced. |

## Claims that are machine-checked

`src/degeneracy_audit.py` asserts each of the following, so `--check` fails if any
breaks:

- **D1** Substituting the calibration into the rate-limited available impulse cancels
  `beta` and `R` identically, leaving `F_avg * t_c^2 / T_c`.
- **D2** The resulting top-speed prediction (9.01 m s⁻¹) is the average-force comparison
  restated; it contains no rate information.
- **D3** The ratio of the required rate to the rate obtained from the reported rise time
  is `kappa/(2(kappa-1)) = 1.175`, independent of the peak force they share.
- **D4** The required rate as a function of speed is a quadratic that is monotonically
  increasing by construction once the shape factor, contact length and aerial time are
  held speed-invariant.
- **D5** The one transferable quantity is the capacity slope `A`, and freezing it gives a
  prediction that varies by more than 1 m s⁻¹ across plausible cohort parameters — so
  the out-of-sample test is not trivially satisfied.

`src/rfd_maxspeed_model.py` additionally checks the closed form against a numerical root
across four decades of `R` and re-derives it symbolically.

## Licence and citation

Released for review alongside the manuscript. Please cite the paper. A public repository
with a DOI will be created on acceptance; the identifier will be added to the manuscript's
data and resource availability statement.
