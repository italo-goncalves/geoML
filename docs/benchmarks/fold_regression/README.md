# The folded-ore and chapter 17 regressions (2026-10-19)

`fold_test.py` runs the deep model of the user's `non_stationary_test.py`
(2022 VGP article: three drillholes through a folded ore layer, a walked
field read by a Cubic GP, transform range 2) on whatever geoML is first on
the path, and counts the ore bodies (P > 0.5, 8-connected): one is the
fold, five are separate blobs. `ZERO_BIAS=1` fixes every GP node's bias at
zero. `ch17_test.py` runs chapter 17's deep model as the 0.8.0 chapter
wrote it (`PROPAGATION=marginal` reads the deep input by its marginals).
`fold_maps.py` and `ch17_maps.py` draw the saved maps.

Each version is a package folder on its own path, extracted with

    git archive <tag-or-commit> geoml | tar -x -C versions/<name>

and the user's 2025-09-08 copy (`Lab/geoML/geoML 2025_09_08`) as
`versions/snapshot/geoml` with `c8e4c52`'s `__init__.py`; then, from the
version's folder, `PYTHONPATH=versions/<name> python fold_test.py <name>
<seed> 2 <maps folder>`.

Findings (`figures/`): the fold holds in the 2025-09-08 copy and
`c8e4c52` and is lost in every release from v0.3.5 on, which gave every
GP node a trainable bias (the prior mean); fixed at zero, every version
since folds again. Chapter 17's 0.8.0 model is identical from 0.8.0 to
0.8.8 and changes at 0.9.0, the joint propagation, which the marginal
rule undoes to the digit. The report in `docs/reports/deep_second_moment`
has the numbers.

## The folded vein drilled (2026-10-22)

`fold_drillholes.py` asks the question the three symmetric holes could not
settle: given a folded vein and drillholes through it, does the deep model
draw the fold better than a stationary one? The truth
(`figures/vein_truth.npz`) is the ore region (P > 0.5) of the ablation's
`m32_joint_dall001` map, a folded layer with a hinge and a second closure.
The holes are collared at the top and sampled every 0.2 m from the truth's
nearest cell. The two models are the same but for the walk: a base GP at
range 20 beside a Matern32 GP at range 2, which reads the coordinates
(stationary, a range per axis) or a walk over them (deep, an isotropic
reader). Both use the hierarchical indicator, `GPOptions` defaults (joint
rule, full variance) and 500 steps of the model's own optimizer.
`fold_drillholes_maps.py` draws the maps and `fold_drillholes_walk.py`
draws the walk (`predict_node` on the walk's start and end).

| scored against the truth | 5 holes, stationary | 5 holes, deep | 3 holes, stationary | 3 holes, deep |
|---|---|---|---|---|
| samples (ore) | 497 (131) | 497 (131) | 303 (87) | 303 (87) |
| accuracy | 0.940 | **0.962** | 0.912 | **0.929** |
| IoU of the ore | 0.790 | **0.859** | 0.701 | **0.733** |
| Brier | 0.046 | **0.029** | 0.068 | **0.054** |
| log score | 0.172 | **0.100** | 0.306 | **0.239** |
| bound | -104.0 | **-93.5** | **-66.7** | -69.0 |
| bodies (truth: 1) | 1 | 1 | 1 | 1 |
| seconds | 25 | 700 | 23 | 490 |

Seeds 1-3 agree to the third decimal (`figures/drillholes_results.jsonl`),
so each column is one result.

- **Five holes** (the default `HOLES`: four inclined, every hole cutting
  the vein two or three times, 11 intercepts; `figures/drillholes_5.png`).
  The deep model is better on every score and on the bound. It draws the
  upper limb as one band up to the surface, a sharp hinge and the
  lower-left limb at its thickness. The stationary model pinches the upper
  limb into a blob and swells the lower-left limb.
- **Three holes** near the article's, less symmetric (`holes=-4.5/5_-0.5/0_5.5/-8`:
  collars 4 and 6 m apart, the outer holes tilted 5 and 8 degrees opposite
  ways; `figures/drillholes_3.png`). The deep model is still better against
  the truth, by less, at a bound 2.4 nats lower. Neither finds the upper
  limb above the middle hole's first intercept. Both draw one body, where
  the article's symmetric holes gave the 0.9.4 deep model five blobs.
- **The walk** (`figures/drillholes_walk.png`) does not unfold the vein
  into a flat layer. In the reader's space the truth keeps its shape. The
  five-hole walk (amp 1.77) moves points by up to 0.33 transformed units
  and 0.13 on average, where the section is about 2.2 units wide. Most of that is in the hinge and along
  the right limb, pulling the limbs together across the waste between
  them, so the stationary reader above sees the intercepts closer than
  they are on the section. The three-hole walk (amp 1.39) moves points
  0.17 at most and 0.05 on average, barely bending the holes. The deep
  model's gain is a local change of spacing, a non-stationary range, not
  a restoration of the fold.
