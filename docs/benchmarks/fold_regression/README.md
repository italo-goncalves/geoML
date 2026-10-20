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
