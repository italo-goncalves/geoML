# 1. Why another geostatistics

geoML models spatial variables (grades, thicknesses, rock types, whole
assay suites) with variational Gaussian processes. That sentence contains
no rejection of geostatistics, and this manual spends its first chapters
proving the opposite. The Gaussian process **is** kriging, re-derived as a
probability model (chapter 2), and everything the package adds is built by
editing that model where kriging left it implicit.

The additions are the reason to bother, so here they are up front, each
with the chapter that delivers it.

- **One objective instead of a fitted variogram.** Ranges, sills, nugget
  and anisotropy angles, the rotation included, are estimated jointly by
  maximum likelihood. The experimental variogram becomes a *check*
  (chapter 13) rather than the fitting surface (chapter 2).
- **Scale.** Inducing points compress the model and stochastic training
  feeds it in batches, so tens of thousands of assays train on a desktop
  GPU where the closed form stops at a few thousand (chapter 3).
- **Data as it comes.** Skewed, bounded, compositional and censored
  variables are handled by warpings, and outliers by robust likelihoods,
  up to a mixture that *names* the contaminated samples instead of merely
  surviving them (chapter 4).
- **Structure as hypothesis.** Models are networks of latent nodes, added,
  multiplied, combined and stacked into deep GPs, so the model's shape can
  say what the geologist believes: a trend plus a local structure, a
  lithology feeding a grade, a folded space unfolded (chapters 5 and 6).
- **Uncertainty as a product, not a by-product.** Simulation is native
  (chapter 7) and support is explicit (chapter 8: a block's value, the
  spread inside it, and the scatter of an assay of it are three different
  numbers, kept apart). The intervals are *audited*, by a cross-validation
  built for spatial data and a calibration that repairs what it finds
  (chapter 13).
- **The workflow, not just the model.** Drillhole desurveying and
  compositing, variable-size block models, triangulated surfaces and their
  booleans, and exports to the tools a mine already uses (Part II).

And what kriging keeps: at a few hundred samples, with a variable a normal
score transform genuinely tames, ordinary kriging in any established
package is fast, auditable and fine. The gains above are bought with
gradient-based training and a probability model that has to be checked.
The package ships the checking machinery precisely because the flexibility
is worthless without it.

## How to read this manual

Part I (chapters 2–8) is the model, one idea per chapter, in
geostatistical terms first and code second. Each theory section closes
with an **In the code** box naming the classes that implement it, and each
chapter carries a small runnable example on the bundled data. Derivations
stay in the papers, which are cited where they apply and listed in full at
the end of every chapter.

Part II (chapters 9–14) is the workflow, from a drillhole database to a
reported block model. Part III runs three case studies end to end, and it
is the place to start if you would rather read code than prose. Chapter
15's Walker Lake study touches almost everything once.

The examples assume `import geoml` and nothing else. Every code block in
this manual is executed, in order and in one namespace per chapter, by
`docs/manual/run_blocks.py`, so if the text and the package ever disagree
the chapter fails before it can mislead anyone.
