# Mixed precision: measured and rejected

geoML computes everything in float64. On consumer GPUs (here an RTX 4060
Ti) float64 runs at 1/64 of float32's FLOP rate, which makes a mixed
scheme — the kernel/covariance arithmetic in float32, the Cholesky
decompositions kept in float64 — look like an obvious win. It was tried
on 2026-08-12, measured end to end, and rejected. This is the record, so
the idea is not re-litigated from scratch.

## The scheme

A script-local patch on `BasicGP.covariance_matrix`: cast the inputs and
`ranges` to float32, run the whole chain (difference tensor, distances,
`kernelize`, normalization) in float32, cast the result to float64
before anything decomposes or solves with it. Nothing else changes —
training, likelihoods and solves stay float64. A/B on the Walker Lake
VGP (470 samples, 576 inducing points, Gaussian kernel/likelihood, same
seed both arms).

## What was measured

**Local coordinates are mandatory, not advisable.** At UTM magnitudes
with fractional coordinates, float32 snaps a northing of 4.1e6 to a
0.125 m lattice, errs pairwise distances by up to 0.25 m and covariance
entries by **1.1e-2** — four orders of magnitude above any usable
jitter. The same arithmetic in a local frame (centroid removed in
float64 first) errs at 6e-7, ordinary float32 roundoff. Benchmark trap
found on the way: integer coordinates below 2^24 are *exact* in
float32, so integer-gridded data (Walker) shows no error at all until
the coordinates are de-integerized.

**The Cholesky needs float64 as a matter of arithmetic, not caution.** A
float32 Cholesky of a realistic Gaussian-kernel matrix (1500–3000
points, 1e-6 jitter) fails outright — the matrix is not positive
definite at float32 precision. cuSOLVER returns NaNs. There is no slow
version of this to tolerate; it does not run.

**Gradients keep flowing through the casts.** Zero disconnected
gradients, initial loss identical to stock to 8 significant digits,
matching gradient norms, median per-element deviation 2.6%.
Structurally the scheme trains; differentiability is not the problem.

**Stability is the problem, and the cure costs more than the disease.**
The float64 Cholesky is fed a matrix that is only float32-accurate
(~6e-7), so the jitter must cover that noise:

| jitter | mixed arm |
|---|---|
| 1e-9 (default) | NaN at the first decomposition |
| 1e-6 | tracks stock to ~iteration 50, then one Cholesky fails and poisons the rest |
| 1e-5 | stable over 200 iterations; matches stock@1e-5 (prediction RMSE 0.44 on sd 206) |

But raising the jitter from 1e-6 to 1e-5 moves the *float64* model
itself by prediction RMSE 7.1 — **the jitter the scheme forces perturbs
the model sixteen times more than the float32 arithmetic contributes.**

**And the speedup is not there.** The covariance chain alone is 2.5×
faster in float32 (it is bandwidth-bound; halving the bytes is the
entire gain — the 1/64 FLOP penalty never binds because the workload is
not FLOP-bound). End to end: training 1.1–1.2×, prediction 1.13×, and
**nothing on top of XLA** — 0.22 s against 0.23 s on the 78k-node grid,
because XLA fusion already eliminates the intermediate tensors whose
traffic float32 would have halved. What remains after XLA is Choleskys
and solves, which float32 cannot touch by rule and, per the above, by
arithmetic.

## Verdict

`GPOptions(jit_predict=True)` is the shipped answer: 3–5×, numerically
exact, no jitter tax. Mixed precision buys ~1.15× where it works and
sits one Cholesky failure from NaN where it doesn't. Revisit only on
hardware where float64 bandwidth is the true limit *and* training's
n×m chains dwarf the m³ Choleskys — and then in a local frame, with the
jitter floor raised knowingly.
