# Sparse expert activation in geoML — analysis, measurements, plan

Nothing in `geoml/` has been changed. Everything below is measured on the
current `master` (0.5.5) with benchmark scripts kept in the session scratchpad
(`b0`–`b8`, `c1`–`c8`), run in the WSL `geoml` conda env on CPU.

A second round of measurements answered several questions in the *negative*,
and the plan in §8 is smaller and differently shaped than the first draft.

---

## 1. What "expert" already means in the code

Three unrelated things are called experts; only one is the subject here.

| Mechanism | Where | What it is |
|---|---|---|
| **Per-node experts** | `latent.BasicInput(inducing_points=[a, b, c])` | `n_experts = len(inducing_points)`. *Every* node carries one copy of its state per expert, and `BasicGP` combines them per query point. **This is the one that matters.** |
| `latent.ProductOfExperts` | a node | Combines whole sibling subnetworks, each with its own root. |
| `models.GPEnsemble` | a model | K separate legacy `GP` models combined at prediction. |

`BasicGP` holds `alpha_white_i`, `delta_i`, `bias_i` for `i in range(K)`
(`latent.py:686`), and every state tensor is a **Python tuple of length K**.
Experts may have different numbers of inducing points (`root.n_ip` is a tuple).

Combination is precision weighting (`_GPNode.get_expert_weights`,
`latent.py:449`): `w_e = (1 - var_e) / var_e`, normalised over `e`. A far-away
expert has `var_e → 1` and so `w_e → 0` — but it still costs a full `[n, m]`
cross-covariance and an `[n, m] × [m, m]` solve to discover that.

**No test exercises K > 1.** Every `BasicInput(...)` call in `geoml/test/`
passes a single container. The path works (b0 builds, trains and predicts a
K = 3 model on Walker Lake), but it is unguarded — including in
`test_model_persistence.py`.

---

## 2. Where the time actually goes (b1, b2)

Walker Lake, 470 data points, m = 60 inducing points per expert, one 4096-point
prediction batch.

| K | `refresh` (eager, as `predict` calls it) | same in a `tf.function` | `predict_raw` / 4096 pts | full `predict` |
|---:|---:|---:|---:|---:|
| 1 | 5.2 ms | 0.3 ms | 6.4 ms | 12.6 ms |
| 8 | 119.7 ms | 1.2 ms | 22.6 ms | 155.5 ms |
| 32 | **1694.5 ms** | 11.9 ms | 107.0 ms | 1886.8 ms |

At K = 32, **90 % of a `predict` call is one eager `refresh`**, and it is
quadratic in K. `BasicGP.refresh` ends with a double loop
(`latent.py:812–836`) that predicts every expert's inducing points from every
other, to hand them to the next node — and it runs even when the node has no
children. `models.py:1023` calls the refresh in eager Python; training does
not, because `_training_elbo` is a `@tf.function`.

Only `BasicGP` and `GradientConstrainedInput` have the K² loop — they are the
only nodes that *interpolate* the inducing points. `Add`, `LinearCombination`,
`Stack`, `Concatenate` and `GPWalk` combine them elementwise and are O(K).

---

## 3. Two exact fixes, independent of sparsity (b6)

Both prototyped; both give **bit-identical** predictions (max difference 0.0).

| K | `predict` now | + skip the childless node's propagation | + refresh in graph mode |
|---:|---:|---:|---:|
| 8 | 154.0 ms | 40.7 ms | 29.8 ms |
| 32 | 1914.6 ms | **178.8 ms** | **152.4 ms** |

- **Skip the propagation on a childless node** — a one-line `if self.children:`
  guard. It only helps the *terminal* node, so the win shrinks sharply with
  depth (c9, K = 8, output identical in every case):

  | GP layers | `predict` now | with the guard | speedup |
  |---:|---:|---:|---:|
  | 1 | 112.9 ms | 20.1 ms | **5.6×** |
  | 2 | 213.6 ms | 134.3 ms | 1.6× |
  | 3 | 312.1 ms | 232.7 ms | 1.3× |

  The saving in absolute terms is flat — about 80–90 ms, one node's worth of
  propagation — while the total grows with depth. Free either way.
- **Run the once-per-`predict` refresh in graph mode.** Not a bare
  `tf.function(refresh)`: `refresh` writes results onto node attributes, and an
  attribute written while tracing holds a symbolic tensor that is unusable
  afterwards. The refresh must *return* every node's state so it can go into
  the `_state_var` Variables `cache_prediction_state` already maintains.

---

## 4. How concentrated are the weights?

### 4.1 Two dimensions — very (b3, b4)

K = 16 on Walker Lake, 16384 query points: 99.9 % of the weight needs 4.3
experts on average, 9 at worst. Truncation error against the full combination,
as a fraction of the field's standard deviation:

| k | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| mean error | 9.41 % | 2.29 % | 0.74 % | 0.23 % | **0.03 %** | 0.00 % |

And k barely grows with K:

| K | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| experts for 99.9 % (mean) | 2.15 | 2.55 | 2.93 | 3.24 | 4.25 |
| k for < 0.1 % error | 4 | 3 | 4 | 4 | **5** |

### 4.2 Three dimensions — much less (c6)

Macpass Zn assays, 8000 samples, block model restricted to the 4598 cells
within 40 m of a hole (far from data every expert predicts the prior, so the
weights there are uninformative but the predictions agree anyway).

| K | experts for 99.9 % (mean) | k=4 | k=8 | k=16 | k=24 | k=32 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8.0 | 20.7 % | 0.00 % | – | – | – |
| 16 | 14.8 | 36.2 % | 13.6 % | 0.00 % | – | – |
| 32 | 21.2 | 44.1 % | 15.0 % | 1.66 % | 0.09 % | 0.00 % |
| 64 | 22.6 | 12.3 % | 4.76 % | 1.01 % | 0.21 % | 0.04 % |
| 128 | 23.0 | 8.86 % | 3.17 % | **0.67 %** | 0.20 % | 0.07 % |

**k lands near 20–24 instead of 5, but it saturates** (21.2 → 22.6 → 23.0 as K
goes 32 → 64 → 128), which is what matters: the fraction of kept work still
falls as K grows. A point in space has ~15 Voronoi neighbours against ~6 in the
plane, plus second-order ones inside the kernel range, so ~23 is about right.

A synthetic cube — 8000 points filling 100³ — behaves quite differently, and
the difference is the whole point (c6b):

| K | experts for 99.9 % | k=8 | k=12 | k=16 | k=24 | k=32 | k=48 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 14.8 | 3.03 % | 0.82 % | 0.00 % | – | – | – |
| 32 | 24.6 | 5.64 % | 2.59 % | 1.28 % | 0.26 % | 0.00 % | – |
| 64 | **37.8** | 8.13 % | 4.56 % | 2.68 % | 1.06 % | 0.45 % | 0.07 % |

**The cube does not saturate**: 14.8 → 24.6 → 37.8 as K goes 16 → 32 → 64, so
the relevant experts stay a roughly constant *fraction* of all of them and the
saving never improves. At K = 64, k = 32 — half the experts — is needed for
0.45 %.

### 4.3 What actually decides it: the range against the expert spacing (e1)

Neither dimension nor drillhole-versus-cube is the real variable. It is
**how far the kernel reaches compared with how far apart the experts are**: a
range that spans several experts keeps several of them at non-negligible
weight, and there is nothing left to truncate.

Cube of 8000 points in 100³, K = 64, so the experts sit 25 apart. The transform
range is pinned and `BasicGP(fix_range=True)` holds its multiplier at one, so
the effective range is exactly what is set:

| range | range ÷ spacing | experts for 99.9 % | k=4 | k=8 | k=16 | k=32 |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.24 | 33.9 | 0.53 % | 0.49 % | 0.42 % | 0.28 % |
| 12 | 0.48 | **4.4** | 0.22 % | 0.02 % | 0.01 % | 0.01 % |
| 25 | 1.00 | 11.9 | 3.33 % | 0.53 % | 0.04 % | 0.00 % |
| 50 | 2.00 | 32.7 | 13.69 % | 6.10 % | 1.76 % | 0.24 % |
| 100 | 4.00 | 58.6 | 32.62 % | 21.82 % | 10.88 % | 3.40 % |

Monotone from 0.48 upward, and steeply: at four times the spacing, 59 of the 64
experts carry weight and even half of them leaves 3.4 % error.

The shortest range is the one exception worth understanding. At 0.24 the weight
looks spread over 33.9 experts, but truncating costs almost nothing (0.53 % at
k = 4, barely improving with more). That is the regime where *no* expert
explains the point: every variance is ~1, so the weights are near-uniform and
uninformative, but every expert is predicting the prior anyway. The count is
misleading there; the error is not.

**And free training lands in the bad regime.** Left to train, the same model
settles at an effective range of 31.3 — the transform's 25.0 times `BasicGP`'s
1.25 — which is **1.25 × the expert spacing**, and gives 16.6 experts for
99.9 %. That is why the cube did badly at K = 64: the field's own correlation
length is longer than the spacing that many experts implies.

So the design rule is not about geometry but about resolution: **experts may be
subdivided only down to the kernel range.** Past that they all see the same
structure and none can be dropped. Macpass saturates because its experts stay
far apart relative to what the kernel reaches; the cube does not because
64 experts in 100³ put them closer together than the field's correlation
length. It also means K is bounded above for any given field — roughly, the
domain divided by the range cubed — and pushing past that costs accuracy
whether or not activation is sparse.

The practical consequence: **in 3D there is nothing to gain below K ≈ 32.** At
K = 128, k = 24 keeps 19 % of the work for 0.20 % error — roughly a 5× saving,
against ~15× in the plane. Small k is not an option there: at K = 128, k = 4
still carries 8.9 % error.

---

## 5. Four ways to write it in TensorFlow (b5, c2, c5)

### 5.1 The first comparison, which was misleading

n = 4096, m = 60. A: Python loop over **all K**. B: all K stacked and batched.
C: gather **k of K**. D: MoE capacity routing.

| K | k | A: loop (today) | B: stacked | C: gather k | D: dispatch |
|---:|---:|---:|---:|---:|---:|
| 32 | 5 | 64.8 ms | 70.0 ms | **10.1 ms** | 53.7 ms |
| 128 | 8 | 252.6 ms | 296.4 ms | **16.2 ms** | 121.3 ms |

B reproduces A exactly and is **not faster** — batching the experts into one
einsum buys nothing on CPU, and allocates `K × n × m` at once (2.4 GB at
K = 128 on a 40 000-point grid). **D — the standard Mixture-of-Experts trick —
is the wrong tool**: spatial routing is badly load-imbalanced, so at K = 128 the
busiest expert needs 1154 slots against an average of 256.

So C's win came **entirely from evaluating fewer experts**, not from batching.

### 5.2 Ragged experts are not possible (c2)

Tried directly, on TF 2.20:

| operation on a `RaggedTensor` | result |
|---|---|
| broadcast subtract `x[:, None, :] - ragged` | OK → `(3, 4096, None, 2)` |
| `tf.matmul` on a ragged inner dim | `ValueError: inner values have inconsistent shape` |
| `tf.einsum` with a ragged operand | `ValueError: inner values have inconsistent shape` |
| `tf.linalg.cholesky` | `ValueError: TypeError: object of type 'RaggedTensor' has no len()` |

The distances can be formed raggedly; none of the linear algebra can. **Ragged
is out.** The rectangular alternative is padding to `m_max` and zeroing the
cross-covariance on the padded columns, which is *exact* (max difference
1e-16) but costs:

| spread of m across experts | `m_max` / mean | padded vs an exact loop over the same k |
|---|---:|---:|
| all equal (60) | 1.00× | 1.30× slower |
| ±10 % (54–66) | 1.08× | 1.35× |
| ±25 % (45–75) | 1.23× | 1.51× |
| ±50 % (30–90) | 1.43× | 1.89× |
| one outlier (60, one 150) | 1.00× | 3.20× |

### 5.3 The real choice: loop over the selected experts, or gather them (c5)

Note that the padded column above is slower than *an exact loop over the same k
experts*. That reshapes the design: once k experts are selected, a plain Python
loop over those k beats any batched form. But selecting in Python makes the
graph depend on the selection, so each group needs its own trace.

Whole-grid prediction, batches of 2048 points, one group per expert:

| K | k | dense (today) | loop, cold | loop, warm | gather, cold | gather, warm |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 6 | 0.79 s | 3.46 s | **0.18 s** | 0.29 s | 0.22 s |
| 128 | 8 | 12.14 s | 18.07 s | **0.87 s** | **1.08 s** | 1.07 s |

Traces built: dense 1, loop K, gather 1.

- **A one-shot prediction must use `gather`**: 1.08 s against 18.07 s, because
  tracing K graphs dwarfs everything else.
- **Training could use either**: over thousands of iterations the traces
  amortise, the loop is ~20 % faster, and it is ragged-native.

### 5.4 Several graphs can share the Variables (c3)

Directly answering the question: **yes, and the state is not duplicated.**
Eight traces of the same `tf.function` over eight batch sizes captured 64
distinct Variables — exactly the 64 the model has. Variables are resources,
captured by handle, not copied into each graph.

What each extra trace does cost, for a K = 32 network:

| | per trace | 8 traces |
|---|---:|---:|
| build time | ~0.22 s | 1.8 s |
| RSS | ~17 MB | 134 MB |

Padding every batch to one fixed size and carrying a mask gives **1 trace,
12 MB, 0.3 s** for the same eight calls. At K = 128 the untamed version would
be ~28 s of tracing and ~2 GB of graphs.

---

## 6. End to end, on a trained model (b7)

Walker Lake subsampled to 5000 points, 90 000-point grid, dense = the Python
loop geoML uses today, sparse = grouped by primary expert with k gathered.

| K | k | points per group | dense | sparse | speedup | error as % of the field's spread |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 6 | 2876 | 2524.8 ms | 435.5 ms | **5.8×** | 0.37 % |
| 128 | 8 | 712 | 10190.4 ms | 875.2 ms | **11.6×** | 0.17 % |

**Groups must stay big enough to be worth a graph call.** An earlier run with
only 10 000 grid points and K = 128 (78 points per group) got 4.3× where the
kernel benchmark predicts ~15×. This pays off on block models, not test grids.

---

## 7. Training (c4, c7, c8)

This is where the plan changed most.

### 7.1 Spatial batches do train worse, and the obvious fixes do not help

Walker Lake, 6000 training points, 2000 held out, K = 16 experts, 12 batches of
~500, identical seeds. Held-out RMSE (the truth has sd 248):

| epoch | random | spatial, fixed order | spatial, shuffled order | spatial, KL rebalanced | spatial, re-clustered |
|---:|---:|---:|---:|---:|---:|
| 1 | 225.65 | 246.81 | 246.50 | 246.30 | 248.74 |
| 12 | **165.32** | 201.97 | 202.89 | 202.42 | 204.91 |
| 24 | **156.14** | 192.77 | – | – | 194.90 |

- **Shuffling the cluster order does nothing** (202.89 vs 201.97), which
  confirms the earlier finding rather than fixing it.
- **Rebalancing the KL does nothing either.** The hypothesis was that with
  spatial batches an expert gets likelihood gradient in 1 batch out of K at K×
  magnitude, while its KL gradient arrives on every step at full magnitude, and
  that Adam's RMS normalisation would damp the intermittent signal against the
  steady one. Weighting each expert's KL by how often that expert is active —
  so that per epoch each accumulates its full KL exactly once, matching what it
  accumulates of the likelihood — moved the result by 0.5 %. Refuted.

The curves descend at the same rate at epoch 12 (0.87 vs 0.88 RMSE per epoch),
so the deficit is a persistent offset rather than a slower start.

### 7.2 What is left to try

Recorded so the next attempt does not repeat the above. The remaining suspects:

- **Gradient diversity — refuted** (c8). A fixed partition gives the optimiser
  the same 12 directions for the whole run, so re-clustering with a new seed
  every epoch should have helped. It does not: 194.90 at 24 epochs, marginally
  *worse* than the fixed partition's 192.77 and nowhere near random's 156.14.
  (Note the batches must be drawn to a fixed size — a fresh partition gives a
  new batch length every epoch, and `_training_elbo` retraces for each one; the
  first attempt built up to 288 traces and ran for hours.)
- **The shared global parameters.** The transform range, the kernel range, the
  warping and the noise are estimated from a single region per step, and unlike
  the per-expert parameters they have no locality to fall back on. This is the
  last suspect standing.

Until one of these works, **training stays dense**. Prediction sparsity is a
pure approximation that leaves the fitted model untouched; training sparsity
changes the objective, and as measured it also costs accuracy.

---

## 8. Plan

Steps 1–4 are exact: same numbers, faster. Steps 5–6 trade a measured
approximation for speed. Step 7 is blocked on §7.2.

| # | Step | Effect | State |
|---|---|---|---|
| 1 | `if self.children:` guard on the propagation loop | 10.7× at K = 32 on a single-layer net, less with depth | **done** |
| 2 | Tests for K > 1 — build, train, predict, save, load | guards everything below | **done** — `test_experts.py`, 18 tests |
| 3 | `geoml/inducing.py`: `from_kmeans`, `from_grid`, `combine`, `grid_experts`, `experts` | usability; produces the neighbour structure the rest needs | **done** |
| 4 | Refresh traced, state returned and assigned | removes the remaining refresh cost | **done** — `latent.refresh_cached` |
| 4b | Trace the training step (§9) | 1.5× at K = 5, 2.8× at K = 40 | **done** — `VGPNetwork._training_step` |
| 5 | Stack the derived state; gather k experts in `interpolate`; group prediction batches by expert | 5.8–11.6× on prediction, but only for K ≳ 32 | **not worth building** — see below |
| 6 | Neighbour-restricted propagation in `refresh` | only matters for multi-layer networks | dropped with step 5 |
| 7 | Sparse activation during training | costs accuracy (§7.1) and unexplained | dropped |

**Steps 5–7 are closed, not deferred.** In practice a model is built with about
5 to 20 experts, chosen from an idea of the spatial extent each should cover,
unless the goal is to tile a whole region. At that size there is nothing to
truncate: §4.2 finds nothing to gain below K ≈ 32 in space, and in the plane
K = 4–16 already needs 3–4 experts of the 4–16 available. The machinery would
only pay in a regime nobody works in — and §4.3 shows that regime is bounded
anyway, since experts cannot be subdivided below the kernel range.

What remains, and is inherent rather than a defect, is the cost in the number
of inducing points **per** expert: the O(m³) factorization and O(m²) covariance
that any variational GP carries.

---

## 9. Few big experts against many small ones (c10–c12)

The working Macpass setup is 5 experts of 100–200 inducing points. The question
that decides whether any of this is worth it: can the *total* number of
inducing points go up without training or prediction taking longer than that?

Macpass Zn, largest deposit, 8000 samples, 10 SVI epochs, 72 000-cell block
model. `master` is before steps 1–4, `claude` after.

| | K | m | total IP | train, 10 epochs | predict 72k | held-out RMSE |
|---|---:|---:|---:|---:|---:|---:|
| master | 5 | 150 | 750 | 11.5 s | 1.8 s | 5.137 |
| claude | 5 | 150 | 750 | 11.6 s | 1.9 s | 5.137 |
| claude | 40 | 40 | 1600 | **47.3 s** | **6.7 s** | 4.925 |

Two things to read off it. The identical RMSE at K = 5 confirms steps 1–4 are
exact — they change how the posterior is computed, not what it is — and at
5 experts they buy nothing, as expected. And **the goal is not met yet**: 2.1×
the inducing points costs 4.1× the training and 3.5× the prediction. (RMSE
improves, but at 10 epochs on a heavily skewed variable it is above the truth's
own standard deviation either way, so it is not evidence of much.)

### Why not — three explanations tested, all refuted

**It is not the FLOPs.** Cost should scale as `K·m³` for the refresh and
`K·n·m²` for the interpolation, both of which favour many small experts. The
measurement goes the other way, so the cost is not arithmetic.

**It is not the per-op overhead in `interpolate`** (c11). Stacking the experts
into one batched einsum is *slower* in every regime, because the `[K, n, m, D]`
difference tensor it needs costs more memory traffic than K small ones:

| n | K | m | looped | stacked |
|---:|---:|---:|---:|---:|
| 1000 | 5 | 150 | 5.0 ms | 6.9 ms |
| 1000 | 40 | 40 | 6.6 ms | 14.2 ms |
| 1000 | 128 | 40 | 23.6 ms | 45.7 ms |
| 20000 | 40 | 40 | 225.2 ms | 243.9 ms |

Stacking does collapse the *tracing* (1.2 s → 0.1 s at K = 128), which is a
real cost — the first training epoch takes 25.9 s at K = 40 against 4.7 s at
K = 5 — but not the execution.

Looped `interpolate` at K = 40, m = 40 is only 1.3× the cost of K = 5, m = 150,
nowhere near the 3.2× per step that was measured.

**It is not the refresh or the KL** (c12). Batching those helps a little
forward at small m (1.5–1.8× at K = 40–128) and hurts with the gradient, and
in absolute terms the whole refresh + KL is 2–7 ms against a 93–297 ms step —
a few per cent.

**It is not the eager parameter loop** (c13). `train_svi` runs
`for pr in self._all_parameters: pr.refresh()` after every step in eager
Python, and `BasicGP` creates three parameters per expert, so that loop grows
with K — but it is 1–2 % of a step:

| K | m | parameters | traced step | parameter loop | share |
|---:|---:|---:|---:|---:|---:|
| 5 | 150 | 20 | 87.9 ms | 1.1 ms | 1 % |
| 20 | 75 | 65 | 196.5 ms | 3.6 ms | 2 % |
| 40 | 40 | 125 | 282.0 ms | 6.8 ms | 2 % |

**Nor is it raw op count** (c14). The traced graph does grow at about 100 ops
per expert (591 → 3881 as K goes 5 → 40), but the forward-and-backward
arithmetic alone runs in 7.5 → 14.3 ms, where the real step goes 88 → 282 ms.

**Nor is it the optimizer, in itself** (c15). Adam keeps two slots per variable
and updates each one, and there are 20 variables at K = 5 against 125 at K = 40
— but *inside a traced step* that apply is 2 % of the time.

### It is the eager training step (c16)

**`tftools.training_step` (`tftools.py:68`) is not a `tf.function`.** The
`GradientTape`, `tape.gradient` and `apply_gradients` all run in eager Python.
The forward pass is graphed, since `_training_elbo` is decorated, but Adam then
issues its updates variable by variable, eagerly — and `BasicGP` creates three
parameters *per expert*.

Same model, same batch, same seed, both paths in one script:

| K | m | variables | `tftools.training_step` | one `tf.function` | gain | gap ÷ variables |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 150 | 20 | 88.9 ms | 58.8 ms | 1.5× | **1.50 ms** |
| 20 | 75 | 65 | 201.4 ms | 109.9 ms | 1.8× | **1.41 ms** |
| 40 | 40 | 125 | 277.7 ms | 97.5 ms | **2.8×** | **1.44 ms** |

The gap per variable is constant at ~1.45 ms across all three — an eager
per-variable cost, nothing to do with experts as such, except that three
parameters per expert makes it ~4.3 ms per expert per step.

What that does to the goal in §9: 40 experts × 40 inducing points, traced,
costs 97.5 ms per step against the current setup's eager 88.9 ms. **2.1× the
inducing points for 1.10× the training time.** For training, the constraint is
met by tracing the step, not by anything to do with experts.

Prediction does not go through `training_step`, so this leaves the 1.8 s →
6.7 s regression at K = 40 untouched. That is now the *only* case for sparse
activation.

This was looked at before and dismissed: `geoml-efficiency-findings` records
"graph-compiling the whole optimizer step into one `tf.function` gave no
measurable speedup", measured on a 16-variable single-expert model, where
16 × 1.45 ms is buried in a 45 ms step. The earlier measurement was not wrong,
only taken where the effect cannot appear.

**Step 5 is still not the next thing to do.** Tracing the training step is
independent of experts, helps every model, and is what makes many small experts
affordable at all.

### What it costs

Tracing forward, backward and optimizer as one graph makes that graph bigger to
build, and a short run does not repay it. Measured per test, master against the
branch:

| test | master | branch |
|---|---:|---:|
| `test_jura_mixed_categorical_and_elements` (5 iterations) | 30.27 s | **38.52 s** |
| `test_arctic_lake_composition` | 5.81 s | 5.05 s |
| `test_vgp.py` + `test_experts.py` together | 84 s | 51 s |

The one regression is the largest model trained for the fewest iterations — the
worst case for a one-off build cost. At 88.9 → 58.8 ms a step even at K = 5,
eight seconds of extra tracing is repaid in roughly 270 iterations, and real
runs are in the thousands. Worth knowing before timing anything short.

---

Found while testing, not touched: **`predict(newdata, n_sim=0)` raises.**
`_GPNode.predict` returns two values when `n_sim=0` and `predict_raw` unpacks
five (`models.py:944`). It fails the same way on master, so it is not something
the changes above introduced, but nothing in the suite covers it.

### 8.1 `geoml/inducing.py`

    from_kmeans(data, n)                  -> PointData
    from_grid(data, step)                 -> list[PointData]   (experts)
    combine(*point_data)                  -> PointData
    experts(data, n_experts, n_ip)        -> list[PointData]

`from_grid` lays a regular grid of expert blocks over the bounding box, each
extended by **one step** in every direction so that it overlaps its
neighbours. Their neighbours then follow from the grid rather than being
measured: the Moore neighbourhood, 8 in 2D and 26 in 3D, so the active set is
`k = 9` or `k = 27` including the expert itself.

The measurements in §4 back that choice up rather than contradicting it: the
weight at a point is carried by ~5 experts in the plane (against 9 available)
and ~23 in space (against 27). A fixed Moore neighbourhood is a safe
over-estimate in both, and it needs no search — which also makes the routing
a pure index computation.

`experts` is the k-means/balanced-clustering route for irregular data, after
`overlapping_clusters.py`, with the Mahalanobis overlap mapping onto **the
routing rule** — how many experts are active at a point — not onto which data
each expert is fitted against.

`BasicInput` derives the neighbour sets from the inducing point sets it is
handed, so sparsity works for hand-built expert lists too and `experts()` can
just return `list[PointData]`, which `BasicInput` already accepts.

`warping.py:644` already imports `sklearn.cluster.KMeans`, so no new dependency.

### 8.2 What changes in `latent.py`

1. The **derived** state (`alpha`, `cov_inv`, `cov_smooth_inv`, `chol_r`, the
   propagated inducing points) becomes stacked `[K, ...]` tensors at the end of
   `refresh`. The **trainable parameters** `alpha_white_i`/`delta_i`/`bias_i`
   stay per-expert, so `persistence.py` — which replays constructor arguments
   and restores parameters by name — is untouched and saved models keep
   loading.
2. Stacking needs equal m. `from_grid` and `experts` both produce equal sizes;
   an unequal set is padded to `m_max` with the cross-covariance masked (exact,
   §5.2), or falls back to the dense path when the spread is bad.
3. `interpolate` takes the active expert indices as a **tensor** and gathers —
   one trace (§5.3).
4. `predict` builds batches by primary expert instead of by consecutive index.
   `batch_index` already returns index arrays, and both
   `get_batched_coordinates(batch)` and `variables[v].update(batch, …)` accept
   them, so batches need not be contiguous. Groups below a size threshold are
   merged, or fall back to dense.

One routing decision serves the whole network: expert `e` means the same
inducing set at every node, and a deeper node's inducing points are the images
of the root's under a smooth map. For deep networks that is an extra
approximation — widen k if it shows.

### 8.3 Deferred

`influence` (`latent.py:868`) is computed by every node and discarded by both
`predict_raw` and `_log_lik`; line 879 also weights it by `explained_var` where
the three lines above weight by `weights`. Left alone, by decision.
