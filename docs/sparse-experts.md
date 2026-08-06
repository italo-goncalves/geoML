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
  guard. This only helps the *terminal* node: a deep network has several
  `BasicGP`s and the inner ones genuinely need their propagated inducing
  points, so the win shrinks with depth. Still free.
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

A synthetic cube — 8000 points filling 100³, the worst case, since experts have
to tile space in all three directions — agrees at the one size measured before
the environment failed (c6b, K = 16): 14.8 experts for 99.9 % of the weight,
k = 12 for 0.82 % error, k = 16 for 0.00 %. **K = 32 and 64 are still to run.**

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

| epoch | random | spatial, fixed order | spatial, shuffled order | spatial, KL rebalanced |
|---:|---:|---:|---:|---:|
| 1 | 225.65 | 246.81 | 246.50 | 246.30 |
| 6 | 174.49 | 211.86 | 212.41 | 212.15 |
| 12 | **165.32** | 201.97 | 202.89 | 202.42 |

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

- **Gradient diversity.** A fixed partition gives the optimiser the same 12
  directions for the whole run; a random partition draws a fresh one every
  epoch. Re-clustering with a new seed each epoch keeps batches spatially
  coherent while never repeating one. *(c8 was written for this and has not
  run yet.)*
- **The shared global parameters.** The transform range, the kernel range, the
  warping and the noise are estimated from a single region per step, and unlike
  the per-expert parameters they have no locality to fall back on.

Until one of these works, **training stays dense**. Prediction sparsity is a
pure approximation that leaves the fitted model untouched; training sparsity
changes the objective, and as measured it also costs accuracy.

---

## 8. Plan

Steps 1–4 are exact: same numbers, faster. Steps 5–6 trade a measured
approximation for speed. Step 7 is blocked on §7.2.

| # | Step | Effect | Risk |
|---|---|---|---|
| 1 | `if self.children:` guard on the propagation loop | 10.7× at K = 32 on a single-layer net, less with depth (c9 will quantify) | one line, output identical |
| 2 | Tests for K > 1 — build, train, predict, save, load | guards everything below | none |
| 3 | `geoml/inducing.py`: `from_kmeans`, `from_grid`, `combine`, `experts` | usability; produces the neighbour structure the rest needs | new module |
| 4 | Refresh in graph mode, state returned and assigned | removes the remaining refresh cost | touches `predict`'s caching |
| 5 | Stack the derived state; gather k experts in `interpolate`; group prediction batches by expert | 5.8–11.6× on prediction | approximation, bounded in §4 |
| 6 | Neighbour-restricted propagation in `refresh` | only matters for multi-layer networks | approximation |
| 7 | Sparse activation during training | blocked: costs accuracy (§7.1) | changes the objective |

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
