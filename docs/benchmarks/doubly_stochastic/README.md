# Realizations through the tree: the doubly stochastic prototype

The measurements behind the roadmap's "Want (1) measured 2026-10-16" under
"Propagate individual realizations through the tree". A prototype outside
the package: nothing here is imported by geoML. `run_all.sh` remakes every
number and figure; results and figures are in
`docs/benchmarks/figures/doubly_stochastic/`.

| Script | What it does |
|---|---|
| `ds_prototype.py` | The leaf reading draws of its parent, its inducing inputs conditioned on each draw through the chain; the sampled bound, the second-moment bound by a Hermite grid, field realizations, a training loop |
| `ds_experts.py` | The same with several experts, each drawing its own parent expert's chain |
| `check_identities.py` | Gates: the conditioned kernel averaged over draws is the joint expected kernel; the mixture's mean is the moment model's |
| `run_folded.py` | One training arm on the folded section, scored on 200 new holes by moments and by the mixture, with the variance decomposition, field scores and maps |
| `scale.py` | Time and peak memory of a training step and a prediction at larger sizes |
| `bound_accuracy.py` | The Hermite grid's data term against 4000 draws |
| `basins.py` | Bias or basin: the 15x15 grid, and each bound continued from the other's model |
| `warmup.py` | A sampled warm-up, then the deterministic grid |
| `run_experts.py`, `experts_decomp.py` | Experts: the per-expert gate, the trainings, and the variance the moment path reports |
| `figures.py` | The figures, from the saved results and maps |

The folded section and its scoring are gate 2 of `../expected_kernel.py`,
imported from there.
