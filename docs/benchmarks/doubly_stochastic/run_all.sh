#!/bin/bash
# Every run behind the roadmap's "Want (1) measured 2026-10-16", in the order
# they were made, one process each so timings and peak memory are their own.
# Results and figures go to docs/benchmarks/figures/doubly_stochastic/. About
# four hours on four cores; the arms of a round are independent.
set -e
cd "$(dirname "$0")"
OUT=../figures/doubly_stochastic
mkdir -p "$OUT"

python check_identities.py                        # the coupling's gates

# round 1: the three trainings at 16 draws a step, then seeds and draw counts
mv -f "$OUT/results.jsonl" "$OUT/results.jsonl.old" 2>/dev/null || true
for spec in "vgp vgp" "dgp moment" "dgp ds-coupled" "dgp ds-indep" \
            "dgpc moment" "dgpc ds-coupled" "dgpc ds-indep"; do
  set -- $spec; python run_folded.py 2000 $1 $2 16
done
for seed in 11 12; do
  for spec in "dgp moment 16" "dgp ds-coupled 16" "dgpc moment 16" "dgpc ds-coupled 16"; do
    set -- $spec; python run_folded.py 2000 $1 $2 $3 $seed
  done
done
for S in 1 4 64; do python run_folded.py 2000 dgpc ds-coupled $S 2026; done
for spec in "dgpc 11" "dgpc 12" "dgp 2026" "dgp 11" "dgp 12"; do
  set -- $spec; python run_folded.py 2000 $1 ds-coupled 64 $2
done
for seed in 2026 11 12; do python run_folded.py 10000 dgpc moment 16 $seed; done
mv "$OUT/results.jsonl" "$OUT/results_round1.jsonl"

# cost: a training step and a prediction, one configuration a process
rm -f "$OUT/scale.jsonl"
for size in "200 100" "200 300" "800 100" "800 300"; do
  for arm in "moment 1" "ds 1" "ds 4" "ds 16" "ds 64"; do
    set -- $size $arm
    python scale.py $1 $2 $3 $4 | grep '^{' >> "$OUT/scale.jsonl" \
      || echo "{\"n_holes\": $1, \"m\": $2, \"arm\": \"$3\", \"s\": $4, \"failed\": \"out of memory\"}" >> "$OUT/scale.jsonl"
  done
done

# round 2: the decomposition, the second-moment grid, one draw a step
python run_folded.py 2000 vgp vgp 16 2026
for seed in 2026 11 12; do
  for net in dgp dgpc; do
    python run_folded.py 2000 $net moment 16 $seed
    python run_folded.py 2000 $net ds-coupled 64 $seed
    python run_folded.py 2000 $net quad3 16 $seed
    python run_folded.py 2000 $net quad5 16 $seed
    python run_folded.py 10000 $net ds-coupled 1 $seed
  done
done

# bias or basin, and the warm-up
python bound_accuracy.py dgpc 11
python basins.py 11
rm -f "$OUT/warmup.jsonl"
for net in dgpc dgp; do for seed in 2026 11 12; do
  for w in "200 64" "500 64" "500 1"; do set -- $w; python warmup.py $net $seed $1 $2; done
done; done

# experts
python run_experts.py gate 2026
rm -f "$OUT/experts.jsonl"
for seed in 2026 11 12; do for arm in moment draws64 warm; do
  python run_experts.py $arm $seed 3
done; done
for seed in 2026 11 12; do for arm in moment draws64; do
  python run_experts.py $arm $seed 1
done; done
for e in 3 1; do python experts_decomp.py 2026 $e; done

python figures.py 11
