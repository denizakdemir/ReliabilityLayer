#!/usr/bin/env bash
# One-command rerun of every number and figure in arxiv_submission/main.tex.
# Non-destructive: writes only to $OUT (default repro_out/), never to the
# tracked experiments/results or experiments/figures.
#
# Prerequisites: python with numpy, scipy, scikit-learn, pandas, matplotlib,
# and defkit installed (pip install -e ../PACKAGES/defkit). Covertype is
# fetched by scikit-learn; 20 Newsgroups needs network on first use.
# Runtime: several hours (exact LP, bootstrap). Use  QUICK=1  for a smoke run.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-repro_out}"
PY="${PYTHON:-python}"
mkdir -p "$OUT/results" "$OUT/figures" "$OUT/logs"

if [ "${QUICK:-0}" = "1" ]; then
  T1="--mode fast"; GAP="--n-trials 3 --n-bins 12"
else
  T1="--mode final"; GAP="--n-trials 30 --n-bins 12 50"
fi

echo "== tests";            $PY -m pytest tests -q 2>&1 | tee "$OUT/logs/pytest.log" | tail -n 2
echo "== Theorem 1 (c=1)";  $PY experiments/validate_theorem1.py $T1 --quiet \
    --save-path "$OUT/figures/validate_theorem1.png" > "$OUT/logs/theorem1.log"
echo "== drift vs deficiency"; $PY experiments/validate_drift_deficiency.py \
    "$OUT/figures/validate_drift_deficiency.png" > "$OUT/logs/drift.log"
echo "== true-TV gap";      $PY experiments/validate_tv_gap.py $GAP --out "$OUT/results/tv_gap.csv" \
    > "$OUT/logs/tv_gap.log"
echo "== benchmarks";       $PY experiments/wilds_benchmark.py \
    --datasets camelyon_fallback amazon_fallback --allow-blockers \
    --solver exact --n-bins-exact 5 --n-bootstrap 20 --seed 42 \
    --output-dir "$OUT/results" > "$OUT/logs/benchmark.log"
echo "== benchmark figures"; $PY experiments/make_benchmark_figures.py \
    --results-dir "$OUT/results" --out-dir "$OUT/figures"
echo "Done. Outputs in $OUT/. Environment:"; $PY -m pip freeze | grep -iE "^(numpy|scipy|scikit-learn|pandas|matplotlib|defkit)=" | tee "$OUT/logs/versions.txt"
