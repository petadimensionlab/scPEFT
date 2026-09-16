#!/usr/bin/env bash
# 各タスクの実測所要時間を、GPU を奪い合わないよう 1 本ずつ測る。
set -u
cd "$(dirname "$0")"
PY="$HOME/workspace/research/Geneformer/.venv/bin/python"
DS=../outputs/01_identification/tokenized/ident.dataset
OUT=../outputs/timing.txt
: > "$OUT"

# 先に走っている摂動（rank shift）の終了を待つ。GPU を共有すると計測が濁るため。
while pgrep -f '03_perturbation.py' >/dev/null 2>&1; do sleep 20; done

run() {
  local name="$1"; shift
  local t0 t1
  t0=$(date +%s)
  PYTHONPATH=.. "$PY" "$@" > "../logs/timing_${name}.log" 2>&1
  local rc=$?
  t1=$(date +%s)
  printf '%-34s %6d 秒  exit=%d\n' "$name" "$((t1 - t0))" "$rc" | tee -a "$OUT"
}

run 01_identification \
  "$PWD/01_cell_type_identification.py" --h5ad /tmp/scpeft_demo.h5ad --label-key celltype \
  --max-cells 1200 --batch 4 --out ../outputs/01_identification
run 02_batch_correction \
  "$PWD/02_batch_correction.py" --dataset "$DS" --max-cells 1200 --batch-key batch \
  --method harmony --out ../outputs/02_batch
run 04_cell_population_discovery \
  "$PWD/04_cell_population_discovery.py" --dataset "$DS" --max-cells 1200 \
  --label-key celltype --out ../outputs/04_population
run 05_marker_gene_detection \
  "$PWD/05_marker_gene_detection.py" --h5ad /tmp/scpeft_demo.h5ad --label-key celltype \
  --query-genes SNCA,TREM2 --top-n 20 --batch 4 --out ../outputs/05_marker

echo TIMING_DONE
