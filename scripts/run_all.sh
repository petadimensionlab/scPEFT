#!/usr/bin/env bash
# 5 タスクを順に実行する（Apple Silicon / MPS 想定）。
# 使い方:  bash scripts/run_all.sh /tmp/scpeft_demo.h5ad
set -euo pipefail
H5AD="${1:?使い方: bash scripts/run_all.sh <h5ad>}"
PY="${PYTHON:-$HOME/workspace/research/Geneformer/.venv/bin/python}"
cd "$(dirname "$0")"
OUT=../outputs
DS="$OUT/01_identification/tokenized/ident.dataset"

echo "== 0. 準備 =="
"$PY" -c "from scpeft_mps.device import describe; print(' ', describe())" 2>/dev/null || \
  PYTHONPATH=.. "$PY" -c "from scpeft_mps.device import describe; print(' ', describe())"

echo "== 1. 細胞型の同定 =="
"$PY" 01_cell_type_identification.py --h5ad "$H5AD" --max-cells 5000 --out "$OUT/01_identification"

echo "== 2. バッチ補正 =="
"$PY" 02_batch_correction.py --dataset "$DS" --method harmony --out "$OUT/02_batch"

echo "== 3. 摂動 =="
"$PY" 03_perturbation.py --dataset "$DS" --genes SNCA,LRRK2,PINK1 \
    --max-ncells 300 --out "$OUT/03_perturbation"

echo "== 4. 細胞集団の発見 =="
"$PY" 04_cell_population_discovery.py --dataset "$DS" --out "$OUT/04_population"

echo "== 5. マーカー遺伝子の検出 =="
"$PY" 05_marker_gene_detection.py --h5ad "$H5AD" --query-genes SNCA,TREM2 \
    --out "$OUT/05_marker"

echo "完了: $OUT"
