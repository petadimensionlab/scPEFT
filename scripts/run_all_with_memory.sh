#!/usr/bin/env bash
# 5 タスクを順に走らせ、タスクごとのピークメモリを記録する。
# 出力: outputs/<スクリプト名>_memory.json と logs/task_NN.log
set -uo pipefail
H5AD="${1:-/tmp/scpeft_demo.h5ad}"
PY="${PYTHON:-$HOME/workspace/research/Geneformer/.venv/bin/python}"
cd "$(dirname "$0")"
OUT=../outputs
LOGS=../logs
mkdir -p "$OUT" "$LOGS"
DS="$OUT/01_identification/tokenized/ident.dataset"

echo "== 1. 細胞型の同定 =="
PYTHONPATH=.. "$PY" 01_cell_type_identification.py --h5ad "$H5AD" \
    --max-cells 1200 --batch 4 --out "$OUT/01_identification" > "$LOGS/task_01.log" 2>&1
echo "   exit=$? → $LOGS/task_01.log"

echo "== 2. バッチ補正 =="
PYTHONPATH=.. "$PY" 02_batch_correction.py --dataset "$DS" --max-cells 1200 \
    --batch-key batch --method harmony --out "$OUT/02_batch" > "$LOGS/task_02.log" 2>&1
echo "   exit=$? → $LOGS/task_02.log"

echo "== 3. 摂動 =="
PYTHONPATH=.. "$PY" 03_perturbation.py --dataset "$DS" --genes SNCA,TREM2 \
    --celltype MG --max-ncells 200 --batch 4 --out "$OUT/03_perturbation" > "$LOGS/task_03.log" 2>&1
echo "   exit=$? → $LOGS/task_03.log"

echo "== 4. 細胞集団の発見 =="
PYTHONPATH=.. "$PY" 04_cell_population_discovery.py --dataset "$DS" --max-cells 1200 \
    --label-key celltype --out "$OUT/04_population" > "$LOGS/task_04.log" 2>&1
echo "   exit=$? → $LOGS/task_04.log"

echo "== 5. マーカー遺伝子の検出 =="
PYTHONPATH=.. "$PY" 05_marker_gene_detection.py --h5ad "$H5AD" --label-key celltype \
    --query-genes SNCA,TREM2 --top-n 20 --batch 4 --out "$OUT/05_marker" > "$LOGS/task_05.log" 2>&1
echo "   exit=$? → $LOGS/task_05.log"

echo "== メモリの記録 =="
"$PY" - <<'PYEOF'
import json, glob, os
for f in sorted(glob.glob('../outputs/*_memory.json')):
    d = json.load(open(f))
    print(f"  {os.path.basename(f):46s} MPS {d['peak_mps_driver_gib']:6.2f} GiB  "
          f"RSS {d['peak_rss_gib']:6.2f} GiB  {d['elapsed_sec']:7.1f}s")
PYEOF
echo "完了"
