#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""タスク3: perturbation（in silico 遺伝子削除 / Geneformer・MPS 対応）。

scPEFT の `InSilicoPerturber` を使い、指定した遺伝子を 1 つずつ削除したときの
埋め込みの変位（`Shift_to_goal_end`）を測る。正 = 削除で goal 側へ動く。

MPS の制約（実測）:
  - 316M は forward batch を小さくしないと落ちる（batch × 18 heads × seq² が INT_MAX 超）
  - 既定は env `SCPEFT_BATCH`（316M は 4、104M は 8）

使い方:
    python scripts/03_perturbation.py --dataset ... --genes SNCA,LRRK2,PINK1 \
        --celltype MG --state-key disease --start tg --goal WT --max-ncells 300 \
        --out outputs/03_perturbation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import DEFAULT_BATCH, DEVICE, MODEL_DIR, banner, log, out_dir, save_json, tokenize_h5ad, tracked

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="perturbation（in silico 遺伝子削除）")
    p.add_argument("--h5ad", default=None)
    p.add_argument("--dataset", default=None, help="既存 .dataset（トークン化済み）")
    p.add_argument("--genes", required=True, help="削除する遺伝子（カンマ区切り）")
    p.add_argument("--celltype", default=None, help="細胞プールで絞る（例 MG）")
    p.add_argument("--state-key", default="disease", help="状態の列名")
    p.add_argument("--start", default=None, help="start_state（例 tg）。省略時は全細胞")
    p.add_argument("--goal", default=None, help="goal_state（例 WT）")
    p.add_argument("--max-ncells", type=int, default=300)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="forward_batch_size")
    p.add_argument("--model-type", default="Pretrained", choices=["Pretrained", "CellClassifier"])
    p.add_argument("--num-classes", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


@tracked
def main() -> int:
    args = parse()
    out = out_dir(args.out, "03_perturbation")
    banner("タスク3: perturbation（削除シミュレーション）")
    log(f"  デバイス: {DEVICE}  出力先: {out}")

    if args.dataset:
        ds = Path(args.dataset)
    else:
        if not args.h5ad:
            raise SystemExit("--h5ad か --dataset のどちらかが必要です")
        ds = tokenize_h5ad(Path(args.h5ad), out / "tokenized", prefix="perturb",
                           label_key="celltype",
                           extra_keys=tuple([args.state_key]) if args.state_key else ())

    from geneformer_peft.geneformer.in_silico_perturber import InSilicoPerturber
    from geneformer_peft.geneformer.in_silico_perturber_stats import InSilicoPerturberStats

    states = None
    if args.start and args.goal:
        states = {args.state_key: ([args.start], [args.goal], [])}
    isp = InSilicoPerturber(
        perturb_type="delete",
        genes_to_perturb=args.genes.split(","),
        combos=0,
        anchor_gene=None,
        model_type=args.model_type,
        num_classes=args.num_classes,
        emb_mode="cell",
        cell_emb_style="mean_pool",
        filter_data={"celltype": [args.celltype]} if args.celltype else None,
        cell_states_to_model=states,
        max_ncells=args.max_ncells,
        emb_layer=-1,
        forward_batch_size=args.batch,
        nproc=1,
    )
    log(f"  削除対象: {args.genes}  max_ncells={args.max_ncells}  batch={args.batch}")
    isp.perturb_data(model_directory=str(MODEL_DIR.parent),
                     input_data_file=str(ds),
                     output_directory=str(out),
                     output_prefix="perturb")

    stats = InSilicoPerturberStats(mode="goal_state_shift",
                                   genes_perturbed=args.genes.split(","),
                                   combos=0,
                                   anchor_gene=None,
                                   cell_states_to_model=states)
    stats.get_stats(input_data_directory=str(out),
                    null_dist_data_directory=None,
                    output_directory=str(out / "stats"),
                    output_prefix="shift")

    logs = sorted((out / "stats").glob("*"))
    save_json({"device": str(DEVICE), "genes": args.genes.split(","),
               "celltype": args.celltype, "state_key": args.state_key,
               "start": args.start, "goal": args.goal,
               "max_ncells": args.max_ncells, "batch": args.batch,
               "model_type": args.model_type,
               "outputs": [str(p) for p in logs]}, out / "provenance.json")
    log(f"\n  出力: {out}")
    log("  正の Shift = 削除で goal 側へ移動（goal を健常にした場合は介入候補の材料）")
    log("  必ず null（無関係な遺伝子）と比較してから順位を主張すること。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
