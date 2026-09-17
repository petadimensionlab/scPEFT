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

from _common import (DEFAULT_BATCH, DEVICE, MODEL_DIR, NAME_ID_DICT, TOKEN_DICT, banner, log, out_dir, save_json,
                     symbols_to_ensembl, tokenize_h5ad, tracked)

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
    p.add_argument("--stats-only", action="store_true",
                   help="摂動は再実行せず、既存の raw 出力から集計だけやり直す")
    p.add_argument("--per-gene", action="store_true",
                   help="複数遺伝子をまとめず 1 遺伝子ずつ摂動する（既定は同時摂動）")
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

    raw = args.genes.strip()
    if raw.lower() == "all":
        # 全遺伝子の rank shift（genes_to_perturb="all"）。群モードを使わない経路。
        genes = ["all"]
        log("  モード: 全遺伝子の rank shift（genes_to_perturb='all'）")
    else:
        # 辞書のキーは Ensembl ID。記号のまま渡すと実行が止まるため変換する。
        genes = symbols_to_ensembl(raw.split(","))
        if not genes:
            raise SystemExit("指定した遺伝子が辞書に見つかりません（Ensembl ID かを確認してください）")
    log(f"  削除対象（Ensembl）: {', '.join(genes)}")

    states = None
    if args.start and args.goal:
        states = {args.state_key: ([args.start], [args.goal], [])}

    # 事前確認: 遺伝子ごとの保有細胞数と、全遺伝子を共発現する細胞数を数える。
    # ISP は「指定した全遺伝子を持つ細胞」だけを摂動するので、共発現が 0 だと止まる。
    import datasets as _ds
    import pickle as _pk
    from _common import TOKEN_DICT as _TD

    allowed = _pk.load(open(_TD, "rb"))
    if genes == ["all"]:
        log("  検出確認: 全遺伝子の rank shift のためスキップ")
    else:
        toks = _ds.load_from_disk(str(ds))
        sub = toks.filter(lambda ex: ex.get("celltype") == args.celltype) if args.celltype else toks
        usable, hits = [], []
        for gene in genes:
            tid = allowed.get(gene)
            n_hit = sum(1 for row in sub["input_ids"] if tid in row)
            log(f"  検出確認: {gene} → {n_hit} / {len(sub)} 細胞（{n_hit / max(len(sub), 1):.1%}）")
            if n_hit == 0:
                log(f"  !! {gene} は対象細胞に存在しないため除外します（削除しても何も起きない）")
                continue
            usable.append(gene)
            hits.append(tid)
        if not usable:
            raise SystemExit("削除できる遺伝子がありません（発現する遺伝子を指定してください）")
        if len(usable) > 1:
            n_all = sum(1 for row in sub["input_ids"] if all(t in row for t in hits))
            log(f"  共発現: 指定した {len(usable)} 遺伝子をすべて持つ細胞 → {n_all} / {len(sub)}"
                f"（{n_all / max(len(sub), 1):.1%}）")
            if n_all == 0:
                raise SystemExit(
                    "指定した遺伝子をすべて持つ細胞が無いため、同時摂動できません。"
                    " --per-gene を付けると 1 遺伝子ずつ摂動できます。")
        genes = usable

    # 実行単位。Geneformer の list 指定は「まとめて同時に摂動」の意味なので、
    # 既定は 1 回の ISP で全遺伝子を渡す（複数遺伝子を一度に扱える）。
    # --per-gene を付けたときだけ 1 遺伝子ずつに分ける。
    units = [[g] for g in genes] if args.per_gene else [genes]
    log(f"  実行単位: {len(units)} 回（{'1 遺伝子ずつ' if args.per_gene else 'まとめて同時摂動'}）")

    made = []
    for unit in units:
        tag = "+".join("all" if g == "all" else g for g in unit)
        tagdir = out / tag
        tagdir.mkdir(parents=True, exist_ok=True)   # ISP は作成しない
        if not args.stats_only:
            log(f"  摂動: {tag}（同時に {len(unit)} 遺伝子）")
            isp = InSilicoPerturber(
                perturb_type="delete",
                genes_to_perturb=("all" if unit[0] == "all" else list(unit)),
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
                token_dictionary_file=TOKEN_DICT,   # 既定は別の辞書。gc104M を明示しないとトークンが一致しない
            )
            isp.perturb_data(model_directory=str(MODEL_DIR),
                             input_data_file=str(ds),
                             output_directory=str(tagdir) + "/",
                             output_prefix=tag)
        else:
            log(f"  摂動: {tag}（--stats-only: 既存の raw 出力を集計）")
        made.append(str(tagdir))

        # 集計モードは実行内容で決まる。
        #   goal_state_shift: 状態対があるときだけ（無いと None.keys() で落ちる）
        #   aggregate_data  : 遺伝子を明示した摂動（1 細胞に 1 摂動）をまとめる
        #   注意: aggregate_data は genes_perturbed="all" を拒否する
        if states is not None:
            stats_mode = "goal_state_shift"
        elif unit[0] != "all":
            stats_mode = "aggregate_data"
        else:
            raise SystemExit(
                "rank shift（--genes all）の集計には状態対が必要です。"
                " --start/--goal を指定してください。")
        (tagdir / "stats").mkdir(parents=True, exist_ok=True)   # 集計も作成しない
        stats = InSilicoPerturberStats(mode=stats_mode,
                                       genes_perturbed=("all" if unit[0] == "all" else list(unit)),
                                       combos=0,
                                       anchor_gene=None,
                                       cell_states_to_model=states,
                                       token_dictionary_file=TOKEN_DICT,
                                       gene_name_id_dictionary_file=NAME_ID_DICT)
        stats.get_stats(input_data_directory=str(tagdir),
                        null_dist_data_directory=None,
                        output_directory=str(tagdir / "stats"),
                        output_prefix="shift")

    logs = sorted((out).glob("*"))
    save_json({"device": str(DEVICE), "genes": args.genes.split(","),
               "ensemble_ids": genes,
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
