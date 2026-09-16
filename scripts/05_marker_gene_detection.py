#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""タスク5: marker gene detection（Geneformer / MPS 対応）。

2 つの見方を出す:
  A. 発現に基づくマーカー: 集団ごとに `rank_genes_groups`（Wilcoxon）で上位遺伝子を出す
  B. Geneformer の遺伝子埋め込みに基づくマーカー: 遺伝子埋め込みのコサイン類似で
     「同じ文脈で使われる遺伝子」を並べる（発現量に依存しない見方）

2 つが一致する遺伝子は、発現でも文脈でも特徴的なマーカーである可能性が高い。

使い方:
    python scripts/05_marker_gene_detection.py --h5ad data/demo.h5ad \
        --cluster-key leiden --top-n 20 --query-genes SNCA,TREM2 --out outputs/05_marker
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import (DEFAULT_BATCH, DEVICE, banner, ensembl_to_symbols, extract_embeddings,
                     log, out_dir, save_json, symbols_to_ensembl, tokenize_h5ad, tracked)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="marker gene detection（Geneformer）")
    p.add_argument("--h5ad", required=True, help="発現マトリクスのある h5ad")
    p.add_argument("--cluster-key", default="leiden", help="集団ラベルの列名")
    p.add_argument("--label-key", default="celltype", help="集団ラベルが無い場合の代用")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--query-genes", default=None,
                   help="遺伝子埋め込みの類似で並べたい遺伝子（カンマ区切り）")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--out", default=None)
    return p.parse_args()


def expression_markers(h5ad: Path, key: str, top_n: int, out: Path) -> dict:
    """A. 発現に基づくマーカー（Wilcoxon）。"""
    import scanpy as sc
    a = sc.read_h5ad(h5ad)
    if key not in a.obs.columns:
        raise SystemExit(f"{key} が obs にありません: {list(a.obs.columns)[:12]}")
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.tl.rank_genes_groups(a, key, method="wilcoxon", n_genes=top_n)
    res = {}
    names = a.uns["rank_genes_groups"]["names"]
    scores = a.uns["rank_genes_groups"]["scores"]
    for g in names.dtype.names:
        res[str(g)] = [{"gene": str(names[g][i]), "score": float(scores[g][i])}
                       for i in range(min(top_n, len(names[g])))]
    save_json(res, out / "markers_expression.json")
    log(f"  A. 発現マーカー: {len(res)} 集団 / 各 {top_n} 遺伝子")
    for g in list(res)[:3]:
        log(f"    {g}: {', '.join(x['gene'] for x in res[g][:6])} …")
    return res


def context_markers(dataset: Path, queries: list[str], top_n: int, out: Path,
                    batch: int) -> dict:
    """B. Geneformer の**入力埋め込み行列**を使った遺伝子の類似（発現量に依存しない見方）。

    scPEFT の EmbExtractor は `emb_mode="gene"` を「開発中」として拒否するため、
    モデルの `word_embeddings`（語彙 × 隠れ次元）を直接使う。
    コサイン類似が高い = 同じ文脈で使われる遺伝子。
    """
    import pickle
    import torch
    from geneformer_peft.geneformer.in_silico_perturber import load_model
    from scpeft_mps.device import DEVICE, empty_cache

    from _common import GF_DIR, MODEL_DIR, TOKEN_DICT

    tok_dict = pickle.load(open(TOKEN_DICT, "rb"))          # {Ensembl: token_id}
    ensg_to_symbol = ensembl_to_symbols(list(tok_dict.keys()))   # {Ensembl: 記号}
    # token_id → 遺伝子記号（トークン ID は連番なので辞書から逆引きする）
    tok_id_to_symbol = {tok_dict[e]: s for e, s in ensg_to_symbol.items() if e in tok_dict}
    log(f"  辞書: {len(tok_dict):,} 遺伝子 / 記号に戻せた数: {len(tok_id_to_symbol):,}")
    model = load_model("Pretrained", 0, str(MODEL_DIR)).to(DEVICE)
    model.eval()
    with torch.no_grad():
        w = model.bert.embeddings.word_embeddings.weight.detach().float().cpu().numpy()
    empty_cache()
    w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-9)
    log(f"  入力埋め込み行列: {w.shape}")

    queries = symbols_to_ensembl(queries)
    res = {}
    for q in queries:
        tid = tok_dict.get(q)
        if tid is None or tid >= w.shape[0]:
            log(f"    {q}: 語彙に存在しません")
            continue
        sim = w @ w[tid]
        order = np.argsort(-sim)
        rows = []
        for i in order:
            if i == tid:
                continue
            name = tok_id_to_symbol.get(int(i)) or f"token_{i}"
            rows.append({"gene": name, "cosine": float(sim[i])})
            if len(rows) >= top_n:
                break
        qsym = tok_id_to_symbol.get(tid, q)
        res[qsym] = rows
        log(f"    {qsym}: {', '.join(x['gene'] for x in rows[:6])} …")
    save_json(res, out / "markers_context.json")
    return res


@tracked
def main() -> int:
    args = parse()
    out = out_dir(args.out, "05_marker")
    banner("タスク5: marker gene detection")
    log(f"  デバイス: {DEVICE}  出力先: {out}")

    key = args.cluster_key
    try:
        expr = expression_markers(Path(args.h5ad), key, args.top_n, out)
    except SystemExit:
        log(f"  {key} が無いため {args.label_key} で代用します")
        expr = expression_markers(Path(args.h5ad), args.label_key, args.top_n, out)

    ctx = {}
    if args.query_genes:
        ds = tokenize_h5ad(Path(args.h5ad), out / "tokenized", prefix="marker",
                           label_key=args.label_key)
        log("  B. 遺伝子埋め込みのコサイン類似")
        ctx = context_markers(ds, args.query_genes.split(","), args.top_n, out, args.batch)

    both = {}
    for q, rows in ctx.items():
        ctx_genes = {r["gene"] for r in rows}
        expr_genes = {r["gene"] for rows2 in expr.values() for r in rows2}
        both[q] = sorted(ctx_genes & expr_genes)
        if both[q]:
            log(f"  両方で上位: {q} → {', '.join(both[q][:8])}")
    save_json({"expression": "markers_expression.json", "context": "markers_context.json",
               "agreement": both, "device": str(DEVICE)}, out / "summary.json")
    log("\n  完了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
