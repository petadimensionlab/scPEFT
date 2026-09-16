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

from _common import (DEFAULT_BATCH, DEVICE, banner, extract_embeddings, log, out_dir,
                     save_json, tokenize_h5ad, tracked)


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
    """B. Geneformer の遺伝子埋め込みのコサイン類似によるマーカー。"""
    from _common import GF_DIR
    import pickle
    import torch
    from geneformer_peft.geneformer.emb_extractor import EmbExtractor
    from scpeft_mps.device import DEVICE, empty_cache

    emb_dir = out / "gene_emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    ex = EmbExtractor(model_type="Pretrained", emb_mode="gene", cell_emb_style="mean_pool",
                      max_ncells=200, emb_layer=-1, forward_batch_size=batch, nproc=1,
                      token_dictionary_file=GF_DIR / "geneformer" / "token_dictionary_gc104M.pkl")
    from _common import MODEL_DIR
    ex.extract_embs(model_directory=str(MODEL_DIR.parent), input_data_file=str(dataset),
                    output_directory=str(emb_dir), output_prefix="gene")
    npys = sorted(emb_dir.glob("*.npy"))
    if not npys:
        log("  遺伝子埋め込みが得られませんでした（B は省略）")
        return {}
    arr = np.load(npys[0], allow_pickle=True).item()
    name_id = pickle.load(open(GF_DIR / "geneformer" / "gene_name_id_dict_gc104M.pkl", "rb"))
    id_name = {v: k for k, v in name_id.items()}
    vecs, names = [], []
    for ensg, v in arr.items():
        vecs.append(np.asarray(v, dtype=np.float32).ravel())
        names.append(id_name.get(ensg, str(ensg)))
    m = np.vstack(vecs)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    idx = {n: i for i, n in enumerate(names)}
    res = {}
    for q in queries:
        if q not in idx:
            log(f"    {q}: 遺伝子埋め込みに存在しません")
            continue
        sim = m @ m[idx[q]]
        order = np.argsort(-sim)[1:top_n + 1]
        res[q] = [{"gene": names[i], "cosine": float(sim[i])} for i in order]
        log(f"    {q}: {', '.join(x['gene'] for x in res[q][:6])} …")
    empty_cache()
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
