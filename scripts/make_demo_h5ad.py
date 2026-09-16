#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""デモ用の小さな h5ad を作る（MPS 動作検証用）。

GSE174367（ROSMAP 前頭前皮質 snRNA-seq）から細胞型を絞って取り出し、
Geneformer が要求する形式に整える:
  - var index を Ensembl ID に変換
  - obs に n_counts、filter_pass、celltype、batch（検体）を付与
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import anndata as ad
import numpy as np

GF_DIR = Path.home() / "workspace/research/Geneformer/geneformer_hf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-h5ad", default=str(Path.home() / "workspace/opencodews/data/GSE174367/GSE174367_snRNA.h5ad"))
    ap.add_argument("--out", default="/tmp/scpeft_demo.h5ad")
    ap.add_argument("--per-type", type=int, default=500)
    ap.add_argument("--types", default="MG,ODC,EX")
    args = ap.parse_args()

    print(f"  入力: {args.in_h5ad}")
    a = ad.read_h5ad(args.in_h5ad, backed="r")
    ct = a.obs["Cell.Type"].astype(str).to_numpy()
    rng = np.random.default_rng(0)
    keep = []
    for t in args.types.split(","):
        idx = np.where(ct == t)[0]
        if len(idx) == 0:
            print(f"    {t}: 0 細胞（飛ばします）")
            continue
        take = rng.choice(idx, size=min(args.per_type, len(idx)), replace=False)
        keep.extend(take.tolist())
        print(f"    {t}: {len(take)} 細胞")
    keep = np.sort(np.array(keep))
    sub = a[keep].to_memory()
    print(f"  抽出: {sub.n_obs:,} 細胞 × {sub.n_vars:,} 遺伝子")

    # 遺伝子名 → Ensembl
    name_id = pickle.load(open(GF_DIR / "geneformer/gene_name_id_dict_gc104M.pkl", "rb"))
    ensg = np.array([name_id.get(str(g), None) for g in sub.var_names], dtype=object)
    m = np.array([e is not None for e in ensg])
    sub = sub[:, m].copy()
    sub.var_names = [str(e) for e in ensg[m]]
    sub.var_names_make_unique()
    print(f"  Ensembl に変換: {sub.n_vars:,} 遺伝子")

    x = sub.X
    n_counts = np.asarray(x.sum(axis=1)).ravel() if hasattr(x, "sum") else None
    sub.obs["n_counts"] = n_counts.astype(float)
    sub.obs["filter_pass"] = 1
    sub.obs["celltype"] = sub.obs["Cell.Type"].astype(str)
    sub.obs["batch"] = sub.obs["SampleID"].astype(str)
    print(f"  celltype: {dict(sub.obs['celltype'].value_counts())}")
    print(f"  batch: {sub.obs['batch'].nunique()} 検体")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(out, compression="lzf")
    print(f"  保存: {out}  {out.stat().st_size / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
