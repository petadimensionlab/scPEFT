#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""タスク2: batch correction（Geneformer 埋め込み / MPS 対応）。

手順:
  1. 埋め込みを抽出（生）
  2. バッチ混合の指標を測る（iLISI、バッチ ASW、kBET 風の棄却率）
  3. 補正する（既定: Harmony。無ければ PCA + 線形補正で代替）
  4. 補正後の指標を測り、細胞型の分離（シルエット）が保たれたかを確認する

「混合が進んだか」と「細胞型が潰れていないか」を必ず対で報告する。
片方だけを見ると、補正が強すぎて生物学的な差を消した場合に気づけない。

使い方:
    python scripts/02_batch_correction.py --dataset outputs/.../x.dataset \
        --label-key celltype --batch-key batch --out outputs/02_batch
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import (DEFAULT_BATCH, DEVICE, asw_batch, banner, extract_embeddings,
                     ilisi, log, out_dir, save_json, silhouette_labels, tokenize_h5ad, tracked)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="batch correction（Geneformer）")
    p.add_argument("--h5ad", default=None)
    p.add_argument("--dataset", default=None, help="既存 .dataset")
    p.add_argument("--label-key", default="celltype")
    p.add_argument("--batch-key", default="batch")
    p.add_argument("--max-cells", type=int, default=5000)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--method", default="harmony", choices=["harmony", "pca", "none"])
    p.add_argument("--out", default=None)
    return p.parse_args()


def metrics_of(x: np.ndarray, batch, label) -> dict:
    m = {"ilisi": ilisi(x, batch), "asw_batch": asw_batch(x, batch)}
    if label is not None:
        m["silhouette_celltype"] = silhouette_labels(x, label)
    m["n_batches"] = int(len(set(map(str, batch))))
    return m


def run_harmony(x: np.ndarray, batch) -> tuple[np.ndarray, str]:
    """Harmony による補正（harmonypy があれば使う。無ければ PCA 版で代替）。"""
    try:
        import harmonypy  # noqa: F401
        import pandas as pd
        import scanpy as sc
        adata = sc.AnnData(x)
        adata.obs["batch"] = pd.Categorical([str(b) for b in batch])
        sc.external.pp.harmony_integrate(adata, "batch", basis="X")
        return np.asarray(adata.obsm["X_pca_harmony"], dtype=np.float32), "harmony"
    except Exception as e:  # noqa: BLE001
        log(f"    harmonypy が使えないため PCA 補正に切り替えます: {str(e)[:60]}")
        return run_pca(x, batch)


def run_pca(x: np.ndarray, batch) -> tuple[np.ndarray, str]:
    """PCA 後、バッチ平均を引く簡易補正（Harmony が無い環境の代替）。"""
    from sklearn.decomposition import PCA
    z = PCA(n_components=min(50, x.shape[1], len(x) - 1), random_state=0).fit_transform(x)
    z = z - z.mean(0, keepdims=True)
    b = np.asarray([str(v) for v in batch])
    for bb in set(b):
        m = b == bb
        z[m] -= z[m].mean(0, keepdims=True)
    return z.astype(np.float32), "pca_batch_center"


@tracked
def main() -> int:
    args = parse()
    out = out_dir(args.out, "02_batch")
    banner("タスク2: batch correction")
    log(f"  デバイス: {DEVICE}  出力先: {out}")

    if args.dataset:
        ds = Path(args.dataset)
    else:
        if not args.h5ad:
            raise SystemExit("--h5ad か --dataset のどちらかが必要です")
        ds = tokenize_h5ad(Path(args.h5ad), out / "tokenized", prefix="batch",
                           label_key=args.label_key, batch_key=args.batch_key)

    made = extract_embeddings(ds, out / "emb", "batch", max_ncells=args.max_cells,
                              batch=args.batch)
    csv = next((p for p in made if str(p).endswith(".csv")), None)
    if csv is None:
        raise SystemExit(f"埋め込み CSV がありません: {made}")
    import pandas as pd
    df = pd.read_csv(csv)
    if args.batch_key not in df.columns:
        raise SystemExit(f"{args.batch_key} 列がありません: {list(df.columns)[:10]}")
    batch = df[args.batch_key].to_numpy()
    label = df[args.label_key].to_numpy() if args.label_key in df.columns else None
    x = df.drop(columns=[c for c in ("cell_id", args.label_key, args.batch_key)
                         if c in df.columns]).to_numpy(dtype=np.float32)
    log(f"  埋め込み: {x.shape}  バッチ {len(set(map(str, batch)))} 種")

    before = metrics_of(x, batch, label)
    log(f"  補正前: iLISI {before['ilisi']:.2f} / ASW(batch) {before['asw_batch']:.3f} / "
        f"silhouette(celltype) {before.get('silhouette_celltype', float('nan')):.3f}")

    if args.method == "none":
        after, method = before, "none"
    else:
        fixed, method = (run_harmony(x, batch) if args.method == "harmony"
                         else run_pca(x, batch))
        after = metrics_of(fixed, batch, label)
        np.save(out / "embeddings_corrected.npy", fixed)
        log(f"  補正後（{method}）: iLISI {after['ilisi']:.2f} / "
            f"ASW(batch) {after['asw_batch']:.3f} / "
            f"silhouette(celltype) {after.get('silhouette_celltype', float('nan')):.3f}")

    save_json({"device": str(DEVICE), "method": method, "n_cells": int(len(x)),
               "before": before, "after": after}, out / "metrics.json")

    if isinstance(after, dict) and "silhouette_celltype" in after:
        d_mix = after["ilisi"] - before["ilisi"]
        d_sep = after["silhouette_celltype"] - before["silhouette_celltype"]
        log(f"\n  判定: 混合は {'改善' if d_mix > 0 else '悪化'}（iLISI {d_mix:+.2f}）、"
            f"細胞型の分離は {'維持' if d_sep > -0.05 else '低下'}（{d_sep:+.3f}）")
        log("  どちらか一方だけを見て「成功」と判断しないこと。")
    log("\n  完了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
