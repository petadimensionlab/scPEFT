#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""タスク4: cell population discovery（Geneformer 埋め込みのクラスタリング / MPS 対応）。

埋め込み空間で近傍グラフを作り、Leiden で集団を分ける。
既存の細胞型ラベルがある場合は、集団との対応（混同行列）を出して「分かりやすい集団か」を確かめる。

使い方:
    python scripts/04_cell_population_discovery.py --dataset ... \
        --label-key celltype --resolution 1.0 --out outputs/04_population
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import (DEFAULT_BATCH, DEVICE, banner, extract_embeddings, log, out_dir,
                     save_json, silhouette_labels, tokenize_h5ad, tracked)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cell population discovery（Geneformer）")
    p.add_argument("--h5ad", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--label-key", default="celltype")
    p.add_argument("--max-cells", type=int, default=10000)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--out", default=None)
    return p.parse_args()


@tracked
def main() -> int:
    args = parse()
    out = out_dir(args.out, "04_population")
    banner("タスク4: cell population discovery")
    log(f"  デバイス: {DEVICE}  出力先: {out}")

    if args.dataset:
        ds = Path(args.dataset)
    else:
        if not args.h5ad:
            raise SystemExit("--h5ad か --dataset のどちらかが必要です")
        ds = tokenize_h5ad(Path(args.h5ad), out / "tokenized", prefix="pop",
                           label_key=args.label_key)

    made = extract_embeddings(ds, out / "emb", "pop", max_ncells=args.max_cells,
                              batch=args.batch)
    csv = next((p for p in made if str(p).endswith(".csv")), None)
    if csv is None:
        raise SystemExit(f"埋め込み CSV がありません: {made}")
    import pandas as pd
    import scanpy as sc
    df = pd.read_csv(csv)
    label = df[args.label_key].to_numpy() if args.label_key in df.columns else None
    x = df.drop(columns=[c for c in ("cell_id", args.label_key) if c in df.columns]
                ).to_numpy(dtype=np.float32)
    log(f"  埋め込み: {x.shape}")

    adata = sc.AnnData(x)
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep="X")
    sc.tl.leiden(adata, resolution=args.resolution, key_added="population",
                 flavor="igraph", n_iterations=2, directed=False)
    pops = adata.obs["population"].astype(str).to_numpy()
    sizes = pd.Series(pops).value_counts().sort_index()
    log(f"  集団数: {len(sizes)}")
    for k, v in sizes.items():
        log(f"    {k}: {v} 細胞 ({v / len(pops):.1%})")

    result = {"device": str(DEVICE), "n_cells": int(len(x)),
              "n_populations": int(len(sizes)), "resolution": args.resolution,
              "sizes": {str(k): int(v) for k, v in sizes.items()},
              "silhouette_population": silhouette_labels(x, pops)}
    if label is not None:
        result["silhouette_label"] = silhouette_labels(x, label)
        ct = pd.crosstab(pd.Series(label, name="celltype"), pd.Series(pops, name="population"))
        result["cross_tab"] = ct.to_dict()
        log("\n  既存ラベルとの対応（行=ラベル、列=集団）")
        log(str(ct))
        purity = (ct.max(axis=1) / ct.sum(axis=1)).mean()
        result["mean_purity"] = float(purity)
        log(f"\n  平均純度: {purity:.3f}（1.0 = 各ラベルが単一集団に収まる）")
    log(f"  シルエット（集団）: {result['silhouette_population']:.3f}")
    save_json(result, out / "metrics.json")
    _plot(adata, x, pops, label, out)
    log("\n  完了")
    return 0


def _plot(adata, x, pops, label, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scanpy as sc
    except Exception as e:  # noqa: BLE001
        log(f"  （作図は省略: {e}）")
        return
    sc.tl.umap(adata, random_state=0)
    um = adata.obsm["X_umap"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=140)
    for ax, lab, title in ((axes[0], pops, "Leiden 集団"),
                           (axes[1], label if label is not None else pops, "既存ラベル")):
        for v in sorted(set(map(str, lab))):
            m = np.asarray([str(z) for z in lab]) == v
            ax.scatter(um[m, 0], um[m, 1], s=3, alpha=0.7, label=str(v))
        ax.set_title(title)
        ax.legend(markerscale=3, fontsize=6, frameon=False, ncol=2)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "umap_populations.png")
    log(f"    保存: {out / 'umap_populations.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
