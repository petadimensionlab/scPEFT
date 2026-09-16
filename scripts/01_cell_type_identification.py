#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""タスク1: cell type identification（Geneformer / MPS 対応）。

2 通りの方法を出す:
  1. probe（既定）: 凍結埋め込み + ロジスティック回帰。速く、再現性が高い
  2. --peft: LoRA アダプタを追加学習（scPEFT の考え方）。少ないパラメータだけを更新する

使い方:
    python scripts/01_cell_type_identification.py \
        --h5ad data/demo.h5ad --label-key celltype --batch-key batch \
        --max-cells 5000 --out outputs/01_identification
    python scripts/01_cell_type_identification.py --h5ad ... --peft --peft-steps 200
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _common import (DEFAULT_BATCH, DEVICE, MODEL_DIR, banner, classification_metrics,
                     extract_embeddings, log, out_dir, save_json, tokenize_h5ad, tracked)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cell type identification（Geneformer）")
    p.add_argument("--h5ad", required=True, help="入力 h5ad（var は Ensembl ID、obs に n_counts）")
    p.add_argument("--label-key", default="celltype", help="正解ラベルの列名")
    p.add_argument("--batch-key", default="batch", help="バッチの列名（無くてもよい）")
    p.add_argument("--dataset", default=None, help="既存の .dataset を使う場合そのパス")
    p.add_argument("--max-cells", type=int, default=5000, help="埋め込みを取る細胞数の上限")
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="forward batch（MPS は小さく）")
    p.add_argument("--out", default=None)
    p.add_argument("--peft", action="store_true", help="LoRA で追加学習する")
    p.add_argument("--peft-steps", type=int, default=200)
    p.add_argument("--peft-lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def probe(x: np.ndarray, labels: np.ndarray, test_frac: float, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    x_tr, x_te, y_tr, y_te = train_test_split(x, labels, test_size=test_frac,
                                              random_state=seed, stratify=labels)
    sc = StandardScaler().fit(x_tr)
    clf = LogisticRegression(max_iter=3000, n_jobs=-1).fit(sc.transform(x_tr), y_tr)
    y_pred = clf.predict(sc.transform(x_te))
    m = classification_metrics(y_te, y_pred)
    m["method"] = "probe(frozen embeddings + logistic regression)"
    m["n_train"] = int(len(y_tr))
    m["n_test"] = int(len(y_te))
    return m


def peft_finetune(dataset: Path, labels: np.ndarray, n_classes: int, steps: int,
                  lr: float, batch: int, seed: int) -> tuple[np.ndarray, list[float]]:
    """LoRA アダプタを追加学習し、最後の埋め込みを返す（MPS 対応）。"""
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader

    import sys
    sys.path.insert(0, str(MODEL_DIR.parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from geneformer_peft.geneformer.collator_for_classification import (  # noqa: E402
        DataCollatorForCellClassification)
    from geneformer_peft.geneformer.in_silico_perturber import load_model  # noqa: E402
    from scpeft_mps.device import DEVICE, empty_cache  # noqa: E402

    torch.manual_seed(seed)
    tok = load_from_disk(str(dataset))
    tok = tok.map(lambda b: {"label": b}, batched=False) if False else tok
    keep = [i for i, lab in enumerate(tok["celltype"]) if lab in set(labels)]
    sub = tok.select(keep)
    model = load_model("CellClassifier", n_classes, str(MODEL_DIR)).to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                     target_modules=["query", "value"], task_type="FEATURE_EXTRACTION")
    model = get_peft_model(model, cfg)
    model.to(DEVICE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  LoRA 学習対象パラメータ: {n_train:,}")

    collator = DataCollatorForCellClassification(token_dictionary_file=None, model_input_size=4096)
    dl = DataLoader(sub, batch_size=batch, shuffle=True, collate_fn=collator)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    hist = []
    step = 0
    t0 = time.time()
    while step < steps:
        for batch_data in dl:
            out = model(input_ids=batch_data["input_ids"].to(DEVICE),
                        attention_mask=batch_data["attention_mask"].to(DEVICE),
                        output_hidden_states=True)
            h = out.hidden_states[-1][:, 0]           # <cls> 埋め込み
            logits = model.classifier(h) if hasattr(model, "classifier") else h.sum(-1, keepdim=True)
            loss = torch.nn.functional.cross_entropy(
                logits, batch_data["label"].to(DEVICE))
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            hist.append(float(loss.item()))
            if step % 50 == 0:
                log(f"    step {step}/{steps}  loss {np.mean(hist[-50:]):.4f}  "
                    f"{time.time() - t0:.0f}s")
            if step >= steps:
                break
    empty_cache()
    model.eval()
    embs = []
    with torch.no_grad():
        for batch_data in DataLoader(sub, batch_size=batch, collate_fn=collator):
            out = model(input_ids=batch_data["input_ids"].to(DEVICE),
                        attention_mask=batch_data["attention_mask"].to(DEVICE),
                        output_hidden_states=True)
            embs.append(out.hidden_states[-1][:, 0].float().cpu().numpy())
    return np.concatenate(embs), hist


@tracked
def main() -> int:
    args = parse()
    out = out_dir(args.out, "01_identification")
    banner("タスク1: cell type identification")
    log(f"  出力先: {out}")

    ds = Path(args.dataset) if args.dataset else tokenize_h5ad(
        Path(args.h5ad), out / "tokenized", prefix="ident",
        label_key=args.label_key, batch_key=args.batch_key)
    made = extract_embeddings(ds, out / "emb", "ident", model_type="Pretrained",
                              max_ncells=args.max_cells, batch=args.batch)
    csv = next((p for p in made if str(p).endswith(".csv")), None)
    if csv is None:
        raise SystemExit(f"埋め込み CSV が見つかりません: {made}")
    import pandas as pd
    df = pd.read_csv(csv)
    labels = df[args.label_key].astype(str).to_numpy() if args.label_key in df else None
    if labels is None:
        raise SystemExit(f"{args.label_key} 列が埋め込み CSV にありません: {list(df.columns)[:8]}")
    x = df.drop(columns=[c for c in ("cell_id", args.label_key, args.batch_key)
                         if c in df.columns]).to_numpy(dtype=np.float32)
    log(f"  埋め込み: {x.shape}  クラス数 {len(set(labels))}")

    metrics = {"device": str(DEVICE), "n_cells": int(len(x)),
               "n_classes": int(len(set(labels))), "embedding_dim": int(x.shape[1])}
    metrics["probe"] = probe(x, labels, args.test_frac, args.seed)
    log(f"  probe: accuracy {metrics['probe']['accuracy']:.4f} / "
        f"macro F1 {metrics['probe']['macro_f1']:.4f}")

    if args.peft:
        embs, hist = peft_finetune(ds, labels, len(set(labels)), args.peft_steps,
                                   args.peft_lr, args.batch, args.seed)
        metrics["peft"] = probe(embs, labels, args.test_frac, args.seed)
        metrics["peft"]["method"] = f"LoRA({args.peft_steps} steps)"
        metrics["peft"]["loss_first"] = float(np.mean(hist[:10]))
        metrics["peft"]["loss_last"] = float(np.mean(hist[-10:]))
        log(f"  LoRA: accuracy {metrics['peft']['accuracy']:.4f} / "
            f"macro F1 {metrics['peft']['macro_f1']:.4f} "
            f"(loss {metrics['peft']['loss_first']:.3f} → {metrics['peft']['loss_last']:.3f})")

    save_json(metrics, out / "metrics.json")
    _plot(x, labels, out)
    log("\n  完了")
    return 0


def _plot(x: np.ndarray, labels: np.ndarray, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap
    except Exception as e:  # noqa: BLE001
        log(f"  （作図は省略: {e}）")
        return
    emb = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(x)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    for lab in sorted(set(labels)):
        m = labels == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.7, label=lab)
    ax.set_title("Geneformer embeddings (cell type)")
    ax.legend(markerscale=3, fontsize=7, frameon=False)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "umap_celltype.png")
    log(f"    保存: {out / 'umap_celltype.png'}")


if __name__ == "__main__":
    raise SystemExit(main())
