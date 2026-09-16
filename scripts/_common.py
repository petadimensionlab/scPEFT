#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scPEFT の 5 タスクで共通に使う処理（Apple Silicon / MPS 対応）。

方針:
  - デバイスは `scpeft_mps.device` が唯一の決定点（MPS → CUDA → CPU の順）
  - トークン化は Geneformer 本体の `TranscriptomeTokenizer`（V2 / rank value 符号化）
  - 埋め込みと摂動は scPEFT の `geneformer_peft` を使う（MPS パッチ適用済み）
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    import psutil
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore[assignment]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
# `transformerslocal`（geneformer_peft/ 直下の最上位モジュール）は、リポジトリ直下の
# シンボリックリンク `transformerslocal -> geneformer_peft/transformerslocal` で解決する。
# geneformer_peft/ を sys.path に入れる方式は使わない:
# scPEFT 同梱の `geneformer_peft/geneformer/` がインストール済みの `geneformer`
# （TranscriptomeTokenizer / V2 対応）を隠してしまうため。
# リンクが無ければ作る（`patches/apply_vendor_patch.py` と同じ処理）。
_link = REPO / "transformerslocal"
_target = REPO / "geneformer_peft" / "transformerslocal"
if not _link.exists() and _target.exists():
    try:
        _link.symlink_to(Path("geneformer_peft") / "transformerslocal", target_is_directory=True)
    except OSError:
        pass

from scpeft_mps.device import DEVICE, NAME, describe, empty_cache  # noqa: E402

# --- 既定のパス（環境変数で上書きできる） -------------------------------------
GF_DIR = Path(os.environ.get("GENEFORMER_DIR",
                             str(Path.home() / "workspace/research/Geneformer/geneformer_hf")))
MODEL_NAME = os.environ.get("GENEFORMER_MODEL", "Geneformer-V2-316M")
MODEL_DIR = GF_DIR / "geneformer" / MODEL_NAME
TOKEN_DICT = GF_DIR / "geneformer" / "token_dictionary_gc104M.pkl"
NAME_ID_DICT = GF_DIR / "geneformer" / "gene_name_id_dict_gc104M.pkl"

# MPS の実測制約: 316M は batch × 18 heads × seq² が INT_MAX を超えるため forward batch を小さく
DEFAULT_BATCH = int(os.environ.get("SCPEFT_BATCH", "4" if "316M" in MODEL_NAME else "8"))


def log(msg: str) -> None:
    print(msg, flush=True)


def banner(title: str) -> None:
    log("")
    log(f"=== {title} ===")
    log(f"    {describe()}")


def out_dir(path: str | None, default_name: str) -> Path:
    p = Path(path) if path else (REPO / "outputs" / default_name)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    log(f"    保存: {path}")


# --- メモリ計測（タスクごとの必要量を記録する） -------------------------------
class MemoryTracker:
    """MPS の割当量とプロセスの RSS を一定間隔でサンプリングし、ピークを残す。

    MPS は CPU とメモリを共有するため「VRAM」はユニファイドメモリの割当量として見る。
    `torch.mps.driver_allocated_memory()` がドライバ側の実際の確保量、
    `current_allocated_memory()` がテンソルが占めている量。
    """

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.peak_mps = 0.0
        self.peak_current = 0.0
        self.peak_rss = 0.0
        self.samples = 0
        self._stop = False
        self._thread = None

    def _loop(self) -> None:
        import threading

        def sample() -> None:
            while not self._stop:
                try:
                    self.peak_mps = max(self.peak_mps, torch.mps.driver_allocated_memory())
                    self.peak_current = max(self.peak_current, torch.mps.current_allocated_memory())
                    self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
                except Exception:  # noqa: BLE001
                    pass
                self.samples += 1
                time.sleep(self.interval)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def start(self) -> "MemoryTracker":
        import torch  # noqa: F401  （MPS が使えない環境では値が 0 のまま）
        self._loop()
        return self

    def stop(self) -> None:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        return {
            "device": NAME,
            "peak_mps_driver_gib": round(self.peak_mps / 2**30, 3),
            "peak_mps_tensor_gib": round(self.peak_current / 2**30, 3),
            "peak_rss_gib": round(self.peak_rss / 2**30, 3),
            "samples": self.samples,
        }

    def log_summary(self) -> None:
        s = self.summary()
        log(f"  メモリのピーク: MPS {s['peak_mps_driver_gib']:.2f} GiB / "
            f"RSS {s['peak_rss_gib']:.2f} GiB")


def memory_guard(interval: float = 0.2):
    """`with memory_guard() as mem:` で使う（ピーク計測つきのコンテキスト）。"""
    try:
        import psutil  # noqa: F401
    except Exception:  # noqa: BLE001
        class _Noop:
            def __enter__(self):
                return MemoryTracker()
            def __exit__(self, *a):
                return False
        return _Noop()  # type: ignore[return-value]
    return _Started(MemoryTracker(interval))


class _Started:
    def __init__(self, tracker: MemoryTracker):
        self.tracker = tracker

    def __enter__(self) -> MemoryTracker:
        return self.tracker.start()

    def __exit__(self, *exc) -> bool:
        self.tracker.stop()
        self.tracker.log_summary()
        return False


def tracked(fn):
    """タスク関数をメモリ計測で包む。結果は outputs/<スクリプト名>_memory.json に残す。

    使い方: `@tracked` を `main()` に付けるだけ。
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        with memory_guard() as mem:
            code = fn(*args, **kwargs)
        rec = mem.summary()
        rec["script"] = fn.__module__
        rec["elapsed_sec"] = round(time.time() - t0, 1)
        dest = REPO / "outputs" / f"{fn.__module__}_memory.json"
        try:
            save_json(rec, dest)
        except Exception:  # noqa: BLE001
            pass
        return code

    return wrapper


# --- 1. トークン化 ------------------------------------------------------------
def tokenize_h5ad(h5ad: Path, out: Path, prefix: str = "scpeft",
                  label_key: str = "celltype", batch_key: str = "batch",
                  extra_keys: tuple[str, ...] = (), nproc: int = 4) -> Path:
    """h5ad を Geneformer V2 の dataset に変換する。

    入力 h5ad の要件（Geneformer 公式と同じ）:
      - var index が Ensembl ID
      - obs に `n_counts`
      - obs に `filter_pass`
    """
    import anndata as ad

    out.mkdir(parents=True, exist_ok=True)
    log(f"  [1] 入力: {h5ad}")
    a = ad.read_h5ad(h5ad)
    log(f"      {a.n_obs:,} 細胞 × {a.n_vars:,} 遺伝子")

    tok_dict = pickle.load(open(TOKEN_DICT, "rb"))
    keep = np.array([g in tok_dict for g in a.var_names.astype(str)])
    log(f"  [2] 語彙フィルタ: {int(keep.sum()):,} / {a.n_vars:,}")
    if not keep.all():
        a = a[:, keep].copy()
    a.var_names_make_unique()
    a.var.index.name = None
    a.obs.index.name = None
    if "n_counts" not in a.obs:
        raise SystemExit("h5ad に n_counts がありません（Geneformer の要件）")
    if "filter_pass" not in a.obs:
        a.obs["filter_pass"] = 1

    carry = [c for c in (label_key, batch_key, *extra_keys) if c in a.obs]
    in_dir = out / "_tokenizer_in"
    in_dir.mkdir(parents=True, exist_ok=True)
    tmp = in_dir / f"{prefix}.h5ad"
    a.write_h5ad(tmp, compression="lzf")
    del a

    from geneformer import TranscriptomeTokenizer

    tk = TranscriptomeTokenizer(custom_attr_name_dict={c: c for c in carry},
                                nproc=nproc, chunk_size=512,
                                model_version="V2", use_h5ad_index=True)
    tk.tokenize_data(data_directory=str(in_dir), output_directory=str(out),
                     output_prefix=prefix, file_format="h5ad")
    ds = out / f"{prefix}.dataset"
    if not ds.exists():
        raise SystemExit("トークン化に失敗しました")
    import datasets
    toks = datasets.load_from_disk(str(ds))
    lens = np.array(toks["length"])
    log(f"  [3] dataset: {len(toks):,} 細胞 / トークン長 中央値 {int(np.median(lens))} / 最大 {lens.max()}")
    return ds


# --- 2. 埋め込み抽出 ----------------------------------------------------------
def extract_embeddings(dataset: Path, out: Path, prefix: str,
                       model_type: str = "Pretrained", emb_mode: str = "cell",
                       batch: int | None = None, max_ncells: int | None = None,
                       cell_type: str | None = None,
                       num_classes: int = 0,
                       emb_label: list[str] | None = None) -> list[Path]:
    """scPEFT の EmbExtractor で埋め込みを抽出する（MPS 対応）。

    `model_directory` は「モデルのフォルダを入れた親ディレクトリ」を指す
    （Geneformer の慣習。例: `.../geneformer` の中に `Geneformer-V2-316M/` がある）。
    """
    from geneformer_peft.geneformer.emb_extractor import EmbExtractor

    batch = batch or DEFAULT_BATCH
    out.mkdir(parents=True, exist_ok=True)
    ex = EmbExtractor(
        model_type=model_type,
        num_classes=num_classes,
        emb_mode=emb_mode,
        cell_emb_style="mean_pool",
        filter_data={"celltype": [cell_type]} if cell_type else None,
        max_ncells=max_ncells if max_ncells is not None else 1000,
        emb_layer=-1,
        emb_label=emb_label or (["celltype", "batch"] if emb_mode == "cell" else None),
        forward_batch_size=batch,
        nproc=1,
        token_dictionary_file=TOKEN_DICT,
    )
    log(f"  EmbExtractor: model_type={model_type} emb_mode={emb_mode} batch={batch} "
        f"max_ncells={max_ncells}")
    made = ex.extract_embs(model_directory=str(MODEL_DIR.parent),
                           input_data_file=str(dataset),
                           output_directory=str(out),
                           output_prefix=prefix)
    empty_cache()
    if made is None:
        return sorted(out.glob(f"{prefix}*"))
    return [Path(p) for p in (made if isinstance(made, (list, tuple)) else [made])]


def load_embeddings(path: Path):
    """EmbedExtractor の出力（.csv か .npy）を読む。"""
    import pandas as pd
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        meta_cols = [c for c in ("cell_id", "celltype", "batch", "label") if c in df.columns]
        x = df.drop(columns=meta_cols).to_numpy(dtype=np.float32)
        return x, df[meta_cols] if meta_cols else None
    arr = np.load(path)
    return np.asarray(arr, dtype=np.float32), None


# --- 3. 評価の小道具 ---------------------------------------------------------
def classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    labels = sorted(set(map(str, y_true)))
    return {
        "n": int(len(y_true)),
        "n_classes": len(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def knn_indices(x: np.ndarray, k: int = 30, seed: int = 0) -> np.ndarray:
    """コサイン距離の k 近傍（バッチ混合の指標に使う）。"""
    from sklearn.neighbors import NearestNeighbors
    x = np.asarray(x, dtype=np.float32)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(x)), metric="cosine").fit(x)
    return nn.kneighbors(x, return_distance=False)[:, 1:]


def ilisi(x: np.ndarray, batches: np.ndarray, k: int = 30) -> float:
    """iLISI: 近傍に何種類のバッチが混ざっているか（1 = 混ざっていない、B = 完全に混合）。"""
    idx = knn_indices(x, k)
    b = np.asarray([str(v) for v in batches])
    values = [len(set(b[i])) for i in idx]
    return float(np.mean(values))


def asw_batch(x: np.ndarray, batches: np.ndarray) -> float:
    """バッチをラベルにした平均シルエット幅（0 に近いほど混合している）。"""
    from sklearn.metrics import silhouette_score
    b = np.asarray([str(v) for v in batches])
    if len(set(b)) < 2 or len(b) < 10:
        return float("nan")
    idx = np.random.default_rng(0).choice(len(x), size=min(5000, len(x)), replace=False)
    return float(silhouette_score(x[idx], b[idx], metric="cosine"))


def silhouette_labels(x: np.ndarray, labels) -> float:
    from sklearn.metrics import silhouette_score
    y = np.asarray([str(v) for v in labels])
    if len(set(y)) < 2 or len(y) < 10:
        return float("nan")
    idx = np.random.default_rng(0).choice(len(x), size=min(5000, len(x)), replace=False)
    return float(silhouette_score(x[idx], y[idx], metric="cosine"))
