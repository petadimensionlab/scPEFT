#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scPEFT のデバイス解決（Apple Silicon の MPS を含む）。

scPEFT 本体は `cuda` を直書きしています。このモジュールが唯一のデバイス決定点になり、
`patches/apply_mps_patch.py` が直書き箇所をここ経由に置き換えます。

決定順序:
   1. 環境変数 `SCPEFT_DEVICE`（`mps` / `cuda` / `cpu`）
   2. Apple Silicon の MPS（利用可能なとき）
   3. CUDA
   4. CPU

MPS の注意点:
   - `torch.float64` は未対応。倍精度が必要な計算は CPU に移す
   - `torch.cuda.empty_cache()` は存在しない → `torch.mps.empty_cache()`
   - bfloat16 は macOS 14 以降。不安定なときは `SCPEFT_DTYPE=float32`
   - 大きな系列長では attention テンソルが INT_MAX を超える（batch を下げる）
"""
from __future__ import annotations

import os

import torch

__all__ = ["DEVICE", "NAME", "IS_MPS", "pick_device", "empty_cache", "to_device",
           "resolve_dtype", "describe"]


def pick_device(prefer: str | None = None) -> torch.device:
    """利用できる最良のデバイスを返す。"""
    want = (prefer or os.environ.get("SCPEFT_DEVICE") or "").strip().lower()
    if want:
        if want == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("SCPEFT_DEVICE=mps ですが MPS が使えません")
        if want == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SCPEFT_DEVICE=cuda ですが CUDA が使えません")
        return torch.device(want)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE: torch.device = pick_device()
NAME: str = DEVICE.type
IS_MPS: bool = NAME == "mps"


def empty_cache() -> None:
    """バックエンドに応じたキャッシュ解放。"""
    if IS_MPS:
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_device(obj, device: torch.device | str | None = None):
    """テンソルやモデルをデバイスへ移す（MPS では float64 を float32 に落とす）。"""
    dev = torch.device(device) if device is not None else DEVICE
    if isinstance(obj, torch.Tensor) and obj.dtype == torch.float64 and dev.type == "mps":
        obj = obj.to(torch.float32)
    return obj.to(dev)


def resolve_dtype(name: str | None = None) -> torch.dtype:
    """計算精度を解決する。既定は float32（MPS で最も安定）。"""
    key = (name or os.environ.get("SCPEFT_DTYPE") or "float32").strip().lower()
    return {"float32": torch.float32, "fp32": torch.float32,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16}.get(key, torch.float32)


def describe() -> str:
    if IS_MPS:
        return f"device=mps（Apple Silicon / torch {torch.__version__}）"
    if NAME == "cuda":
        return f"device=cuda（{torch.cuda.get_device_name(0)}）"
    return f"device=cpu（torch {torch.__version__}）"
