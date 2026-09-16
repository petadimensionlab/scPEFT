"""scPEFT を Apple Silicon（MPS）で動かすための補助パッケージ。"""
from .device import DEVICE, NAME, IS_MPS, describe, empty_cache, pick_device, resolve_dtype, to_device

__all__ = ["DEVICE", "NAME", "IS_MPS", "describe", "empty_cache", "pick_device",
           "resolve_dtype", "to_device"]
