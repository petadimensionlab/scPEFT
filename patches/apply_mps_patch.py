#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scPEFT の `cuda` 直書きを MPS でも動くように置換する。

置換する対象（Geneformer 系のモジュール）:
    geneformer_peft/geneformer/*.py

置換規則:
    .to("cuda")            → .to(_DEVICE)
    .cuda()                → .to(_DEVICE)
    device="cuda"          → device=_DEVICE
    torch.device("cuda")   → _DEVICE
    torch.cuda.empty_cache() → _empty_cache()

挿入する import:
    from scpeft_mps.device import DEVICE as _DEVICE, empty_cache as _empty_cache

使い方:
    python patches/apply_mps_patch.py --check     # 対象と件数を表示（書き換えない）
    python patches/apply_mps_patch.py             # 適用（バックアップ .orig を作る）
    python patches/apply_mps_patch.py --revert    # 元に戻す
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = sorted((ROOT / "geneformer_peft" / "geneformer").glob("*.py"))

IMPORT_LINE = ("from scpeft_mps.device import DEVICE as _DEVICE, "
               "empty_cache as _empty_cache  # MPS/CUDA/CPU を自動で選ぶ")
MARK = "# MPS/CUDA/CPU を自動で選ぶ"

RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\.to\(\s*["\']cuda(?::\d+)?["\']\s*\)'), ".to(_DEVICE)"),
    (re.compile(r'\.cuda\(\s*\)'), ".to(_DEVICE)"),
    (re.compile(r'torch\.device\(\s*["\']cuda(?::\d+)?["\']\s*\)'), "_DEVICE"),
    (re.compile(r'torch\.cuda\.empty_cache\(\s*\)'), "_empty_cache()"),
    (re.compile(r'device\s*=\s*["\']cuda(?::\d+)?["\']'), "device=_DEVICE"),
]


def apply_import(src: str) -> str:
    """import 群の直後に 1 行だけ挿入する（冪等）。

    複数行にまたがる import（バックスラッシュ継続、括弧）の途中に入れると壊れるため、
    「完結した import 文」の最後を見つけてから挿す。
    """
    if MARK in src:
        return src
    lines = src.split("\n")
    last, depth = -1, 0
    for i, ln in enumerate(lines[:120]):
        s = ln.strip()
        started = depth == 0 and s.startswith(("import ", "from "))
        depth += ln.count("(") - ln.count(")")
        cont = ln.rstrip().endswith("\\") or depth > 0
        if started and not cont and depth == 0:
            last = i
        if last >= 0 and i > last + 3 and depth == 0:
            break
    if last < 0:
        return IMPORT_LINE + "\n" + src
    lines.insert(last + 1, IMPORT_LINE)
    return "\n".join(lines)


def count_hits(src: str) -> int:
    return sum(len(p.findall(src)) for p, _ in RULES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="対象と件数の表示のみ")
    ap.add_argument("--revert", action="store_true", help=".orig から戻す")
    args = ap.parse_args()

    total = 0
    for f in TARGETS:
        orig = f.with_suffix(f.suffix + ".orig")
        if args.revert:
            if orig.exists():
                shutil.copy2(orig, f)
                print(f"  戻しました: {f.relative_to(ROOT)}")
            continue
        src = f.read_text()
        hits = count_hits(src)
        if args.check:
            if hits:
                print(f"  {f.relative_to(ROOT)}: {hits} 件")
            total += hits
            continue
        if hits == 0 and MARK not in src:
            continue
        if not orig.exists():
            shutil.copy2(f, orig)
        elif MARK in src:            # 既に適用済みのファイルは .orig を保つ
            pass
        new = src
        for pat, rep in RULES:
            new = pat.sub(rep, new)
        new = apply_import(new)
        if new != src:
            f.write_text(new)
            print(f"  適用: {f.relative_to(ROOT)}（{hits} 件）")
            total += hits

    if args.check:
        print(f"\n  置換対象: 合計 {total} 件 / {len(TARGETS)} ファイル")
    elif not args.revert:
        print(f"\n  置換: 合計 {total} 件")
        print("  戻すときは --revert（.orig から復元）")


if __name__ == "__main__":
    main()
