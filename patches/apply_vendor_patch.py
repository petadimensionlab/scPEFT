#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""同梱の vendored transformers（`geneformer_peft/transformerslocal`）を使えるようにする。

このリポジトリは transformers 4.36 のフォークを zip で同梱しています。素のままだと 2 点で止まります。

1. **zip の展開先が 1 階層深い**
   `transformerslocal/transformerslocal/src/...` に展開されるが、コードは
   `transformerslocal/src/...` を期待する（`import transformerslocal.src...`）。

2. **`tokenizers` の版が厳しすぎる**
   vendored 側は `tokenizers>=0.14,<0.19` を要求するが、近年の venv は 0.20 台。
   scPEFT の Geneformer 経路は Geneformer 独自のトークン辞書を使い、HF の tokenizers を
   呼ばないため、この検査だけを緩めても実害がない（呼ぶのは HF トークナイザを使う経路のみ）。

使い方:
    python patches/apply_vendor_patch.py --check
    python patches/apply_vendor_patch.py
    python patches/apply_vendor_patch.py --revert
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "geneformer_peft" / "transformerslocal"
ZIP = VENDOR / "transformerslocal.zip"
VERSIONS = VENDOR / "src" / "transformers" / "utils" / "versions.py"
GUARD = re.compile(r'(require_version\([^)]*tokenizers[^)]*\))')


def flatten_zip() -> bool:
    """zip を展開し、1 階層深い構造を直す。"""
    if (VENDOR / "src").exists():
        return False
    if not ZIP.exists():
        raise SystemExit(f"zip がありません: {ZIP}")
    subprocess.run(["unzip", "-q", "-o", str(ZIP)], cwd=str(VENDOR), check=True)
    nested = VENDOR / "transformerslocal"
    if nested.exists():
        for f in nested.iterdir():
            shutil.move(str(f), str(VENDOR / f.name))
        nested.rmdir()
    return True


def relax_tokenizers_guard() -> int:
    """tokenizers の版検査を無効化する（条件式 1 行だけを置換する）。

    本体（raise ImportError …）は残すので、構文が崩れない。
    """
    if not VERSIONS.exists():
        raise SystemExit(f"vendored transformers が見つかりません: {VERSIONS}")
    src = VERSIONS.read_text()
    if "SCPEFT_TOKENIZERS_GUARD_RELAXED" in src:
        return 0
    old_cond = "if not ops[op](version.parse(got_ver), version.parse(want_ver)):"
    new_cond = "if False:  # SCPEFT_TOKENIZERS_GUARD_RELAXED（tokenizers の版差を許容）"
    if old_cond not in src:
        raise SystemExit("版検査の条件式が見つかりません。vendored 側の実装を確認してください")
    Path(str(VERSIONS) + ".orig").write_text(src)
    VERSIONS.write_text(src.replace(old_cond, new_cond, 1))
    return 1


def revert() -> None:
    for f in (VERSIONS, ):
        orig = Path(str(f) + ".orig")
        if orig.exists():
            shutil.copy2(orig, f)
            print(f"  戻しました: {f.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if args.revert:
        revert()
        return

    flat = (VENDOR / "src").exists()
    relaxed = VERSIONS.exists() and "SCPEFT_TOKENIZERS_GUARD_RELAXED" in VERSIONS.read_text()
    if args.check:
        print(f"  展開済み: {'はい' if flat else 'いいえ（要展開）'}")
        print(f"  tokenizers 検査の緩和: {'済み' if relaxed else '未適用'}")
        return

    if flatten_zip():
        print("  zip を展開し、階層を直しました")
    n = relax_tokenizers_guard()
    print(f"  tokenizers の版検査を緩和: {n} 箇所")
    print("  次に import を確認してください:")
    print("    PYTHONPATH=. python -c \"from geneformer_peft.geneformer.emb_extractor import EmbExtractor\"")


if __name__ == "__main__":
    main()
