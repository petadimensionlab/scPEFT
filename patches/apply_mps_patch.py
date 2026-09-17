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
    # Jupyter 専用の tqdm は CLI で ImportError（IProgress）になるため auto に寄せる
    (re.compile(r'from tqdm\.notebook import'), "from tqdm.auto import"),
    # 1) まず cuda 直書きを _DEVICE に寄せる
    (re.compile(r'\.to\(\s*["\']cuda(?::\d+)?["\']\s*\)'), ".to(_DEVICE)"),
    (re.compile(r'\.cuda\(\s*\)'), ".to(_DEVICE)"),
    (re.compile(r'torch\.device\(\s*["\']cuda(?::\d+)?["\']\s*\)'), "_DEVICE"),
    (re.compile(r'torch\.cuda\.empty_cache\(\s*\)'), "_empty_cache()"),
    (re.compile(r'device\s*=\s*["\']cuda(?::\d+)?["\']'), "device=_DEVICE"),
    # 2) その後に datasets 4.x 対応（Column/list は `.to()` を持たない）を当てる。
    #    この順序が逆だと置換対象がまだ `.to("cuda")` のままで一致しない。
    (re.compile(r'\binput_data_minibatch\.to\(_DEVICE\)'), "_to_device_tensor(input_data_minibatch)"),
    (re.compile(r'\boriginal_input_data_minibatch\.to\(_DEVICE\)'),
     "_to_device_tensor(original_input_data_minibatch)"),
    (re.compile(r'\binput_data\.to\(_DEVICE\)'), "_to_device_tensor(input_data)"),
    # 3) 上流の欠陥: 比較バッチと摂動バッチの系列長がトークン 1 個ずれる場合がある
    #    （削除で短くなった摂動側と、パディングで揃えられた比較側）。コサイン類似の前に
    #    共通の短い方へ切り詰める。位置は発現量順の並びなので余った末尾に対応物はない。
    (re.compile(r'cos_sims \+= \[cos\(minibatch_emb, minibatch_comparison\)\.to\("cpu"\)\]'),
     'cos_sims += [cos(*_align_token_len(minibatch_emb, minibatch_comparison)).to("cpu")]'),
    # 4) 上流の欠陥: `torch.squeeze` が 1 細胞ミニバッチのバッチ次元を潰し、
    #    トークン数を細胞数として数えてインデックスが範囲外になる。バッチ次元は残す。
    #    対象はミニバッチを扱う 2 箇所だけ（forward_pass_single_cell は 1 細胞専用で
    #    2 次元を期待するため触らない）。
    (re.compile(r'minibatch_emb = torch\.squeeze\(outputs\.hidden_states\[layer_to_quant\]\)'),
     'minibatch_emb = _squeeze_keep_batch(outputs.hidden_states[layer_to_quant])'),
    (re.compile(r'original_minibatch_emb = torch\.squeeze\(original_outputs\.hidden_states\[layer_to_quant\]\)'),
     'original_minibatch_emb = _squeeze_keep_batch(original_outputs.hidden_states[layer_to_quant])'),
    # 5) 上流の欠陥: 群摂動の比較バッチに **全細胞分** のインデックスを渡している。
    #    ループはミニバッチの行を数えるので、別の細胞の位置で切って長さが合わなくなる
    #    （`Sizes of tensors must match ... Expected size 3851 but got size 3161`）。
    #    ミニバッチに対応する区間だけを渡す。
    (re.compile(r'minibatch_comparison = make_comparison_batch\(original_minibatch_emb,\n'
                r'(\s+)indices_to_perturb,\n'),
     r'minibatch_comparison = make_comparison_batch(original_minibatch_emb,\n'
     r'\1indices_to_perturb[i:max_range],\n'),
    # 6) 同じ datasets 4.x の Column 問題が rank shift 側にもある（torch.squeeze に渡す前に
    #    テンソル化する）。
    (re.compile(r'gene_list = torch\.squeeze\(example_cell\["input_ids"\]\)'),
     'gene_list = _tensor_from(example_cell["input_ids"]).squeeze()'),
    (re.compile(r'j_index = torch\.squeeze\(j_index\)'), 'j_index = _tensor_from(j_index).squeeze()'),
    # 7) datasets 4.x の Column は `*` で繰り返せない（rank shift の系列生成）。
    (re.compile(r'"input_ids": example_cell\["input_ids"\] \* length,'),
     '"input_ids": list(example_cell["input_ids"]) * length,'),
    # 8) 上流の欠陥: ミニバッチごとにパディング長が違うコサイン類似をそのまま連結する。
    #    トークン方向を最短に揃えてから連結する（位置は発現量順で対応が取れる）。
    (re.compile(r'cos_sims_stack = torch\.cat\(cos_sims\)'),
     'cos_sims_stack = _cat_align(cos_sims)'),
]

# 関数まるごとの差し替え（部分置換では直しきれない上流欠陥）。
# `def <name>(` から次のトップレベル定義（`def ` か `#` で始まる行）までの区間を置き換える。
FUNC_MARK = "# scpeft-mps: 次元に依存しない実装"
FUNC_REWRITES: list[tuple[str, str]] = [
    ("make_comparison_batch", '''def make_comparison_batch(original_emb_batch, indices_to_perturb, perturb_group):
    # scpeft-mps: 次元に依存しない実装
    all_embs_list = []

    # 入力を「細胞ごとの 2 次元埋め込み（トークン × 隠れ次元）」の並びに正規化する。
    # 上流はここで 2 次元を前提に切片を作るが、実際には (B, L, H) の 3 次元が渡る。
    # そのまま切ると細胞の軸を切り、長さの違う細胞同士を連結して落ちる。
    if isinstance(original_emb_batch, torch.Tensor):
        if original_emb_batch.dim() >= 3:
            cells = [original_emb_batch[i] for i in range(original_emb_batch.size(0))]
        else:
            cells = [original_emb_batch]
    else:
        cells = list(original_emb_batch)

    # 群摂動（複数遺伝子をまとめて 1 細胞ずつ）か、単一細胞に複数の摂動か。
    if perturb_group:
        src = cells[: len(indices_to_perturb)]
    else:
        src = [cells[0]] * len(indices_to_perturb)

    for cell_emb, indices in zip(src, indices_to_perturb):
        if indices == [-100]:
            all_embs_list.append(cell_emb)
            continue
        if any(isinstance(el, list) for el in indices):
            indices = flatten_list(indices)
        pieces = []
        start = 0
        for pos in sorted(indices):
            pieces.append(cell_emb[start:pos])
            start = pos + 1
        pieces.append(cell_emb[start:])
        all_embs_list.append(torch.cat(pieces, dim=0))

    len_set = set([emb.size()[0] for emb in all_embs_list])
    if len_set and len(len_set) > 1:
        max_len = max(len_set)
        all_embs_list = [pad_2d_tensor(emb, None, max_len, 0) for emb in all_embs_list]
    return torch.stack(all_embs_list)
'''),
]

# import 行の直後に置くヘルパー（datasets 4.x の Column 対応）
HELPER = '''

def _squeeze_keep_batch(x):
    """バッチ次元を潰さずに余分なサイズ 1 の次元だけ落とす。

    上流は `torch.squeeze` で (1, L, H) を (L, H) に潰すが、その後の
    `make_comparison_batch` は先頭次元を「細胞の並び」として数えるため、
    トークン方向の長さを細胞数と誤認してインデックスが範囲外になる
    （1 細胞だけのミニバッチで必ず起きる）。
    """
    return x.squeeze() if x.dim() > 3 else x


def _tensor_from(x):
    """datasets 4.x の Column / list をテンソルにする（すでにテンソルならそのまま）。"""
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.asarray(list(x)))


def _to_device_tensor(x):
    """datasets 4.x の Column/list でも動くようにテンソル化してデバイスへ移す。"""
    if hasattr(x, "to"):
        return x.to(_DEVICE)
    return torch.as_tensor(np.asarray(list(x))).to(_DEVICE)


def _cat_align(tensors, dim=1):
    """トークン方向の長さが違うテンソル群を、最短に揃えてから連結する。

    ミニバッチごとにパディング長が異なるため、そのまま連結すると長さが合わない。
    位置は発現量順の並びなので、短い側を基準に揃えるのが比較として素直
    （長い細胞にしか無い下位の位置は、短い側に対応物が無い）。
    """
    if not tensors:
        return tensors
    n = min(t.size(dim) for t in tensors)
    out = []
    for t in tensors:
        sl = [slice(None)] * t.dim()
        sl[dim] = slice(0, n)
        out.append(t[tuple(sl)])
    return torch.cat(out, dim=0)


def _align_token_len(x1, x2, dim=1):
    """トークン方向の長さがずれた 2 つのテンソルを、短い方に合わせて切り詰める。

    上流の比較バッチ生成は、遺伝子を 1 つ削った摂動側と、パディングで長さを
    揃えた比較側とで系列長がトークン 1 個ずれることがある。位置は発現量順の
    並びなので、はみ出した末尾には対応する位置が無く、切り詰めが妥当。
    """
    n = min(x1.size(dim), x2.size(dim))
    sl = [slice(None)] * x1.dim()
    sl[dim] = slice(0, n)
    x1 = x1[tuple(sl)]
    sl = [slice(None)] * x2.dim()
    sl[dim] = slice(0, n)
    return x1, x2[tuple(sl)]
'''


def _docstring_end(lines: list[str]) -> int:
    """先頭のモジュール docstring が終わる行番号を返す（無ければ -1）。"""
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        i += 1
    if i >= len(lines):
        return -1
    s = lines[i].lstrip()
    for q in ('"""', "'''"):
        if s.startswith(q):
            rest = s[len(q):]
            if q in rest:                       # 1 行で閉じている
                return i
            for j in range(i + 1, len(lines)):
                if q in lines[j]:
                    return j
            return -1
    return -1


def apply_import(src: str) -> str:
    """import 群の直後に 1 行だけ挿入する（冪等）。

    次の 2 つで壊れないようにする:
      - 複数行にまたがる import（バックスラッシュ継続、括弧）の途中に入れない
      - **モジュール docstring の中に入れない**（入れると実行されない）
    """
    if MARK in src:
        return src
    lines = src.split("\n")
    start = _docstring_end(lines) + 1
    last, depth = -1, 0
    for i in range(start, min(len(lines), start + 120)):
        ln = lines[i]
        s = ln.strip()
        started = depth == 0 and s.startswith(("import ", "from "))
        depth += ln.count("(") - ln.count(")")
        cont = ln.rstrip().endswith("\\") or depth > 0
        if started and not cont and depth == 0:
            last = i
    if last < 0:
        lines.insert(start, IMPORT_LINE)
        lines.insert(start + 1, HELPER)
    else:
        lines.insert(last + 1, IMPORT_LINE)
        lines.insert(last + 2, HELPER)
    return "\n".join(lines)


def count_hits(src: str) -> int:
    n = sum(len(p.findall(src)) for p, _ in RULES)
    n += sum(1 for name, _ in FUNC_REWRITES if f"def {name}(" in src and FUNC_MARK not in src)
    return n


def apply_func_rewrite(src: str, name: str, code: str) -> str:
    """`def <name>(` から次のトップレベル定義の直前までを差し替える。"""
    if FUNC_MARK in src:
        return src
    lines = src.split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.startswith(f"def {name}(")), None)
    if start is None:
        return src
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("def ") or lines[j].startswith("#"):
            end = j
            break
    return "\n".join(lines[:start] + code.rstrip("\n").split("\n") + [""] + lines[end:])


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
        for name, code in FUNC_REWRITES:
            new = apply_func_rewrite(new, name, code)
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
