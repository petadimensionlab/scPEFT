"""LoRA の有無で ISP（単一遺伝子）のピークメモリが変わるかを実測する。

条件を揃えるため、入力・遺伝子・細胞数・バッチ・系列長を固定し、
モデルに LoRA を挿すかどうかだけを変えて 2 回測る。

使い方:
  PYTHONPATH=.. python 07_lora_isp_memory.py --dataset <ident.dataset> \
      --gene SNCA --celltype MG --max-ncells 60 --batch 4
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (DEVICE, MODEL_DIR, TOKEN_DICT, banner, log, memory_guard,  # noqa: E402
                     out_dir, save_json, symbols_to_ensembl)

LORA_R = 8


def make_lora_loader(enable: bool):
    """ISP が内部で呼ぶ load_model を包み、LoRA を挿すかどうかを切り替える。"""
    import loralib as lora

    import geneformer_peft.geneformer.in_silico_perturber as isp_mod

    original = isp_mod.load_model

    def wrapped(model_type, num_classes, model_directory):
        model = original(model_type, num_classes, model_directory)
        if enable:
            n = 0
            for layer in model.bert.encoder.layer:
                att = layer.attention.self
                for name in ("query", "key", "value"):
                    linear = getattr(att, name)
                    new = lora.Linear(linear.in_features, linear.out_features, r=LORA_R)
                    new.weight.data.copy_(linear.weight.data)
                    if linear.bias is not None:
                        new.bias.data.copy_(linear.bias.data)
                    # 置き換えた層は元と同じデバイス・型に置く（既定は CPU になる）
                    new.to(device=linear.weight.device, dtype=linear.weight.dtype)
                    setattr(att, name, new)
                    n += 1
            lora.mark_only_lora_as_trainable(model)
            log(f"    LoRA を挿しました（{n} 箇所、r={LORA_R}）")
        return model

    return wrapped


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable),
            "frozen": int(total - trainable)}


def run_isp(ds: Path, gene: str, args, out: Path, with_lora: bool) -> dict:
    import geneformer_peft.geneformer.in_silico_perturber as isp_mod

    isp_mod.load_model = make_lora_loader(with_lora)

    from geneformer_peft.geneformer.in_silico_perturber import InSilicoPerturber

    tag = "lora" if with_lora else "base"
    target = out / tag
    target.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with memory_guard() as tracker:
        isp = InSilicoPerturber(
            perturb_type="delete",
            genes_to_perturb=[gene],
            combos=0,
            anchor_gene=None,
            model_type="Pretrained",
            num_classes=0,
            emb_mode="cell",
            cell_emb_style="mean_pool",
            filter_data={"celltype": [args.celltype]} if args.celltype else None,
            cell_states_to_model=None,
            max_ncells=args.max_ncells,
            emb_layer=-1,
            forward_batch_size=args.batch,
            nproc=1,
            token_dictionary_file=TOKEN_DICT,
        )
        isp.perturb_data(model_directory=str(MODEL_DIR),
                         input_data_file=str(ds),
                         output_directory=str(target) + "/",
                         output_prefix=gene)
    rec = tracker.summary()
    rec["tag"] = tag
    rec["elapsed_sec"] = round(time.time() - t0, 1)
    log(f"    {tag}: ピーク MPS {rec['peak_mps_driver_gib']:.2f} GiB / {rec['elapsed_sec']} 秒")
    return rec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--gene", default="SNCA")
    p.add_argument("--celltype", default="MG")
    p.add_argument("--max-ncells", type=int, default=60)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ds = Path(args.dataset)
    out = out_dir(args.out, "07_lora_isp_memory")
    genes = symbols_to_ensembl([args.gene])
    if not genes:
        raise SystemExit("遺伝子が辞書に見つかりません")
    gene = genes[0]

    banner("LoRA の有無による ISP（単一遺伝子）のメモリ比較")
    log(f"  デバイス: {DEVICE}")
    log(f"  条件: 遺伝子 {args.gene}・細胞 {args.celltype}・max_ncells {args.max_ncells}"
        f"・forward_batch {args.batch}・系列長はデータ依存")

    from scpeft_mps.device import empty_cache
    from geneformer_peft.geneformer.in_silico_perturber import load_model

    # 参考: パラメータ数の内訳（凍結と学習対象）
    base_model = load_model("Pretrained", 0, str(MODEL_DIR))
    base_counts = count_params(base_model)
    log(f"  base のパラメータ: 合計 {base_counts['total']:,}"
        f"（学習対象 {base_counts['trainable']:,}）")
    del base_model
    empty_cache()

    results = []
    results.append(run_isp(ds, gene, args, out, with_lora=False))
    empty_cache()
    results.append(run_isp(ds, gene, args, out, with_lora=True))

    b, l = results[0], results[1]
    save_json({"gene": gene, "symbol": args.gene, "celltype": args.celltype,
               "max_ncells": args.max_ncells, "batch": args.batch,
               "lora_r": LORA_R, "base_params": base_counts, "runs": results},
              out / "lora_isp_memory.json")

    log("")
    log("  ── 結果 ──")
    log(f"  LoRA なし: MPS ピーク {b['peak_mps_driver_gib']:.2f} GiB"
        f" / テンソル {b['peak_mps_tensor_gib']:.2f} GiB / {b['elapsed_sec']} 秒")
    log(f"  LoRA あり: MPS ピーク {l['peak_mps_driver_gib']:.2f} GiB"
        f" / テンソル {l['peak_mps_tensor_gib']:.2f} GiB / {l['elapsed_sec']} 秒")
    d = l["peak_mps_driver_gib"] - b["peak_mps_driver_gib"]
    pc = 100.0 * d / b["peak_mps_driver_gib"] if b["peak_mps_driver_gib"] else 0.0
    log(f"  差: {d:+.2f} GiB（{pc:+.1f}%）")
    log("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
