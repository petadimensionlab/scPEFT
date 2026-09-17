"""LoRA 適用時のパラメータ数（凍結・学習対象）を実測する。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loralib as lora  # noqa: E402
import torch  # noqa: E402

from _common import MODEL_DIR, log  # noqa: E402
from geneformer_peft.geneformer.in_silico_perturber import load_model  # noqa: E402


def counts(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def main():
    model = load_model("Pretrained", 0, str(MODEL_DIR))
    total0, train0 = counts(model)
    log(f"  base: 合計 {total0:,} / 学習対象 {train0:,}")

    n = 0
    for layer in model.bert.encoder.layer:
        att = layer.attention.self
        for name in ("query", "key", "value"):
            linear = getattr(att, name)
            new = lora.Linear(linear.in_features, linear.out_features, r=8)
            new.weight.data.copy_(linear.weight.data)
            if linear.bias is not None:
                new.bias.data.copy_(linear.bias.data)
            new.to(device=linear.weight.device, dtype=linear.weight.dtype)
            setattr(att, name, new)
            n += 1
    lora.mark_only_lora_as_trainable(model)
    total1, train1 = counts(model)
    log(f"  LoRA（r=8、{n} 箇所）: 合計 {total1:,} / 学習対象 {train1:,}")
    log(f"  追加パラメータ: {total1 - total0:,}"
        f"（全体の {100.0 * (total1 - total0) / total0:.3f}%）")
    log(f"  学習対象の削減: {train0:,} → {train1:,}"
        f"（{100.0 * train1 / max(train0, 1):.3f}%）")


if __name__ == "__main__":
    main()
