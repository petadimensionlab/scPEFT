# Apple Silicon（MPS）で scPEFT を動かす

このフォークは、Geneformer 系のモジュールに直書きされていた `cuda` を
**MPS / CUDA / CPU を自動で選ぶ 1 行**に置き換え、5 つのタスクを Python スクリプトで実行できるようにしています。

## 1. 何を変えたか

| 変更 | 内容 |
|---|---|
| `scpeft_mps/`（新規） | デバイス決定を 1 か所にまとめた補助パッケージ |
| `patches/apply_mps_patch.py`（新規） | `geneformer_peft/geneformer/*.py` の `cuda` 直書きを置換するパッチャ |
| `scripts/`（新規） | 5 タスクの実行スクリプト |

**置換した件数: 27 件 / 3 ファイル**

| ファイル | 件数 |
|---|---|
| `geneformer_peft/geneformer/emb_extractor.py` | 5 |
| `geneformer_peft/geneformer/emb_extractor_h5ad.py` | 8 |
| `geneformer_peft/geneformer/in_silico_perturber.py` | 14 |

置換規則:

```
.to("cuda")              → .to(_DEVICE)
.cuda()                  → .to(_DEVICE)
torch.device("cuda")     → _DEVICE
torch.cuda.empty_cache() → _empty_cache()
device="cuda"            → device=_DEVICE
```

`_DEVICE` と `_empty_cache` は `scpeft_mps.device` から読み込みます。
決定順序は **環境変数 `SCPEFT_DEVICE` → MPS → CUDA → CPU** です。

## 2. セットアップ（初回だけ）

このリポジトリは **vendored の transformers を zip で同梱**しているため、初回に 3 点の手当てが要ります。
`patches/apply_vendor_patch.py` が 1 と 2 を自動で行います。

```bash
python patches/apply_vendor_patch.py --check   # 状態を確認
python patches/apply_vendor_patch.py           # 1 と 2 を適用
```

| # | 症状 | 対処 |
|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'transformerslocal.src'` | 同梱 zip（19 MB）を展開する。zip は 1 階層深く展開されるので、中身を `transformerslocal/` 直下へ移す |
| 2 | `ImportError: tokenizers>=0.14,<0.19 is required … found tokenizers==0.20.3` | vendored 側の版検査を 1 行だけ緩める。scPEFT の Geneformer 経路は HF の tokenizers を呼ばないため実害がない |
| 3 | `ModuleNotFoundError: No module named 'loralib'` | `pip install loralib`（scPEFT の LoRA 実装が使う） |

追加で入れておくとよいもの:

```bash
pip install loralib          # 必須（vendored BERT が import する）
pip install harmonypy        # 任意（バッチ補正で Harmony を使う場合。無い場合は PCA 補正に切り替わる）
```

## 3. MPS 化パッチの使い方

```bash
# 置換対象の確認（書き換えない）
python patches/apply_mps_patch.py --check

# 適用（各ファイルの .orig を残す）
python patches/apply_mps_patch.py

# 元に戻す
python patches/apply_mps_patch.py --revert
```

**`git pull` で `geneformer_peft/` を更新したら、もう一度適用してください。**
上流のファイルが `cuda` 直書きに戻るためです。

## 4. 環境変数

| 変数 | 既定 | 意味 |
|---|---|---|
| `SCPEFT_DEVICE` | 自動 | `mps` / `cuda` / `cpu` を明示する |
| `SCPEFT_DTYPE` | `float32` | `float32` / `bfloat16` / `float16` |
| `SCPEFT_BATCH` | 316M は 4、104M は 8 | `forward_batch_size` |
| `GENEFORMER_DIR` | `~/workspace/research/Geneformer/geneformer_hf` | 重みと辞書の場所 |
| `GENEFORMER_MODEL` | `Geneformer-V2-316M` | 使うモデル |

## 5. MPS の実測制約（重要）

1. **`float64` が使えない。** 倍精度が必要な計算は CPU に移す必要があります。
   `scpeft_mps.device.to_device()` は MPS に渡すとき float64 を float32 に落とします。
2. **attention テンソルが `INT_MAX` を超えると即落ちする。**
   `batch × heads × seq²` を計算し、`2³¹ − 1`（2,147,483,647）を超えると
   `MPSGraph does not support tensor dims larger than INT_MAX` で停止します。
   V2-316M は 18 heads、系列長 4,096 のとき **batch 4 が上限**（batch 8 は 2.42e9 で超過）。
   104M は 12 heads なので batch 8 でも収まります。`gradient_checkpointing` では回避できません。
3. **bfloat16 は macOS 14 以降。** 不安定な場合は `SCPEFT_DTYPE=float32` にしてください。
4. `torch.cuda.empty_cache()` は存在しないため、`torch.mps.empty_cache()` を使います。

参考（同リポジトリの Geneformer 側の実測）: 316M + 系列長 4,096 で batch 4 が安全、
ISP の `forward_batch_size` にも同じ制約がかかります（埋め込み抽出だけでなく摂動でも同じ）。

## 5. 5 つのタスク

| スクリプト | タスク | 主な出力 |
|---|---|---|
| `01_cell_type_identification.py` | 細胞型の同定 | `metrics.json`（probe と LoRA）、`umap_celltype.png` |
| `02_batch_correction.py` | バッチ補正 | `metrics.json`（補正前後の iLISI・ASW・シルエット） |
| `03_perturbation.py` | in silico 遺伝子削除 | `perturb/`、`stats/`、`provenance.json` |
| `04_cell_population_discovery.py` | 細胞集団の発見 | `metrics.json`、`umap_populations.png` |
| `05_marker_gene_detection.py` | マーカー遺伝子の検出 | `markers_expression.json`、`markers_context.json` |

### 入力 h5ad の要件（Geneformer と同じ）

- `var_names` が **Ensembl ID**
- `obs` に **`n_counts`**（細胞ごとの総カウント）
- `obs` に **`filter_pass`**（無ければスクリプトが 1 を立てます）
- 細胞型の列（既定 `celltype`）、バッチの列（既定 `batch`）

### 実行例

```bash
cd scripts

# 0. デモデータ（任意。手元のデータが無いとき）
python make_demo_h5ad.py --per-type 400 --out /tmp/scpeft_demo.h5ad

# 1. 細胞型の同定
python 01_cell_type_identification.py --h5ad /tmp/scpeft_demo.h5ad \
    --label-key celltype --batch-key batch --max-cells 1200 --batch 4 \
    --out ../outputs/01_identification

# 2. バッチ補正（補正前後を必ず対で見る）
python 02_batch_correction.py --dataset ../outputs/01_identification/tokenized/ident.dataset \
    --label-key celltype --batch-key batch --method harmony --out ../outputs/02_batch

# 3. 摂動（遺伝子削除）
python 03_perturbation.py --dataset ../outputs/01_identification/tokenized/ident.dataset \
    --genes SNCA,LRRK2,PINK1 --celltype MG --state-key disease --start tg --goal WT \
    --max-ncells 300 --batch 4 --out ../outputs/03_perturbation

# 4. 細胞集団の発見
python 04_cell_population_discovery.py --dataset ../outputs/01_identification/tokenized/ident.dataset \
    --label-key celltype --resolution 1.0 --out ../outputs/04_population

# 5. マーカー遺伝子の検出
python 05_marker_gene_detection.py --h5ad /tmp/scpeft_demo.h5ad \
    --label-key celltype --query-genes SNCA,TREM2 --top-n 20 --out ../outputs/05_marker
```

## 6. 解釈のときの注意

- **摂動の `Shift` は単独で意味を持ちません。** 生物学的に無関係な遺伝子（null）を同じ条件で走らせ、
  その分布と比べてから順位を主張してください。実際に測ると、無関係な遺伝子の多くが負側に出ます
  （削除すれば何でも goal から離れる方向に動く）。
- **バッチ補正は「混合が進んだか」と「細胞型が潰れていないか」を対で見ます。** 片方だけでは判断できません。
- **埋め込みの絶対値は計算精度（float32 / bfloat16）で動きます。** 条件を揃えて比較してください。

---

## LoRA は ISP のピークメモリを削減しない（2026-09-17 実測）

**観点**: LoRA（PEFT）を使うと、ISP のピークメモリが下がるか。

**基準**: 削減効果があると言うには、**LoRA あり**のピークが**LoRA なし**を下回ること（差 < 0）。
同一の入力・遺伝子・細胞数・バッチ・系列長で比較します。

**条件**: モデル V2-316M、遺伝子 `SNCA`、細胞型 `MG`、`max_ncells=60`、
`perturb_type="delete"`、`genes_to_perturb=[ENSG00000145335]`、系列長はデータ依存、
LoRA は `loralib` の `Linear` を attention の query・key・value に r=8 で適用（54 箇所）、
base は凍結。実行は `scripts/07_lora_isp_memory.py`、パラメータ数は `scripts/07b_lora_param_count.py`。

**実測**

| forward バッチ | LoRA なし | LoRA あり | 差 | 所要時間（なし / あり） |
|---|---|---|---|---|
| 4 | **8.74 GiB**（テンソル 6.10 GiB） | **8.84 GiB**（テンソル 7.16 GiB） | **+0.10 GiB（+1.1%）** | 26.7 秒 / 27.3 秒 |
| 2 | **5.04 GiB**（テンソル 3.15 GiB） | **5.04 GiB**（テンソル 3.44 GiB） | **±0.00 GiB** | 26.8 秒 / 27.8 秒 |

| パラメータ | 合計 | 学習対象 |
|---|---|---|
| LoRA なし | 316,354,867 | 316,354,867 |
| LoRA あり（r=8、54 箇所） | 317,350,195 | **995,328** |
| 差 | +995,328（+0.315%） | **−99.7%** |

**判定**: **不合格**（LoRA が ISP のピークメモリを削減する、という主張は成立しない）

**解釈**

- ISP は**推論のみ**です。勾配と最適化器の状態を持たないため、**LoRA の節約が効く場所がありません**。
  LoRA が減らすのは「学習対象パラメータ」と、それに付随する勾配・Adam のモーメンタムです（学習時の話）。
- 推論では base の重みを常に保持する必要があり、アダプタの分だけ**わずかに増えます**（+0.315% のパラメータ）。
  batch 4 で +0.10 GiB は、アダプタ本体（約 4 MiB）より大きいため、追加の行列積が作る一時領域と
  アロケータの粒度による変動です。batch 2 では差が 0.00 GiB でした。
- ISP のピークを決めるのは **`バッチ × ヘッド数 × 系列長²`** の attention 活性です。この項は LoRA では
  変わりません（batch 4 で 8.74 GiB、batch 2 で 5.04 GiB と、ほぼ半減することはこの式と整合します）。
- **メモリを下げたい場合は `--batch` を下げます。** LoRA の有無は手段になりません。

**この結果の使いどころ**: LoRA の利点は学習（メモリと計算）にあり、推論（ISP）にはありません。
ISP のメモリ見積もりでは、LoRA の有無を条件に入れる必要はありません。
