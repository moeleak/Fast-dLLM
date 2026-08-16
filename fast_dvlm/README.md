# Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM

[![Project](https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages)](https://nvlabs.github.io/Fast-dLLM/fast_dvlm/)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2604.06832)
[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/Efficient-Large-Model/Fast_dVLM_3B)

Fast-dVLM is a block-diffusion-based Vision-Language Model (VLM) that enables **KV-cache-compatible parallel decoding** and **speculative block decoding** for inference acceleration. Built on **Qwen2.5-VL-3B-Instruct**, Fast-dVLM directly converts the pretrained AR VLM into a block-diffusion model in a single stage.

## Key Highlights

- **Lossless Quality**: Matches the AR baseline (Qwen2.5-VL-3B) across **11 multimodal benchmarks** (74.0 avg).
- **Up to 6.18x Speedup**: With SGLang integration and FP8 quantization.
- **2.63x Tokens/NFE**: With self-speculative block decoding.
- **Direct Conversion**: Single-stage AR-to-diffusion conversion outperforms two-stage approach (73.3 vs 60.2 avg).

## Key Techniques

- **Block-Size Annealing**: Curriculum that progressively increases the block size during training.
- **Causal Context Attention**: Noisy tokens attend bidirectionally within blocks (N2N), to clean tokens from preceding blocks (N2C), while clean tokens follow causal attention (C2C).
- **Auto-Truncation Masking**: Prevents cross-turn leakage in multi-turn dialogue.
- **Vision-Efficient Concatenation**: Vision embeddings included only in the clean stream, reducing peak memory by 15% and training time by 14.2%.

## Benchmark Results

| Model | AI2D | ChartQA | DocVQA | GQA | MMBench | MMMU | POPE | RWQA | SEED2+ | TextVQA | Avg | Tok/NFE |
|-------|------|---------|--------|-----|---------|------|------|------|--------|---------|-----|---------|
| Qwen2.5-VL-3B | 80.8 | 84.0 | 93.1 | 59.0 | 76.9 | 47.3 | 86.2 | 65.1 | 68.6 | 79.1 | 74.0 | 1.00 |
| **Fast-dVLM (MDM)** | 79.7 | 82.8 | 92.1 | 63.0 | 74.2 | 44.6 | 88.6 | 65.1 | 67.2 | 76.1 | 73.3 | 1.95 |
| **Fast-dVLM (spec.)** | 79.7 | 83.1 | 92.9 | 63.3 | 74.3 | 46.6 | 88.6 | 65.1 | 67.2 | 79.3 | **74.0** | **2.63** |

### Inference Acceleration

| Setting | MMMU-Pro-V | TPS | SpeedUp |
|---------|------------|-----|---------|
| AR baseline | 26.3 | 56.7 | 1.00x |
| Fast-dVLM (MDM, τ=0.9) | 21.4 | 82.2 | 1.45x |
| + Spec. decoding (linear) | 24.6 | 112.7 | 1.98x |
| + SGLang serving | 24.1 | 319.0 | 5.63x |
| + SmoothQuant-W8A8 (FP8) | 23.8 | **350.3** | **6.18x** |

## Quick Start

### Installation

```bash
cd fast_dvlm
pip install -r requirements.txt
```

### Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model_name = "Efficient-Large-Model/Fast_dVLM_3B"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
processor.tokenizer = tokenizer

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text], images=image_inputs, videos=video_inputs,
    padding=True, return_tensors="pt",
).to(model.device)

mask_id = tokenizer.encode("|<MASK>|")[0]

generated_ids = model.generate(
    input_ids=inputs.input_ids,
    tokenizer=tokenizer,
    pixel_values=inputs.pixel_values,
    image_grid_thw=inputs.image_grid_thw,
    mask_id=mask_id,
    max_tokens=512,
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### Command Line Chatbot

Checkpoint is `--model-name` (default `Efficient-Large-Model/Fast_dVLM_3B`): HuggingFace repo id or a local path, same as `MODEL_PATH` for `run_eval.sh`.

```bash
# Single query
python run_chatbot.py --prompt "Describe this image." --image path/to/image.jpg

# Interactive mode
python run_chatbot.py --image path/to/image.jpg
```

Commands in interactive mode:
- Type your message and press Enter
- `clear` - Clear conversation history
- `exit` - Quit the chatbot

## Fine-tuning (example launcher)

This repo ships a **minimal multimodal MDM fine-tuning sample** wired to LMFlow’s `custom_multi_modal` backend (LLaVA-style JSON: `image` + `conversations`). The Python entry parses `ModelArguments`, `MultiModalDatasetArguments`, and the LMFlow `FinetunerArguments`/`TrainingArguments` fields; it fixes `return_as_qwen_messages=True`, builds **`DataCollatorForQwenVL`**, and enables Qwen2.5-VL style `pixel_values` / `image_grid_thw`.

| Path | Role |
|------|------|
| [`train_scripts/finetune_dvlm.py`](train_scripts/finetune_dvlm.py) | Invokes LMFlow `finetuner` + `Dataset(..., backend="custom_multi_modal")` + `AutoModel.get_model(...)`. Supports CLI args or a single `.json` config file (`python finetune_dvlm.py /path/to/args.json`). |
| [`train_scripts/finetune_multimodal_example.sh`](train_scripts/finetune_multimodal_example.sh) | DeepSpeed launcher: exports `PYTHONPATH=<repo>/third_party`, optional resume from latest `checkpoint-*` under `--output_dir`, default ZeRO JSON `v2/configs/ds_config_zero2_no_offload.json`. |
| [`data/download_example_dataset.sh`](data/download_example_dataset.sh) | Fetches ALLaVA-4V LAION split (JSON + optional `images_*.zip` chunks) into `fast_dvlm/data/ALLaVA-4V/` and writes `source_training_env.sh`. |

### Prerequisites

- **Deps:** From repo root: `pip install -r fast_dvlm/requirements.txt` and `pip install -e ./v2/` for the LMFlow CLI package—or rely only on **`PYTHONPATH=<repo>/third_party`** (`finetune_multimodal_example.sh` exports this for you).
- **Runtime:** GPU nodes with **torch**, **DeepSpeed**, **transformers**, **Pillow**, **datasets** (`huggingface_hub` for the downloader).
- **Checkpoint:** override the launcher default with a public or local checkpoint, e.g. `MODEL_PATH=Efficient-Large-Model/Fast_dVLM_3B`.

### Dataset (ALLaVA-4V)

From **Fast-dLLM repo root**:

```bash
pip install -U huggingface_hub  # provides `hf` CLI for downloader
bash fast_dvlm/data/download_example_dataset.sh allava
# Smoke test (~one 9 GB chunk): IMAGE_CHUNKS=0 bash fast_dvlm/data/download_example_dataset.sh allava
# JSON manifest only: JSON_ONLY=1 bash fast_dvlm/data/download_example_dataset.sh allava
```

Then:

```bash
source fast_dvlm/data/ALLaVA-4V/source_training_env.sh   # exports DATASET_PATH / IMAGE_FOLDER
```

Alternatively set `DATASET_PATH` (single JSON list of samples) and `IMAGE_FOLDER` (root containing paths like `allava_laion/images/...`) yourself for any LMFlow-compatible multimodal JSON.

### Run training

```bash
# From Fast-dLLM repository root (recommended)
MODEL_PATH=Efficient-Large-Model/Fast_dVLM_3B \
TOKENIZER_NAME=Qwen/Qwen2.5-VL-3B-Instruct \
bash fast_dvlm/train_scripts/finetune_multimodal_example.sh
```

Writes to `OUTPUT_DIR` (default: `Fast-dLLM/output_models/finetune_fast_dVLM_3B_example`). Common overrides via environment variables:

- **`MODEL_PATH`**, **`TOKENIZER_NAME`**, **`OUTPUT_DIR`**, **`DATASET_PATH`**, **`IMAGE_FOLDER`**
- **`DEEPSPEED_CONFIG`** (default points at `v2/configs/ds_config_zero2_no_offload.json`)
- **`MASTER_PORT`** or full **`DEEPSPEED_ARGS`**
- **Hyperparameters:** `NUM_TRAIN_EPOCHS`, `LEARNING_RATE`, `PER_DEVICE_TRAIN_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `SAVE_STEPS`, `MAX_STEPS`, `WARMUP_RATIO`, … (passed through to HF `TrainingArguments` / LMFlow)

MDM knobs such as **`--mdm`**, **`--bd_size`**, and **`--block_size`** are available from LMFlow **`ModelArguments` / dataset args**; add them by editing the launcher or invoking `python fast_dvlm/train_scripts/finetune_dvlm.py --help`.

## GUI Planner + residual Grounder workflow

The production GUI recipe is separate from the minimal ALLaVA example. It
trains one full Fast-dVLM Planner and then a residual Grounder adapter on that
selected Planner checkpoint. The runtime owns one SGLang engine: Planner
requests pass no adapter, while Grounder requests enable one pinned PEFT LoRA.
It never builds a merged checkpoint or loads a second vision tower.

The base snapshot is pinned to
`Efficient-Large-Model/Fast_dVLM_3B@d7977da26e374f3ef7c96c1700b2ebab50ff62fc`.
The native block-diffusion objective, causal auxiliary loss, and block size 32
are retained. A800 runs use BF16; FP8/W8A8 is intentionally disabled.

### Data preparation and audits

```bash
python fast_dvlm/data/prepare_gui_data.py planner \
  --source-dir /path/to/unigui-planner \
  --image-root /path/to/Uni-GUI-OpenMobile \
  --output-dir /new/experiment/data/planner

python fast_dvlm/data/prepare_gui_data.py grounder \
  --mind2web-dir /path/to/mind2web-ocr-train \
  --mobile-dir /path/to/mobile-grounding-train \
  --output-dir /new/experiment/data/grounder \
  --mind2web-validation-root /path/to/mind2web-validation \
  --mind2web-validation-key mind2web_validation \
  --mobile-validation-root /path/to/mobile-validation \
  --mobile-validation-key mobile_validation \
  --mind2web-test-root /path/to/fixed-mind2web-test100 \
  --mind2web-test-key mind2web_test
```

Preparation fails closed on source hashes/counts, missing images, malformed
turns/actions, duplicate IDs, and train/validation/test leakage. Planner
conversion also writes an action-stratified, SHA-256-stable
`validation-100.json`. The first real collated batch is audited to prove that
user tokens are masked and assistant target tokens are supervised.

### Two-stage training

Use a new output path. Both launchers require at least 300 GiB free disk and
70 GiB free on each of exactly two visible GPUs. The preflight launcher runs a
two-step gradient/save smoke and a 1+resume versus uninterrupted consistency
check. It uses ZeRO-2 without offload unless the smoke OOMs or reserves more
than 72 GiB, in which case it changes only to ZeRO-3 without offload.

```bash
STAGE=planner \
DATASET_PATH=/new/experiment/data/planner/train.json \
SOURCE_AUDIT=/new/experiment/data/planner/audit.json \
OUTPUT_DIR=/new/experiment/training/planner \
bash fast_dvlm/train_scripts/run_gui_stage_with_preflight.sh

STAGE=grounder \
MODEL_PATH=/new/experiment/training/planner/checkpoint-SELECTED \
TOKENIZER_NAME=/new/experiment/training/planner \
DATASET_PATH=/new/experiment/data/grounder/train.json \
SOURCE_AUDIT=/new/experiment/data/grounder/audit.json \
OUTPUT_DIR=/new/experiment/training/grounder \
bash fast_dvlm/train_scripts/run_gui_stage_with_preflight.sh
```

Planner trains every language, vision, and connector parameter for 1,626
optimizer steps with separate `1e-6 / 1e-7 / 2e-6` learning rates. Grounder
freezes the selected Planner and trains rank-32, alpha-32, dropout-0.1 LoRA on
exactly 36 × (`q_proj`, `k_proj`, `v_proj`, `o_proj`) = 144 text-attention
modules (14,745,600 parameters). Its domain-balanced epoch contains 16,528
draws, giving exactly 1,033 optimizer steps per epoch and 3,099 total.

For the complete validation-only selection and single held-out test run:

```bash
WORK_ROOT=/new/unique/experiment \
MIND2WEB_TEST=/path/to/fixed-ocr-test100 \
MIND2WEB_TEST_KEY=mind2web_test \
bash fast_dvlm/train_scripts/run_gui_pipeline.sh
```

The pipeline selects Planner checkpoints at steps 813/1626, selects Grounder
checkpoints at 1033/2066/3099 on two independent validation-100 sets, then
selects MDM versus speculative decoding on validation. Only after all choices
are frozen does it run the ordered OCR-aligned Mind2Web test-100 once, with two
persistent one-GPU workers handling 50 samples each. OCR is not run during
inference; the benchmark consumes the pre-generated OCR-aligned crop and SSR
annotation. `final/comparison.json` records quality, mean/p50/p95 latency,
weights, peak GPU memory, and the fixed baselines.

## Evaluation (VLMEvalKit)

[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) is **vendored** at `../third_party/VLMEvalKit` (i.e. `Fast-dLLM/third_party/VLMEvalKit`). `run_eval.sh` runs one dataset per invocation; **default `TASK` is `DocVQA_VAL`** as a concrete example—override with `TASK=…` for any other VLMEval split.

From the **Fast-dLLM repository root**:

```bash
pip install -r fast_dvlm/requirements.txt
pip install -e third_party/VLMEvalKit
```

Example (DocVQA val split by default). Use the **same checkpoint** as the chatbot: HuggingFace id or local directory for `run_chatbot.py --model-name` (default `Efficient-Large-Model/Fast_dVLM_3B`).

```bash
bash fast_dvlm/run_eval.sh --help
MODEL_PATH=Efficient-Large-Model/Fast_dVLM_3B bash fast_dvlm/run_eval.sh
# Local tree: MODEL_PATH=/path/to/Fast_dVLM_3B bash fast_dvlm/run_eval.sh
# Other split: TASK=MMBench_DEV_EN_V11 DATASET_CLASS=ImageMCQDataset MODEL_PATH=… bash fast_dvlm/run_eval.sh
```

By default inference uses the checkpoint’s own `generate` in `modeling.py` (`trust_remote_code` + `AutoModelForCausalLM`), same stack as `run_chatbot.py`. This folder only adds `vlmeval_run.py` (VLMEval config + thin wrapper) and `run_eval.sh`. If weights live in a folder without a processor, set `PROCESSOR_PATH` (e.g. `Qwen/Qwen2.5-VL-3B-Instruct`); otherwise the processor is loaded from `MODEL_PATH` like the chatbot.

**SGLang backend.** Set `BACKEND=sglang` to run the same eval through the vendored SGLang fork (`sgl.Engine`, same stack as `run_chatbot_sglang.py`) instead of HF. Requires `pip install -e third_party/sglang/python`.

```bash
# MDM (HierarchyBlock) via SGLang
BACKEND=sglang ALGORITHM=mdm MODEL_PATH=Efficient-Large-Model/Fast_dVLM_3B bash fast_dvlm/run_eval.sh

# Speculative block decoding
BACKEND=sglang ALGORITHM=spec MODEL_PATH=Efficient-Large-Model/Fast_dVLM_3B bash fast_dvlm/run_eval.sh

# FP8 W8A8 quantized checkpoint (requires SM89+: 4090 / L40 / H100 / H200)
BACKEND=sglang ALGORITHM=spec QUANTIZATION=w8a8_fp8 \
  MODEL_PATH=Sensen02/Fast_dVLM_3B_W8A8_FP8 bash fast_dvlm/run_eval.sh
```

`BACKEND=sglang` adds `ALGORITHM` (`mdm`|`spec`, default `mdm`), `QUANTIZATION` (e.g. `w8a8_fp8`), and `MEM_FRACTION_STATIC` (default `0.75`); other knobs (`TASK`, `MAX_TOKENS`, `PROCESSOR_PATH`, …) behave the same. See the [FP8 Quantized Checkpoint](#fp8-quantized-checkpoint) section for hardware requirements.

To refresh VLMEvalKit, replace `third_party/VLMEvalKit` and commit.

## SGLang-Accelerated Inference

Fast-dVLM ships with a customized SGLang fork that implements two dLLM algorithms:

- **`HierarchyBlock`** — block-diffusion parallel decoding (MDM mode)
- **`SpeculativeBlock`** — self-speculative block decoding (≈2.6× tokens/NFE)

### Install

The customized SGLang fork is **vendored** at `../third_party/sglang` (i.e. `Fast-dLLM/third_party/sglang`), alongside `third_party/VLMEvalKit`. From the repo root:

```bash
pip install -e third_party/sglang/python
```

The install pulls in SGLang's native dependencies (flashinfer, sgl-kernel, transformers, etc.). Use a dedicated conda env to avoid version conflicts.

### Command Line Chatbot (SGLang)

```bash
# MDM (HierarchyBlock)
python run_chatbot_sglang.py --algorithm mdm --prompt "Describe this image." --image path/to/image.jpg

# Speculative block decoding
python run_chatbot_sglang.py --algorithm spec --prompt "Describe this image." --image path/to/image.jpg

# Interactive mode
python run_chatbot_sglang.py --algorithm spec --image path/to/image.jpg
```

Key flags:
- `--algorithm {mdm,spec}` — select MDM (HierarchyBlock) or speculative decoding (SpeculativeBlock)
- `--model-path` — HF id or local path (default `Efficient-Large-Model/Fast_dVLM_3B`)
- `--processor-path` — HF processor for chat template + image preprocessing (default `Qwen/Qwen2.5-VL-3B-Instruct`)
- `--max-tokens`, `--mem-fraction-static` — generation length / GPU memory budget
- `--quantization w8a8_fp8` — load the FP8 checkpoint (see below)

If you hit a CuDNN/PyTorch 2.9 compatibility warning, set `SGLANG_DISABLE_CUDNN_CHECK=1` in the environment before launch.

### FP8 Quantized Checkpoint

We provide a SmoothQuant-W8A8 FP8 checkpoint for the 6.18× speedup reported above:

- [`Sensen02/Fast_dVLM_3B_W8A8_FP8`](https://huggingface.co/Sensen02/Fast_dVLM_3B_W8A8_FP8) — language tower in FP8 (E4M3), visual encoder kept in BF16.

Hardware requirement: **SM89+** (RTX 4090 / L40 / H100 / H200). Earlier GPUs (A100, V100) do not have FP8 tensor cores and are not supported.

Launch with `--quantization`:

```bash
# FP8 inference (requires SM89+)
python run_chatbot_sglang.py \
    --algorithm spec \
    --model-path Sensen02/Fast_dVLM_3B_W8A8_FP8 \
    --quantization w8a8_fp8 \
    --prompt "Describe this image." \
    --image path/to/image.jpg
```

The quantized checkpoint ships with a `quantization_config` entry in `config.json`:

```json
"quantization_config": {
    "quant_method": "w8a8_fp8",
    "is_dynamic": false,
    "ignore": ["re:visual.*"]
}
```

SGLang reads this automatically:
- Layers matching `ignore` (visual encoder) stay in BF16.
- Remaining linear layers use per-channel static FP8 weights + per-token dynamic FP8 activations.

> Running on H100 (SM90)? Diffusion decoding produces short token blocks; the CUTLASS TMA kernel requires ≥64 rows, so we fall back to the Triton FP8 GEMM for short batches automatically. No extra flags needed.

## File Structure

```
Fast-dLLM/
├── third_party/
│   ├── VLMEvalKit/
│   ├── sglang/                 # Customized SGLang with Fast-dVLM model + dLLM algorithms
│   └── lmflow/                 # LMFlow fork (multimodal finetuner; PYTHONPATH via train script)
├── v2/
│   └── configs/                # e.g. ds_config_zero2_no_offload.json (DeepSpeed ZeRO used by train sample)
└── fast_dvlm/
    ├── README.md
    ├── requirements.txt
    ├── train_scripts/
    │   ├── finetune_multimodal_example.sh  # DeepSpeed + env-driven hyperparameters
    │   └── finetune_dvlm.py                # LMFlow finetuner entry (custom_multi_modal)
    ├── data/
    │   └── download_example_dataset.sh     # ALLaVA-4V helper (+ source_training_env.sh)
    ├── run_chatbot.py
    ├── run_chatbot_sglang.py   # SGLang-backed chatbot (MDM + speculative)
    ├── vlmeval_run.py          # config + VLMEval hook (HF ckpt ``generate`` or SGLang backend)
    └── run_eval.sh             # VLMEval driver (default TASK=DocVQA_VAL)
```

## Citation

```bibtex
@misc{wu2026fastdvlmefficientblockdiffusionvlm,
      title={Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM},
      author={Chengyue Wu and Shiyi Lan and Yonggan Fu and Sensen Gao and Jin Wang and Jincheng Yu and Jose M. Alvarez and Pavlo Molchanov and Ping Luo and Song Han and Ligeng Zhu and Enze Xie},
      year={2026},
      eprint={2604.06832},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.06832},
}
```

## Acknowledgements

We thank [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) for the base model architecture.
