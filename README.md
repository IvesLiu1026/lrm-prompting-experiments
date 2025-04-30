# Adaptive Reasoning Prompting

This project explores how different prompting strategies affect the reasoning behavior of large language models (LLMs), specifically using the MMLU dataset.

---

## Prompt Types

Prompt templates are defined in `prompts.py` and include:
- `standard`: Basic prompt with no reasoning guidance.
- `adaptive`: Instructs model to reason more slowly if the question is hard, and respond quickly if it is easy.
- `quick`, `slow`: Enforce overall fast or slow thinking styles.
- `without_wait`, `fast_thinking`, `fast_confident`, `smart`, `stupid`: Stylized or constrained prompting.
- `difficulty_aware`, `meta_reasoning`: Dynamic prompting based on difficulty or meta-level judgment.

All templates return a full prompt string given a question and multiple choices.

---

## File Overview

### `main.py`

Run inference on MMLU questions using different prompting strategies and record results.

```bash
python main.py \
    --prompt adaptive \
    --api_key YOUR_KEY \
    --start 0 --end 14042 \
    --shard_id 0 \
    --mode run \
    --folder 20250430
```

Arguments:

- --prompt: Prompt type (must exist in prompt_map)

- --api_key: Environment variable name for Together/OpenAI key

- --start, --end: Index range to run (can split for parallel jobs)

- --shard_id: Identifier for logging this run

- --folder: Output folder name (inside temp/ and log/)

- --fill_missing: (Optional) Fill missing samples from a previous run using log/missing_lists/*.json

- --stream: (Optional) Use streaming response mode

- --mode: run or test (test prints only, does not write)

### `run_inference_parallel.sh`

Launches multiple shards using tmux, ideal for running parallel API jobs.

```bash
bash run_inference_parallel.sh \
    --prompt adaptive \
    --start_key 1 --end_key 12 \
    --folder 20250430
```

Alternative 1:

```bash
bash run_inference_parallel.sh \
    --prompt adaptive \
    --keys together_1 together_2 together_3 \
    --folder 20250430
```

Alternative 2:

```bash
bash run_inference_parallel.sh \
    --prompt adaptive \
    --keys api_keys.txt \
    --folder 20250430
```

### `combine_shards.py`

Merge multiple shard outputs into a single sorted file.

```bash
python combine_shards.py \
    --prompt standard \
    --date 20250430 \
    --outdir output
```

### `analyze.py`
Analyze the combined outputs and compare with standard results:

- Compute accuracy, <think> word counts, flip statistics.

- Generate violin plots, bar charts and scatter charts by subject.

- Output processed CSVs.

```bash
python analyze.py \
    --i adaptive \
    --folder output \
    --mode run
```
Or analyze all prompt files together:

```bash
python analyze.py --all --folder output
```

Generated files:
- output/plots/violin_diff_total_adaptive.png

- utils/word_diff_standard_vs_adaptive.csv

- output/plots/adaptive_flip_subjects.png, etc.

### `prompts.py`
Defines the prompting strategies used by the pipeline.

Each prompt function takes `(question, choices)` and returns a string. The main script selects prompts via `prompt_map`.

## Installation
```bash
uv venv # optionally to specify the venv name
source .venv/bin/activate

uv pip install -r requirements.txt
```