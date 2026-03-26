# LGX

This repository accompanies *Logic-Guided Data Extraction with Answer Set Programming and Large Language Models*. It contains the LGX implementation, benchmark datasets, application specifications, cached oracle responses, and experiment snapshots used in the empirical evaluation reported in Section 5 of the manuscript.

This document is intended as a practical companion for reproducing the reported experiments, inspecting the checked-in artifacts, and re-running the benchmarks either from the bundled cache or against a live Ollama endpoint.

## Repository Contents

- [`run_benchmark.py`](run_benchmark.py): main benchmark runner.
- [`generate_benchmark_table.py`](generate_benchmark_table.py): summarizes benchmark JSON files into a Table 1 style report.
- [`datasets/`](datasets): benchmark instances.
- [`applications/`](applications): LGX and LLMASP-style extraction configurations for each benchmark.
- [`behaviour/behaviour.lgx.yml`](behaviour/behaviour.lgx_v2.yml): shared LGX behavior specification.
- [`experiment_results/`](experiment_results): checked-in benchmark outputs.
- [`cached_prompt.db`](cached_prompt.db): persistent prompt cache used to stabilize oracle responses across runs.

## Experimental Scope

The manuscript evaluates two benchmark families.

- Graph benchmark (`G`): [`datasets/dataset_graph.json`](datasets/dataset_graph.json), 60 layered-graph instances.
- Logic Puzzle benchmark (`LNRS`): [`datasets/dataset_final_lnrs.json`](datasets/dataset_final_lnrs.json), 128 instances in total, with 32 instances each for Labyrinth, Nomystery, Ricochet Robots, and Sokoban.

The three model configurations discussed in the manuscript are mapped to CLI model identifiers as follows.

| Manuscript label | `--model` value | Output directory |
| --- | --- | --- |
| Small (8B) | `llama3.1:8b` | `experiment_results/llama3.1_8b/` |
| Medium (70B) | `llama3.1:70b` | `experiment_results/llama3.1_70b/` |
| Large (120B) | `gpt-oss:120b` | `experiment_results/gpt-oss_120b/` |

The configurations most directly related to Table 1 are:

| Manuscript row | Result file pattern | Meaning |
| --- | --- | --- |
| LLMASP, `None` | `{graph,lnrs}_llmasp_base.json` | Blind predicate-wise extraction.|
| LLMASP, `A posteriori` | `{graph,lnrs}_llmasp_g.json` | Baseline followed by guarded filtering |
| LGX, `By design` | `{graph,lnrs}_lgx_base_cache_adv.json` | Logic-guided extraction |

The repository also stores two additional LGX ablations, `*_lgx_base_no_cache.json` and `*_lgx_base_cache_non_monotone.json`, which are useful for analyzing solver-side caching behavior.

## Requirements

- Python 3.12
- [Poetry](https://python-poetry.org/)
- For fresh online runs only: a reachable Ollama server with the target model already available

Install the Python dependencies with:

```bash
poetry install
```

## Fastest Reproduction Path

If the objective is to inspect the artifact snapshot already included in the repository, no new benchmark execution is required. The checked-in JSON files under [`experiment_results/`](experiment_results) can be summarized directly:

```bash
poetry run python generate_benchmark_table.py --format markdown
```

This command reads the existing result files and produces a table aligned with the manuscript's Table 1 layout, and reported below.

| Bench. | O | Pipeline | O Calls | Guard Enforcement | F1-Score | Perfect Rate |
| --- | --- | --- | --- | --- | --- | --- |
| G | Small (8B) | LLMASP | 240 | None | 0.800 | 0.317 |
|  |  | LLMASP | 240 | A posteriori | 0.893 | 0.617 |
|  |  | LGX | 203 | By design | 0.893 | 0.617 |
|  | Medium (70B) | LLMASP | 240 | None | 0.870 | 0.417 |
|  |  | LLMASP | 240 | A posteriori | 0.952 | 0.850 |
|  |  | LGX | 193 | By design | 0.952 | 0.850 |
|  | Large (120B) | LLMASP | 240 | None | 0.944 | 0.667 |
|  |  | LLMASP | 240 | A posteriori | 1.000 | 1.000 |
|  |  | LGX | 186 | By design | 1.000 | 1.000 |
| LNRS | Small (8B) | LLMASP | 2998 | None | 0.440 | 0.000 |
|  |  | LLMASP | 2998 | A posteriori | 0.842 | 0.023 |
|  |  | LGX | 876 | By design | 0.842 | 0.023 |
|  | Medium (70B) | LLMASP | 3072 | None | 0.572 | 0.000 |
|  |  | LLMASP | 3072 | A posteriori | 0.963 | 0.469 |
|  |  | LGX | 864 | By design | 0.963 | 0.469 |
|  | Large (120B) | LLMASP | 3072 | None | 0.632 | 0.000 |
|  |  | LLMASP | 3072 | A posteriori | 0.947 | 0.555 |
|  |  | LGX | 864 | By design | 0.947 | 0.555 |

## Re-running the Benchmarks from the Bundled Cache

The repository ships with [`cached_prompt.db`](cached_prompt.db), which stores prompt/response pairs for previously executed oracle calls. When the benchmark is launched with `--skip_ollama`, the runner uses only this cache and does not contact an Ollama server.

Run the manuscript models as follows:

```bash
poetry run python run_benchmark.py --model llama3.1:8b --skip_ollama
poetry run python run_benchmark.py --model llama3.1:70b --skip_ollama
poetry run python run_benchmark.py --model gpt-oss:120b --skip_ollama
```

Important notes:

- Re-running a benchmark overwrites the JSON files in the corresponding `experiment_results/<model>/` directory.
- In `--skip_ollama` mode, any prompt/model pair missing from [`cached_prompt.db`](cached_prompt.db) yields an empty extraction result rather than a live oracle call.
- For manuscript-faithful reruns, keep the provided cache unchanged.

## Running Fresh Experiments with Ollama

To execute the benchmarks against a live Ollama deployment, first make sure the server is running and that the desired model has been pulled:

```bash
ollama pull llama3.1:8b
ollama pull llama3.1:70b
ollama pull gpt-oss:120b
```

Set the endpoint and launch the benchmark:

```bash
export LGX_OLLAMA_URL=http://localhost:11434
poetry run python run_benchmark.py --model llama3.1:70b
```

If the Ollama endpoint requires authentication:

```bash
export LGX_OLLAMA_URL=https://your-ollama-endpoint
export LGX_OLLAMA_KEY=your-api-key
poetry run python run_benchmark.py --model llama3.1:70b
```

If a longer timeout is needed for large models:

```bash
export LGX_OLLAMA_URL=http://localhost:11434
export LGX_OLLAMA_TIMEOUT=1200
poetry run python run_benchmark.py --model gpt-oss:120b
```

The runner sets temperature-related options internally and records all new prompt/response pairs in [`cached_prompt.db`](cached_prompt.db).

## Result Files and Interpretation

Benchmark outputs are written to `experiment_results/<model_name>/`, where `<model_name>` is obtained by replacing `:` with `_` in the CLI model identifier.
Each JSON file stores per-problem statistics, including the `llm_calls` counters used to report Oracle interactions in the manuscript. The table generator computes:

- `O(Calls)` by summing `llm_calls` across benchmark sections
- `F1-Score` and `Perfect Rate` by comparing each prediction with the gold output in the corresponding dataset

The table generator intentionally focuses on the three configurations reported in Table 1. The additional LGX ablations remain available in `experiment_results/` for more detailed inspection.

## Reproducibility Notes

- The manuscript is the authoritative reference for the experimental claims and reported metrics.
- The checked-in `experiment_results/` directory provides an artifact snapshot that can be summarized immediately, while live reruns may differ if the underlying LLM backend has changed.