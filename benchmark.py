"""
Ollama URLs are defined in OLLAMA_INSTANCES below.
Override at runtime via the OLLAMA_URLS environment variable:

    OLLAMA_URLS=http://host1:11434,http://host2:11434 python benchmark.py --test_8b

If OLLAMA_URLS is not set, OLLAMA_INSTANCES is used as the default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
from datetime import datetime
from itertools import groupby
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.lgx import lgx
from src.core.knowledge_base import knowledgeBase
from src.helpers.console import get_logger

log = get_logger(__name__)

# Override all of them at once via: OLLAMA_URLS=url1,url2,...
OLLAMA_INSTANCES: list[str] = [
    "http://localhost:11434",
    "http://localhost:11435",
]


BEHAVIOUR_YAML = "benchmark/behaviours/behaviour.lgx_v3.yml"
# At the end of this string there will be appended the model used for the benchmark (i.e. "llama3.1:8b"), so that different models' results are saved in different files.
OUTPUT_DIR = "./results/run_llmasp_lgx_v3"

def get_dataset(filename: str, samples: int) -> list[dict]:
    """
    Load the entire dataset from the JSON file.

    Returns:
        list[dict]: The complete dataset.
    """
    dataset = json.load(open(filename, "r"))

    if samples <= 0:
        return dataset

    _new_dataset = []
    current_problem_name = "None"
    current_count = 0

    for obj in dataset:
        # Check if we have switched to a new problem group
        if obj["problem_name"] != current_problem_name:
            current_problem_name = obj["problem_name"]
            current_count = 0 # Reset the counter for the new group

        if current_count < samples:
            _new_dataset.append(obj)
            current_count += 1

            

    print(f"Dataset filtered to {len(_new_dataset)} entries with a maximum of {samples} samples per problem.")
    return _new_dataset


def get_ollama_urls() -> list[str]:
    """
    Return the list of active Ollama URLs.
    Reads OLLAMA_URLS env var (comma-separated) if set,
    otherwise falls back to the hardcoded OLLAMA_INSTANCES list.
    """
    raw = os.getenv("OLLAMA_URLS", "").strip()
    if raw:
        return [u.strip() for u in raw.split(",") if u.strip()]
    return list(OLLAMA_INSTANCES)


# LGX_OLLAMA_URL (and cache settings) are process-global env vars read at
# lgx.create() time.  A lock ensures that only one thread sets them and
# completes the create() call at a time.  After creation the URL is baked
# into the ollama.Client inside the instance, so concurrent inference is safe.

_create_lock = threading.Lock()


def _create_lgx(
    model: str,
    application_file: str,
    ollama_url: str,
    cache_enabled: bool = True,
    cache_mode: str = "all",
) -> lgx:
    """
    Create an lgx instance bound to *ollama_url*.
    All environment variables that influence lgx construction are set inside
    the lock so parallel workers cannot interfere with each other.
    """
    with _create_lock:
        os.environ["LGX_OLLAMA_URL"]            = ollama_url
        os.environ["LGX_ENABLE_CONDITION_CACHE"] = "true" if cache_enabled else "false"
        os.environ["LGX_CONDITION_CACHE_MODE"]   = cache_mode
        instance = lgx.create(
            llm_model=model,
            behaviour_filename=BEHAVIOUR_YAML,
            application_filename=application_file,
        )
    return instance



def _get_gpu_energy_uj(gpu_index: int = 0) -> float:
    """Accumulated GPU energy in µJ via rocm-smi (returns 0.0 on failure)."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showenergy"], capture_output=True, text=True, check=True
        )
        match = re.search(
            rf"GPU\[{gpu_index}\].*?Accumulated Energy \(uJ\): ([\d\.]+)",
            result.stdout,
        )
        return float(match.group(1)) if match else 0.0
    except Exception:
        return 0.0


def _snapshot() -> tuple[datetime, float]:
    return datetime.now(), _get_gpu_energy_uj()


def _elapsed(start: datetime, start_energy: float) -> dict[str, float]:
    return {
        "total_time_seconds": (datetime.now() - start).total_seconds(),
        "total_energy_uJ":    _get_gpu_energy_uj() - start_energy,
    }

class BenchmarkRunner:

    def __init__(
        self,
        *,
        model: str,
        application_file: str,
        dataset_path: str,
        output_dir: str,
        test_name: str,
        samples: int,
        ollama_url: str,
        use_condition_cache: bool = True,
        cache_non_monotone_only: bool = False,
        conditional: bool = False,
    ) -> None:
        self.model            = model
        self.application_file = application_file
        self.dataset_path     = dataset_path
        self.samples          = samples
        self.ollama_url       = ollama_url
        self.output_file      = Path(output_dir) / f"{test_name}.json"

        if use_condition_cache:
            if cache_non_monotone_only:
                self._cache_mode = "non_monotone"
            else:
                self._cache_mode = "all"
        else:
            self._cache_mode = "all"   # ignored when cache disabled
        self._cache_enabled = use_condition_cache

        self.statistics: dict[str, Any] = {}


    def _infer_single(self, instance: lgx, text: str) -> str:
        """
        Run one inference.  If the application YAML defines a knowledge_base
        program, execute it against the extracted atoms and return its output;
        otherwise return the raw extracted atoms as facts.
        """
        
        if instance.infer(text).kb is not None:
            result_atoms = instance.execute_knowledge_base()
        else:
            result_atoms = instance.get_extracted_atoms()

        return result_atoms.to_facts()


    def run(self) -> None:
        dataset = get_dataset(self.dataset_path, samples=self.samples)
        dataset.sort(key=lambda x: x["problem_name"])

        print(
            f"[{self.model}] Dataset loaded: {len(dataset)} items"
            f" | output → {self.output_file}"
            f" | ollama → {self.ollama_url}"
        )

        # One lgx instance is created per BenchmarkRunner.
        # cleanup() resets internal atom state between items; the Ollama
        # client (and its URL binding) persists for the whole run.
        instance = _create_lgx(
            self.model,
            self.application_file,
            self.ollama_url,
            cache_enabled=self._cache_enabled,
            cache_mode=self._cache_mode,
        )

        global_start, global_energy = _snapshot()

        for problem_name, group in groupby(dataset, key=lambda x: x["problem_name"]):
            items   = list(reversed(list(group)))
            start, start_energy = _snapshot()
            results: list[str] = []

            for item in tqdm(items, desc=f"  {problem_name}", leave=False):
                try:
                    results.append(
                        (self._infer_single(instance, item["text"]),
                         instance.llm_instance.get_statistics()
                        ))
                except Exception:
                    log.error(f"Inference error on {problem_name}", exc_info=True)
                    results.append("ERROR")
                finally:
                    instance.cleanup()

            all_stats_summed = {
                "cache_hits": sum(r[1]["cache_hits"] for r in results if isinstance(r, tuple)),
                "cache_misses": sum(r[1]["cache_misses"] for r in results if isinstance(r, tuple)),
                "llm_calls": sum(r[1]["llm_calls"] for r in results if isinstance(r, tuple)),
                "total_time": sum(r[1]["total_time"] for r in results if isinstance(r, tuple)),
                "total_in_tokens": sum(r[1]["total_in_tokens"] for r in results if isinstance(r, tuple)),
                "total_out_tokens": sum(r[1]["total_out_tokens"] for r in results if isinstance(r, tuple)),
            }

            self.statistics[problem_name] = {
                "results": [r[0] if isinstance(r, tuple) else r for r in results],
                **all_stats_summed,
                #**_elapsed(start, start_energy),
            }

        self.statistics["final_data"] = {
            'total_time_seconds': sum(stat["total_time"] for stat in self.statistics.values()),
            'total_in_tokens': sum(stat["total_in_tokens"] for stat in self.statistics.values()),
            'total_out_tokens': sum(stat["total_out_tokens"] for stat in self.statistics.values()),
            'total_llm_calls': sum(stat["llm_calls"] for stat in self.statistics.values()),
            'total_cache_hits': sum(stat["cache_hits"] for stat in self.statistics.values()),
            'total_cache_misses': sum(stat["cache_misses"] for stat in self.statistics.values()),
        }

        self.statistics["dataset"]    = self.dataset_path

        self._save()
        print(
            f"✓  {self.output_file.name}"
            f"  time={self.statistics['final_data']['total_time_seconds']:.1f}s"
        )

    def _save(self) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.statistics, f, indent=4)


def lgx_runner(
    test_name: str,
    *,
    model: str                = "llama3.1:8b",
    samples: int              = 2,
    dataset: str              = "./benchamrk/dataset/dataset.json",
    application_file: str     = "",
    cache: bool               = True,
    cache_non_monotone_only: bool = False,
    conditional: bool         = False,
    ollama_url: str           = "http://localhost:11434",
) -> None:
    """Thin wrapper that constructs and runs a BenchmarkRunner."""
    BenchmarkRunner(
        model=model,
        application_file=application_file,
        dataset_path=dataset,
        output_dir=f"{OUTPUT_DIR}_{model}",
        test_name=test_name,
        samples=samples,
        ollama_url=ollama_url,
        use_condition_cache=cache,
        cache_non_monotone_only=cache_non_monotone_only,
        conditional=conditional,
    ).run()


def _build_tasks(model: str, n_samples: int) -> list[dict[str, Any]]:
    """
    Return the ordered list of benchmark task descriptors.
    Each dict is a **kwargs payload for lgx_runner (minus ollama_url,
    which is injected by the worker).
    """
    tasks: list[dict[str, Any]] = []

    for dataset_file, test_prefix, app_path in [
        (
            "./benchmark/dataset/dataset_final_lnrs.json",
            "lnrs",
            "LNRS",
        ),
        (
            "./benchmark/dataset/dataset_final_g.json",
            "graph",
            "G",
        ),
    ]:
        base_args = dict(model=model, samples=n_samples, dataset=dataset_file)
        app_dir   = f"./benchmark/applications/{app_path}"

        tasks.extend([
            dict(
                test_name        = f"{test_prefix}_llmasp_base",
                application_file = f"{app_dir}/llmasp-via-lgx-base.yml",
                cache            = True,
                **base_args,
            ),
            dict(
                test_name        = f"{test_prefix}_llmasp_guard",
                application_file = f"{app_dir}/llmasp-via-lgx-guard.yml",
                cache            = True,
                **base_args,
            ),
            dict(
                test_name              = f"{test_prefix}_lgx_base_no_cache",
                application_file       = f"{app_dir}/lgx.yml",
                cache                  = False,
                cache_non_monotone_only= False,
                **base_args,
            ),
            dict(
                test_name              = f"{test_prefix}_lgx_base_cache_non_monotone",
                application_file       = f"{app_dir}/lgx.yml",
                cache                  = True,
                cache_non_monotone_only= True,
                **base_args,
            ),
            dict(
                test_name              = f"{test_prefix}_lgx_base_cache_adv",
                application_file       = f"{app_dir}/lgx.yml",
                cache                  = True,
                cache_non_monotone_only= False,
                **base_args,
            ),
        ])

    return tasks


def _worker(task_queue: "Queue[dict[str, Any]]", ollama_url: str) -> None:
    """
    Thread body.  Drains *task_queue*, running each task against *ollama_url*.
    Tasks are independent — a failure in one does not stop subsequent tasks.
    """
    thread_name = threading.current_thread().name
    while True:
        try:
            task = task_queue.get_nowait()
        except Empty:
            break

        try:
            log.info(f"[{thread_name}] Starting: {task['test_name']} → {ollama_url}")
            lgx_runner(**task, ollama_url=ollama_url)
        except Exception:
            log.error(
                f"[{thread_name}] Task '{task['test_name']}' failed", exc_info=True
            )
        finally:
            task_queue.task_done()

    log.info(f"[{thread_name}] No more tasks — exiting.")


def benchmark_run(model: str, n_samples: int) -> None:
    """
    Build all tasks for *model*, distribute them round-robin across the
    available Ollama instances, and run each instance's queue in a dedicated
    thread.

    Example with two instances:
        instance 0 → tasks 0, 2, 4, …
        instance 1 → tasks 1, 3, 5, …
    """
    tasks = _build_tasks(model, n_samples)
    urls  = get_ollama_urls()

    print(
        f"\n{'─'*60}\n"
        f"  model   : {model}\n"
        f"  tasks   : {len(tasks)}\n"
        f"  workers : {len(urls)}\n"
        f"  urls    : {urls}\n"
        f"{'─'*60}"
    )

    # Distribute tasks across per-URL queues (round-robin)
    queues: list[Queue] = [Queue() for _ in urls]
    for i, task in enumerate(tasks):
        queues[i % len(urls)].put(task)

    # Print assignment summary
    for url, q in zip(urls, queues):
        names = list(q.queue)   # snapshot before threads start
        print(f"  {url} ({q.qsize()} tasks): {[t['test_name'] for t in names]}")
    print()

    # Launch one worker thread per Ollama URL
    threads = [
        threading.Thread(
            target=_worker,
            args=(q, url),
            name=f"worker@{url.split('/')[-1]}",   # e.g. "worker@localhost:11434"
            daemon=False,
        )
        for q, url in zip(queues, urls)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n✓ All {len(tasks)} benchmark tasks for [{model}] completed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LGX benchmarks (optionally in parallel across Ollama instances)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  OLLAMA_URLS   Comma-separated list of Ollama base URLs.
                Defaults to the OLLAMA_INSTANCES list in this file.
                Example: OLLAMA_URLS=http://gpu1:11434,http://gpu2:11434
 
Examples:
  python benchmark.py --test_8b
  python benchmark.py --test_70b --samples 5
  OLLAMA_URLS=http://host1:11434,http://host2:11434 python benchmark.py --test_8b --test_70b
""",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help=(
            "One or more Ollama model tags to benchmark, e.g. "
            "--models llama3.1:8b llama3.1:70b"
        ),
    )
    # Keep the old shorthands as convenience aliases
    parser.add_argument("--test_8b",  action="store_true", help="Alias for --models llama3.1:8b")
    parser.add_argument("--test_70b", action="store_true", help="Alias for --models llama3.1:70b")
    parser.add_argument(
        "--samples", type=int, default=-1,
        help="Max samples per problem group (-1 = all, default: -1)",
    )
    args = parser.parse_args()
 
    # Collect models: explicit list + shorthand aliases, deduplicated, order preserved
    models: list[str] = list(args.models) if args.models else []
    if args.test_8b and "llama3.1:8b" not in models:
        models.append("llama3.1:8b")
    if args.test_70b and "llama3.1:70b" not in models:
        models.append("llama3.1:70b")
 
    if not models:
        parser.print_help()
    else:
        for model in models:
            benchmark_run(model, args.samples)