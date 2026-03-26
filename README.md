## Running Experiments

Follow these steps to run the benchmark experiments with LGX and Ollama.

### 0. Install Poetry and Set Up the Project

First, install Poetry (if not already installed):


https://poetry.org 


Verify the installation:

```bash
poetry --version
```

Then, go inside the lgx folder and install the project dependencies:

```bash
poetry install
```

This will create a virtual environment and install all required packages.

---

### 1. Install and Start Ollama
Make sure you have Ollama installed and running on your machine.

### 2. Download a Model
Pull a model that you want to use with LGX:

```bash
ollama pull <model-name>
```

### 3. Run the Benchmark

Run the experiment using:

```bash
LGX_OLLAMA_URL=<your-ollama-url> poetry run python run_benchmark.py --model "<model-name>"
```

To replicate the experiments from the paper, use:

```bash
poetry run python run_benchmark.py --skip_ollama --model <llama3.1:8b | llama3.1:70b | gpt-oss:120b>
```

### 4. Using an External Ollama Instance

If you're connecting to an external Ollama server, include your API key:

```bash
LGX_OLLAMA_URL=<your-ollama-url> \
LGX_OLLAMA_KEY=<your-ollama-api-key> \
poetry run python run_benchmark.py --model "<model-name>"
```

### 5. Configure Request Timeout

You can control how long LGX waits for a response from Ollama before raising a timeout error by setting the `LGX_OLLAMA_TIMEOUT` environment variable.

```bash
LGX_OLLAMA_URL=<your-ollama-url> \
LGX_OLLAMA_TIMEOUT=<timeout-in-seconds> \
poetry run python run_benchmark.py --model "<model-name>"
```

For example:

```bash
LGX_OLLAMA_URL=http://localhost:11434 \
LGX_OLLAMA_TIMEOUT=120 \
poetry run python run_benchmark.py --model "<model-name>"
```

### 6. Results

The results of the experiments will be saved in the `experiment_results` directory, organized by model used and the test name (e.g `experiment_results/llama3.1-8b/test_name.json`).

---

### Notes
- If you are replicating the experiments, ensure to have cache_prompt.db in the same directory as run_benchmark.py, which contains the cached prompts for the experiments.
- Replace `<your-ollama-url>` with the URL where Ollama is running (e.g., `http://localhost:11434`).
- Replace `<model-name>` with the model you downloaded using `ollama pull`.
- Increase the timeout if you are using large models or running on limited hardware.