# GenDLN

This is the repository of the paper [**GenDLN: Evolutionary Algorithm-Based Stacked LLM Framework for Joint Prompt Optimization**](https://aclanthology.org/2025.acl-srw.92/) by Pia Chouayfati, Niklas Herbster, Ábel Domonkos Sáfrán, and Matthias Grabmair.

The code implements a genetic algorithm-based framework for optimizing prompts in a two-layer LLM architecture, designed to efficiently utilize commercial LLM APIs for tasks like clause classification and paraphrase detection.


## Setup

### 1) Prerequisites
- Python 3.10 or newer
- Mistral API access (the GA/LLM pipeline in `genetic_dln` uses Mistral)

### 2) Clone and install
```bash
git clone https://github.com/piachouaifaty/LegalNLPLab.git
cd LegalNLPLab
python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Configure environment variables
Create a `.env` file in the repository root (same directory as this README). The GA LLM interface creates multiple Mistral clients to scale in parallel; provide as many keys as you have.

At minimum, set these (recommended: provide v1..v10; optionally v11..v20 for fallback/backup):

```dotenv
# Mistral primary workspaces (used for parallel inference)
MISTRAL_API_KEY_V1=...
MISTRAL_API_KEY_V2=...
MISTRAL_API_KEY_V3=...
MISTRAL_API_KEY_V4=...
MISTRAL_API_KEY_V5=...
MISTRAL_API_KEY_V6=...
MISTRAL_API_KEY_V7=...
MISTRAL_API_KEY_V8=...
MISTRAL_API_KEY_V9=...
MISTRAL_API_KEY_V10=...

# Optional backups (used automatically when primaries rate-limit/fail)
MISTRAL_API_KEY_V11=...
MISTRAL_API_KEY_V12=...
MISTRAL_API_KEY_V13=...
MISTRAL_API_KEY_V14=...
MISTRAL_API_KEY_V15=...
MISTRAL_API_KEY_V16=...
MISTRAL_API_KEY_V17=...
MISTRAL_API_KEY_V18=...
MISTRAL_API_KEY_V19=...
MISTRAL_API_KEY_V20=...
```

Notes:
- The current `genetic_dln/src/models/llm.py` initializes up to 10 primary clients and 10 backups. If you only have a few keys, you may need to reduce parallelism in your run configuration (see hyperparameters), and/or adapt the client initialization code accordingly.
- Logs are written under `logs/`.


### Organization

    GenDLN/
    ├── datasets/
    │   ├── claudette
    │   ├── llm_safe_mrpc
    ├── ga_post_preocessing_R/
    │   ├── GA_Runs.Rmd      # R Markdown file for generating reports
    │   ├── GA_Runs.html     # Rendered example on dummy_logs
    │   ├── R_helpers      # Normalization, parsing, processing, plotting... scripts
    │   ├── dummy_logs     # example logs to run the GA_Runs.Rmd
    ├── genetic_dln/       # GenDLN framework
    ├── baselines/       # some of the baselines we ran


---

## The genetic_dln directory

`genetic_dln/` contains the core prompt-evolution framework.

- `genetic_dln/src/ga_runner.py`
  - Entry point to run a single GA experiment. Loads `data/hyperparameters.yaml` and orchestrates a full run.
- `genetic_dln/src/evolutionary_algorithms/ga_engine.py`
  - GA engine (selection, crossover, mutation, replacement, logging, early stopping).
- `genetic_dln/src/evolutionary_algorithms/genetic_operations/`
  - `replacement.py` and related operators (selection, crossover, mutation, fitness) that form the GA loop.
- `genetic_dln/src/dln/gen_dln.py`
  - Two-layer LLM classifier (Layer 1 feature extraction → Layer 2 classification with few-shots).
  - Handles batching, concurrency, evaluation, and post-processing.
- `genetic_dln/src/models/`
  - `llm.py`: Mistral-based LLM client with multiple API keys for parallel workspaces and retry/backoff.
  - `base_gen_dln_llm.py`: Interface for LLMs.
  - `rate_limiter.py`: Simple per-request rate limiter.
- `genetic_dln/src/prompt_builder/`
  - `base_prompt_builder.py`: Interface for prompt construction.
  - `prompt_builder.py`: Concrete implementation for building Layer 1 and Layer 2 messages from templates and few-shots.
- `genetic_dln/src/post_processor/post_processor.py`
  - Normalizes and interprets model outputs (e.g., JSON parsing, class label mapping).
- `genetic_dln/src/input_loader/input_loader.py`
  - Loads hyperparameters, templates, few-shots, and data splits.
- `genetic_dln/src/constants/constants.py`
  - Global paths and environment variable loading.


Data and configs:
- `genetic_dln/data/hyperparameters.yaml`
  - Main configuration file for GA runs (population size, generations, mutation/crossover rates, selection strategy, temperatures, early stopping, workspaces, etc.).
- `genetic_dln/data/base_prompts/`
  - Base prompt templates per task type (e.g., `binary/`, `multi_label/`).
  - Files like `prompt_01_template.yaml` and `prompt_02_template.yaml`.
- `genetic_dln/data/few_shots/`
  - Few-shot examples for Layer 2 classification (e.g., `prompt_02_few_shots.yaml`).
- `genetic_dln/data/score_cache/`
  - Cache for fitness/evaluation scores to avoid re-scoring identical prompts.

---

### Configure the `Task` class in `ga_runner.py`

In the file `genetic_dln/src/ga_runner.py`, locate the `Task` class instantiation:

```python
TASK = Task(
        layer_1_system_prompt_path="",
        layer_2_system_prompt_path="",
        layer_2_few_shots_path="",
        layer_1_initial_prompts_path="",
        layer_2_initial_prompts_path="",
        train_dataset_path="",
        val_dataset_path="",
)
```

Fill in the paths to the required files and datasets:  

- **layer_1_system_prompt_path**: Path to the Layer 1 system prompt template  
  *(e.g., `genetic_dln/data/base_prompts/prompt_01_template.yaml`)*.

- **layer_2_system_prompt_path**: Path to the Layer 2 system prompt template  
  *(e.g., `genetic_dln/data/base_prompts/prompt_02_template.yaml`)*.

- **layer_2_few_shots_path**: Path to the few-shot examples for Layer 2 classification  
  *(e.g., `genetic_dln/data/few_shots/prompt_02_few_shots.yaml`)*.

- **layer_1_initial_prompts_path**: Path to the initial prompts for Layer 1 (if applicable).

- **layer_2_initial_prompts_path**: Path to the initial prompts for Layer 2 (if applicable).

- **train_dataset_path**: Path to the training dataset  
  *(e.g., `genetic_dln/datasets/claudette/train.json`)*.

- **val_dataset_path**: Path to the validation dataset  
  *(e.g., `genetic_dln/datasets/claudette/val.json`)*.

Ensure these paths point to the correct files in your project directory before running the genetic algorithm.


## Quickstart: Execute a GA run

From the repo root, ensure your Python path includes the project (or run with `-m`):

```bash
python genetic_dln/src/ga_runner.py

```

Note: We provide an "LLM-Safe" MRPC dataset. Details can be found in [Appendix P](https://aclanthology.org/2025.acl-srw.92.pdf) of the paper.

## Analyze the Output

We provide a post-analysis suite in R for plotting and result analysis,
Refer to `ga_post_processing/readme.md` (https://github.com/piachouaifaty/GenDLN/blob/main/ga_post_processing_R/readme.md) for information on how to use and run it.


#### Citation (BibTex)

    @inproceedings{chouayfati-etal-2025-gendln,
    title = "GenDLN: Evolutionary Algorithm-Based Stacked {LLM} Framework for Joint Prompt Optimization",
    author = "Chouayfati, Pia  and
      Herbster, Niklas  and
      S{\'a}fr{\'a}n, {\'A}bel Domonkos  and
      Grabmair, Matthias",
    editor = "Zhao, Jin  and
      Wang, Mingyang  and
      Liu, Zhu",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-srw.92/",
    pages = "1171--1212",
    ISBN = "979-8-89176-254-1",
    abstract = "With Large Language Model (LLM)-based applications becoming more common due to strong performance across many tasks, prompt optimization has emerged as a way to extract better solutions from frozen, often commercial LLMs that are not specifically adapted to a task. LLM-assisted prompt optimization methods provide a promising alternative to manual/human prompt engineering, where LLM ``reasoning'' can be used to make them optimizing agents. However, the cost of using LLMs for prompt optimization via commercial APIs remains high, especially for heuristic methods like evolutionary algorithms (EAs), which need many iterations to converge, and thus, tokens, API calls, and rate-limited network overhead. We propose GenDLN, an open-source, efficient genetic algorithm-based prompt pair optimization framework that leverages commercial API free tiers. Our approach allows teams with limited resources (NGOs, non-profits, academics, ...) to efficiently use commercial LLMs for EA-based prompt optimization. We conduct experiments on CLAUDETTE for legal terms of service classification and MRPC for paraphrase detection, performing in line with selected prompt optimization baselines, at no cost."}

