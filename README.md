# Jenosize Trend & Future Ideas Article Generator

A submission-ready Python prototype for **Option 1: Trend & Future Ideas Articles**.

This repo is intentionally split into **layers** so an interviewer can inspect each part independently:

1. **Layer 1 — Dataset preparation**
2. **Layer 2 — Fine-tuning**
3. **Layer 3 — RAG / source grounding**
4. **Layer 4 — Evaluation + simple auto-tuning**
5. **Layer 5 — FastAPI serving**

The implementation uses **Hugging Face** models for both generation and embeddings.

---

## 1) Model options

The project is designed around Hugging Face model IDs and supports a simple strategy choice:

### Prefer accuracy
- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Why: stronger overall instruction following and article quality
- Trade-off: larger memory footprint and slower inference
- Note: gated model license on Hugging Face

### Prefer speed
- `Qwen/Qwen2.5-3B-Instruct`
- Why: smaller model, easier local experimentation, faster fine-tuning and inference
- Trade-off: weaker long-form stability than 7B/8B models

### Prefer balance
- `Qwen/Qwen2.5-7B-Instruct`
- Why: good instruction following, strong long-text behavior, simpler licensing, practical for LoRA/QLoRA
- Recommended default for this prototype

### Embedding model
- `sentence-transformers/all-MiniLM-L6-v2`
- Why: simple, fast, stable for lightweight retrieval

---

## 2) Repo structure

```text
app/
  api/
    main.py
    routes/articles.py
    schemas/request.py
    schemas/response.py
  evaluation/
    metrics.py
    evaluator.py
    tuner.py
  rag/
    chunker.py
    embedder.py
    indexer.py
    retriever.py
    prompt_builder.py
  services/
    generator.py
    article_pipeline.py
  utils/
    text.py
  config.py
training/
  bootstrap_hf_dataset.py
  prepare_dataset.py
  train_finetune.py
  smoke_test_generation.py
Dockerfile
render.yaml
requirements.txt
README.md
```

---

## 3) Installation

Create a virtual environment and install the pinned dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Notes
- For Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

- If you do not want 4-bit quantization during fine-tuning, disable it in `app/config.py`.
- If you use `meta-llama/Meta-Llama-3.1-8B-Instruct`, authenticate with Hugging Face first:

```bash
huggingface-cli login
```

---

## 4) Model selection modes

The repo now supports both **CLI-based model selection** for training/smoke tests and **environment-variable-based model selection** for API deployment.

### CLI examples

Use the balanced default:

```bash
python training/train_finetune.py --model-strategy balance
```

Prefer speed:

```bash
python training/train_finetune.py --model-strategy speed
```

Prefer accuracy:

```bash
python training/train_finetune.py --model-strategy accuracy
```

Use an exact Hugging Face model ID:

```bash
python training/train_finetune.py --base-model-name Qwen/Qwen2.5-7B-Instruct
```

Override adapter output path:

```bash
python training/train_finetune.py --model-strategy balance --output-dir artifacts/qwen_balance_adapter
```

Smoke test with a speed-focused base model:

```bash
python training/smoke_test_generation.py --model-strategy speed --model-dir artifacts/model_adapter
```

### API / deployment examples

Use environment variables when serving the API:

```bash
BASE_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 \
MODEL_ADAPTER_DIR=artifacts/model_adapter \
uvicorn app.api.main:app --reload
```

You can also use strategy mode from env:

```bash
MODEL_STRATEGY=balance uvicorn app.api.main:app --reload
```

Supported environment variables:
- `MODEL_STRATEGY` = `accuracy` / `speed` / `balance`
- `BASE_MODEL_NAME` = exact Hugging Face model ID
- `EMBEDDING_MODEL_NAME` = embedding model ID
- `MODEL_ADAPTER_DIR` = adapter folder
- `FINETUNED_MODEL_DIR` = alias for `MODEL_ADAPTER_DIR`
- `USE_4BIT` = `true` / `false`

## 5) Data sources used in this prototype

This prototype now supports **automatic dataset bootstrap from Hugging Face** before `prepare_dataset.py` runs.

### Auto-loaded datasets
1. `Alaamer/medium-articles-posts-with-content`
   - used as the main long-form article source
   - filtered for technology, digital transformation, AI, innovation, customer experience, and other business-trend themes
   - suitable for this assignment because it provides article-length business/technology content rather than short social-copy text

2. `danidanou/Reuters_Financial_News`
   - used as a source-grounded business/news supplement
   - useful for creating `source_content` fields and for covering finance and business strategy topics
   - suitable for this assignment because Option 1 accepts website/document source content as an input

### Why these datasets fit the assignment
The assignment is about **business trend / future ideas articles**, not generic chatbot dialog or ad-copy generation. These two datasets give a practical prototype mix of:
- long-form article structure
- business and technology topics
- source-grounded business content

### Style note
For a stronger final submission, you can still manually review a few Jenosize Ideas articles as a **style reference** when checking generated outputs, but the automatic bootstrap in this repo loads data only from Hugging Face.

---

## 6) Layer-by-layer execution

# Layer 1A — Bootstrap processed dataset from Hugging Face

## Purpose
Automatically download and filter seed training data from Hugging Face, then save it as a processed CSV suitable for the next layer.

## Run
```bash
python training/bootstrap_hf_dataset.py
```

### Useful options
```bash
python training/bootstrap_hf_dataset.py --medium-limit 60 --reuters-limit 30
python training/bootstrap_hf_dataset.py --output data/processed/article_training_source.csv
python training/bootstrap_hf_dataset.py --no-streaming
```

## What it does
- downloads/streams data from Hugging Face datasets
- filters for business / technology / future-trend relevance
- normalizes fields into a unified schema
- creates `data/processed/article_training_source.csv`

## Output schema
The bootstrap layer writes a CSV with these fields:
- `topic_category`
- `industry`
- `target_audience`
- `seo_keywords`
- `source_content`
- `article_title`
- `article_body`
- `desired_length`
- `source_dataset`

## Interviewer explanation
This is the first visible **data acquisition + preprocessing layer**. It shows that the project does not assume a manually prepared dataset from the start.

---

# Layer 1B — Convert processed CSV into chat-style fine-tuning JSONL

## Purpose
Convert the processed CSV into a **chat-style supervised fine-tuning dataset**.

## Expected input
`training/bootstrap_hf_dataset.py` creates this automatically by default:

```text
data/processed/article_training_source.csv
```

You may also provide your own processed file with the same required columns.

Required columns:
- `topic_category`
- `industry`
- `target_audience`
- `source_content`
- `seo_keywords`
- `article_body`

Optional columns:
- `article_title`
- `desired_length`

## Run
```bash
python training/prepare_dataset.py
```

### Useful options
```bash
python training/prepare_dataset.py --source-path data/processed/article_training_source.csv
python training/prepare_dataset.py --output-dir data/training
```

## What it does
- validates columns
- removes empty rows
- cleans source and target text
- converts each sample into a `messages` format
- splits into train/validation sets
- writes:
  - `data/training/train.jsonl`
  - `data/training/val.jsonl`

## Interviewer explanation
This layer is the **training-data transformation layer**. It turns processed article records into the exact chat/instruction format expected by the fine-tuning script.

---

# Layer 2 — Fine-tuning

## Purpose
Run **supervised fine-tuning with LoRA/QLoRA** on the Hugging Face base model.

## Default behavior
The default model strategy is set in `app/config.py`.
Recommended default:
- `Qwen/Qwen2.5-7B-Instruct`

## Run
```bash
python training/train_finetune.py --model-strategy balance
```

Alternative runs:
```bash
python training/train_finetune.py --model-strategy speed
python training/train_finetune.py --model-strategy accuracy
python training/train_finetune.py --base-model-name Qwen/Qwen2.5-7B-Instruct
```

## What it does
- loads tokenizer and base model from Hugging Face
- loads `train.jsonl` and `val.jsonl`
- applies chat template
- configures LoRA target modules
- fine-tunes the model
- saves artifacts to:

```text
artifacts/model_adapter/
```

## Important knobs
Open `app/config.py` and tune:
- `base_model_name`
- `use_4bit`
- `num_train_epochs`
- `learning_rate`
- `max_seq_length`

## Interviewer explanation
This layer satisfies the assignment requirement that the prototype must **fine-tune a pre-trained model**, not just call a hosted model with prompting.

---

# Layer 3 — Smoke test generation

## Purpose
Verify that the saved adapter/model can generate a grounded article-like response.

## Run
```bash
python training/smoke_test_generation.py --model-strategy balance
```

Alternative runs:
```bash
python training/smoke_test_generation.py --model-strategy speed
python training/smoke_test_generation.py --base-model-name Qwen/Qwen2.5-7B-Instruct --model-dir artifacts/model_adapter
```

## What it does
- loads the saved adapter if present
- falls back to the base model if no adapter exists
- builds a small article-generation prompt
- prints generated text

## Interviewer explanation
This is a minimal sanity-check layer between training and API deployment. It makes model loading errors visible early.

---

# Layer 4 — RAG / source grounding

## Purpose
Convert user-provided `source_content` into **retrieved model-ready context**.

## Flow
1. clean text
2. chunk text
3. embed chunks
4. build FAISS index
5. retrieve top-k chunks
6. build grounded prompt

## Main modules
- `app/rag/chunker.py`
- `app/rag/embedder.py`
- `app/rag/indexer.py`
- `app/rag/retriever.py`
- `app/rag/prompt_builder.py`

## Interviewer explanation
This is **lightweight RAG**, not a full enterprise retrieval platform. It is intentionally simple because the assignment is short and the focus is still on fine-tuning plus deployment.

---

# Layer 5 — Evaluation + simple tuning

## Purpose
Measure article quality with simple prototype-safe metrics and do **one small retry** if quality is weak.

## Metrics
- `keyword_coverage`
- `length_compliance`
- `structure_score`
- `groundedness_score`
- `readability_score`

## Tuning behavior
If the first generation score is low, the pipeline may:
- lower `temperature`
- increase `retrieval_top_k`
- increase `max_new_tokens`
- slightly raise `repetition_penalty`

This is intentionally heuristic and limited to **one retry** to avoid over-engineering.

## Interviewer explanation
This layer shows optimization awareness without building a heavy automated tuning system.

---

# Layer 6 — API serving

## Purpose
Expose the generation system through FastAPI.

## Run
```bash
uvicorn app.api.main:app --reload
```

## Endpoints
### Health check
```text
GET /health
```

### Article generation
```text
POST /v1/articles/generate
```

### Example request
```bash
curl -X POST "http://127.0.0.1:8000/v1/articles/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "topic_category": "Artificial Intelligence in Customer Experience",
    "industry": "Retail Banking",
    "target_audience": "Business Executives",
    "source_content": "Retail banks are increasingly using AI to personalize customer interactions, reduce call center load, speed up onboarding, improve fraud detection, and offer proactive product recommendations. However, many banks still struggle to unify customer data across channels, creating fragmented experiences.",
    "seo_keywords": ["AI banking trends", "future of customer experience", "retail banking innovation"],
    "article_length": "900-1200 words",
    "top_k": 4
  }'
```

## Response shape
The API returns:
- generated article text
- retrieved chunks
- evaluation scores
- final generation config used

## Interviewer explanation
This layer demonstrates that the prototype is deployable and that retrieval, generation, and evaluation are orchestrated through a clean API surface.

---

## 5) Suggested interview walkthrough

If you need to explain the system quickly, use this order:

1. **Data pipeline** prepares training data.
2. **Fine-tuning layer** teaches style and structure.
3. **RAG layer** grounds output on user-provided source content.
4. **Evaluation layer** measures quality and performs one simple retry.
5. **FastAPI layer** exposes the end-to-end flow as a prototype service.

---

## 6) Practical notes

- This is intentionally a **prototype**, not a production RAG platform.
- Retrieval is rebuilt per request from the user’s source content to keep the design easy to reason about.
- The evaluation/tuning loop is limited to one retry by design.
- The default balance model is `Qwen/Qwen2.5-7B-Instruct`.

---

## 7) Common run order

### Fastest path for a reviewer
```bash
python training/smoke_test_generation.py
uvicorn app.api.main:app --reload
```

### Full path
```bash
python training/prepare_dataset.py
python training/train_finetune.py
python training/smoke_test_generation.py
uvicorn app.api.main:app --reload
```

---

## 8) What each layer proves

- **Layer 1** proves data engineering
- **Layer 2** proves fine-tuning
- **Layer 3** proves model artifact usability
- **Layer 4** proves source grounding
- **Layer 5** proves simple optimization awareness
- **Layer 6** proves deployability

## requirements.txt
This repository now includes a `requirements.txt` file so the project can be installed locally or inside Docker with one command:

```bash
pip install -r requirements.txt
```

## Docker
Build the image:

```bash
docker build -t jenosize-article-generator .
```

Run the container locally:

```bash
docker run --rm -p 8000:8000   -e PORT=8000   -e BASE_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct   -e EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2   -e FINETUNED_MODEL_DIR=artifacts/qwen2_5_7b_jenosize_lora   jenosize-article-generator
```

Open:
- API health check: `http://localhost:8000/health`
- Swagger docs: `http://localhost:8000/docs`

## Render Deployment
This repository includes a `render.yaml` blueprint for simple web deployment on Render.

### What the assignment requires
The assignment explicitly requires deployment via an API using FastAPI or Flask, but it does **not** explicitly require a specific cloud provider or a mandatory cloud deployment command. A hosted prototype link is still useful for submission.

### Deploy on Render
1. Push this repository to GitHub.
2. Create a new Render Web Service from the repo.
3. Render will detect `render.yaml` and use the included Docker deployment configuration.
4. Keep `autoDeploy: false` if you want manual control during the take-home test.
5. After deploy, test:
   - `/health`
   - `/docs`
   - `POST /v1/articles/generate`

### Environment variables in render.yaml
- `PORT=10000`
- `BASE_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct`
- `EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2`
- `FINETUNED_MODEL_DIR=artifacts/qwen2_5_7b_jenosize_lora`

### Practical note for interview/demo
For hosted demo stability, prefer a lighter model in the deployment environment.
- prefer **speed**: `Qwen/Qwen2.5-3B-Instruct`
- prefer **balance**: `Qwen/Qwen2.5-7B-Instruct`
- prefer **accuracy**: `meta-llama/Llama-3.1-8B-Instruct`

## Optional model choices
The project is intentionally designed so the Hugging Face base model can be swapped.

### Prefer speed
- `Qwen/Qwen2.5-3B-Instruct`
- Good for a lighter demo deployment and faster iteration.

### Prefer balance
- `Qwen/Qwen2.5-7B-Instruct`
- Best default choice for this assignment because it balances article quality, instruction following, and deployment practicality.

### Prefer accuracy
- `meta-llama/Llama-3.1-8B-Instruct`
- Good when the goal is better long-form generation quality and you have slightly more compute budget.
