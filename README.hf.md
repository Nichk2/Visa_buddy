## VisaBuddy — Canada Immigration Assistant

### A retrieval‑augmented, QLoRA‑fine‑tuned LLM assistant for Canadian immigration guidance

---

### Badges (text)

- **License**: MIT  
- **Model (Hugging Face)**: `<your-hf-username>/<your-model-repo>`  
- **Tech**: `LLM • QLoRA • RAG • FastAPI • React • TypeScript • TailwindCSS • ChromaDB`

---

## Overview

VisaBuddy — Canada Immigration Assistant is an end‑to‑end system that combines a custom fine‑tuned LLM with a modern web UI to help users explore Canadian immigration programs in a conversational way.

The assistant:

- retrieves information from cleaned IRCC (Immigration, Refugees and Citizenship Canada) documents,
- runs a **RAG (Retrieval‑Augmented Generation)** pipeline over a **ChromaDB** vector store,
- and answers questions through a **FastAPI** backend and **React** chat frontend.

> **Important:** VisaBuddy is **not** an official source of immigration advice and is **not** a substitute for a licensed immigration consultant or lawyer. Always verify final decisions and requirements directly with official IRCC resources.

---

## Architecture

High‑level flow from user to model:

```text
┌───────────┐      ┌──────────────┐      ┌───────────┐      ┌──────────────┐      ┌───────────────┐
│  Browser  │ ───▶ │  Frontend    │ ───▶ │   API     │ ───▶ │   Model +    │ ───▶ │  Vector DB    │
│ (User UI) │      │ (React/TS)   │      │(FastAPI)  │      │   RAG Core   │      │  (ChromaDB)   │
└───────────┘      └──────────────┘      └───────────┘      └──────────────┘      └───────────────┘
       ▲                                                                                     │
       └───────────────────────── streamed / batched responses ◀─────────────────────────────┘
```

Expanded data path:

```text
User Question
   │
   ▼
React Chat UI
   │  (POST /v1/chat)
   ▼
FastAPI Backend (Colab, behind Cloudflare Tunnel)
   │
   ├─► RAG Retriever
   │     ├─ Chunked IRCC docs
   │     ├─ Embeddings (Sentence Transformers)
   │     └─ ChromaDB vector search
   │
   ├─► Prompt Builder (question + retrieved context)
   │
   └─► QLoRA‑fine‑tuned LLM (4‑bit quantized)
           │
           ▼
      Formatted Answer → sent back to UI
```

---

## Features

### AI / Modeling

- **Custom LLM fine‑tuned with QLoRA**
  - Built on a modern base model (e.g. LLaMA 3 / Mistral / Qwen; configurable in code).
  - 4‑bit quantization for efficient inference.
  - Fine‑tuned on an **instruction JSONL dataset** derived from IRCC content.
- **RAG (Retrieval‑Augmented Generation)**
  - Cleans IRCC HTML + PDF documents.
  - Chunks long texts into overlapping segments.
  - Uses a **Sentence Transformers** embedding model.
  - Stores embeddings and metadata in **ChromaDB**.
  - Retrieves top‑k relevant chunks per query and injects them into prompts.
- **Evaluation**
  - Training **loss curve** and **perplexity**.
  - Retrieval metrics: **precision@k**, **recall@k**, **MRR**.
  - Answer‑quality metrics: **factuality**, **hallucination rate**, **groundedness**, **context utilization**.
  - Notebook dashboards for quick visual inspection.

### Backend (FastAPI, Colab)

- **FastAPI server** running in **Google Colab**.
- Loads a **model ZIP** from **Google Drive** (or other storage) and extracts it.
- Uses **PEFT** LoRA adapters on top of the base model.
- Exposes an inference endpoint (e.g. `POST /v1/chat`) for the frontend.
- Tunnelled to the public internet via **Cloudflare Tunnel**.

### Frontend (React + TypeScript)

- Modern **chat interface** tailored to visa/immigration workflows.
- Built with **React**, **TypeScript**, and **TailwindCSS**.
- Configurable `VITE_API_URL` pointing to the backend tunnel URL.
- Multiple conversation threads, persistent history and smooth animations.

---

## Tech Stack

### Frontend

- **React**, **TypeScript**, **Vite**
- **TailwindCSS** for styling
- Optional animation and icon libraries

### Backend

- **Python 3.x**
- **FastAPI** (or compatible ASGI framework)
- **Uvicorn** for serving
- Runs inside **Google Colab**
- **Cloudflare Tunnel** for secure public access

### AI / ML

- **Transformers** + **PEFT**
- **QLoRA** (LoRA on quantized weights, 4‑bit)
- **Sentence Transformers** for embeddings
- **ChromaDB** as vector database

### Infrastructure / Storage

- **Google Drive** for model ZIP storage
- **Google Colab** GPU runtime
- **Cloudflare Tunnel** for public HTTPS endpoint

---

## How the Model Works

### Data & Preprocessing

- Public IRCC documents (HTML and PDF) are:
  - crawled or downloaded,
  - parsed into raw text,
  - cleaned (removing navigation, styling, boilerplate),
  - normalized (whitespace, encoding artifacts).

### Chunking

- Long documents are split into **overlapping chunks**:
  - Sentence‑aware segmentation.
  - A target chunk size (e.g. 800–1,000 tokens or words).
  - Overlap between chunks (e.g. 150–200 tokens/words) to preserve context across boundaries.

### Embeddings & Vector Store

- Each chunk is embedded using a **Sentence Transformers** model.
- Embeddings + metadata (source URL, section, document ID) are stored in **ChromaDB**.
- At query time:
  - the user question is embedded,
  - **k** nearest chunks are retrieved,
  - these chunks become the **context** for RAG.

### QLoRA Fine‑Tuning

- A base model such as **LLaMA 3**, **Mistral**, or **Qwen** (configurable) is used.
- **QLoRA** is applied:
  - The base model is loaded in **4‑bit** quantized mode.
  - Low‑rank adapters (LoRA) are trained on top of frozen base weights.
  - Training uses an instruction dataset in JSONL with fields like:
    - `instruction`, `input`, `output`, plus metadata.
- Benefits:
  - Strong performance with limited GPU memory.
  - Safety: it’s easy to discard or retrain adapters without touching base weights.

### Inference Flow

1. User asks a question via the web UI.
2. The frontend sends a request to the FastAPI endpoint.
3. Backend:
   - embeds the question,
   - retrieves top‑k context chunks from ChromaDB,
   - builds a prompt that includes:
     - system instructions,
     - retrieved context,
     - the user question.
4. The QLoRA‑fine‑tuned model generates an answer.
5. The answer is returned to the frontend and displayed in the chat.

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

---

### 2. Frontend Setup (React + Vite)

#### 2.1 Install Dependencies

```bash
cd frontend  # or project root if frontend is at top-level
npm install
```

#### 2.2 Environment Variables

Create a `.env` (or `.env.local`) file in the frontend root:

```bash
VITE_API_URL=<your-cloudflare-tunnel-url>
```

Examples:

- `VITE_API_URL=https://your-tunnel-subdomain.trycloudflare.com`

This URL should point to the public endpoint that fronts your FastAPI server in Colab.

#### 2.3 Run the Dev Server

```bash
npm run dev
```

Open the printed URL in your browser (typically `http://localhost:5173`) to use VisaBuddy.

---

### 3. Backend Setup (Google Colab + FastAPI)

The backend is intended to run inside a **Google Colab** notebook.

#### 3.1 Upload Model ZIP to Google Drive

1. Train or download the QLoRA adapters and tokenizer.
2. Zip the model folder into something like:
   - `visa-buddy-model.zip`
3. Upload the ZIP to Google Drive, for example:
   - `MyDrive/visa_buddy/visa-buddy-model.zip`

#### 3.2 Mount Google Drive in Colab

In your Colab notebook:

```python
from google.colab import drive
drive.mount("/content/drive")

MODEL_ZIP_PATH = "/content/drive/MyDrive/visa_buddy/visa-buddy-model.zip"
LOCAL_MODEL_DIR = "/content/visa_buddy_model"
```

#### 3.3 Unzip and Load the Model

```python
import os
import zipfile

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zf:
    zf.extractall(LOCAL_MODEL_DIR)
```

Then load the base model and LoRA adapters with **Transformers** + **PEFT**:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL_NAME = "<base-model-name>"  # e.g. meta-llama/Meta-Llama-3-8B-Instruct
ADAPTER_PATH = LOCAL_MODEL_DIR         # folder with adapter_config / adapter_model

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
```

#### 3.4 FastAPI Server

Create a simple FastAPI app in the notebook:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/v1/chat")
def chat(req: ChatRequest):
    # TODO: run RAG retrieval and LLM generation here
    # response_text = generate_answer(req.message)
    response_text = "This is a placeholder response."
    return JSONResponse({"answer": response_text})

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Start the server (in a background cell or with `nest_asyncio` / separate process).

#### 3.5 Expose Backend via Cloudflare Tunnel

1. Install cloudflared in Colab:

   ```bash
   !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   !sudo dpkg -i cloudflared-linux-amd64.deb
   ```

2. Start a tunnel targeting the FastAPI port (8000). Follow Cloudflare’s docs to authenticate and run:

   ```bash
   !cloudflared tunnel --url http://localhost:8000
   ```

3. Copy the **public HTTPS URL** printed by `cloudflared` (e.g. `https://<random>.trycloudflare.com`) and set it as `VITE_API_URL` in your frontend `.env`.

---

## Model Usage

### Direct Usage with Transformers

If you want to use the model directly (outside the web app), you can:

1. Download and unzip the QLoRA model ZIP.
2. Load it using Transformers and PEFT:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL_NAME = "<base-model-name>"
MODEL_DIR = "<path-to-unzipped-model-dir>"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, MODEL_DIR)
model.eval()

def generate(prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Calling the Inference Endpoint

Assuming your FastAPI backend exposes `POST /v1/chat`:

```bash
curl -X POST "<api-url>/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How does Express Entry work?"}'
```

Response (example):

```json
{
  "answer": "Express Entry is an online system used to manage permanent residence applications for skilled workers..."
}
```

---

## Metrics & Evaluation

The project uses several metrics to understand both retrieval quality and answer quality.

### Retrieval Metrics

- **Precision@k**
  - Definition: fraction of the top‑k retrieved documents that are relevant.
  - Interpretation: “Of the first k chunks we retrieved, how many were actually useful for answering the question?”
  - High precision@k ⇒ the system rarely brings in off‑topic or noisy context.

- **Recall@k**
  - Definition: fraction of all relevant documents that are retrieved in the top‑k.
  - Interpretation: “Out of everything that would have been helpful, how much did we actually retrieve?”
  - High recall@k ⇒ the system is good at covering all necessary evidence.

- **MRR (Mean Reciprocal Rank)**
  - Definition: average of `1 / rank_of_first_relevant_document` across queries.
  - Interpretation: rewards systems that put at least one truly relevant document **very high** in the ranking.
  - MRR close to 1 means relevant context usually appears at rank 1.

### Answer‑Quality Metrics

- **Hallucination Rate**
  - Definition: proportion of answers that contain statements not supported by the retrieved context or known ground truth.
  - Lower is better.
  - For immigration use cases, hallucination control is critical; even small rates need careful mitigation.

- **Factuality**
  - Definition: proportion of answers whose key claims match external or reference truth (e.g., official IRCC rules at the time of evaluation).
  - Higher is better; near‑100% is expected when dealing with simple, well‑covered queries.

- **Groundedness**
  - Definition: how consistently answers are explicitly supported by retrieved context.
  - Can be approximated via human or automated checks:
    - Does each factual statement tie back to at least one retrieved snippet?

- **Context Utilization**
  - Definition: fraction of retrieved context that is actually used in the final answer.
  - High utilization means the model is not just “seeing” context but integrating it.

### Training Metrics

- **Loss Curve**
  - Usually plotted per training step or epoch.
  - Decreasing loss indicates the model is learning the supervised signal (instruction → answer).
  - A stable, flattening curve near the end suggests convergence.
  - Rising loss or strong oscillations can indicate:
    - learning rate issues,
    - noisy labels,
    - overfitting or underfitting.

In the VisaBuddy notebooks, you’ll find plots that make these metrics easy to interpret for non‑experts (e.g. annotated final loss, sample points per epoch, etc.).

---

## Folder Structure (Example)

Below is a recommended high‑level layout; adapt to your actual repo:

```text
.
├─ README.md                    # General project README (existing)
├─ README.hf.md                 # This Hugging Face–style README
├─ frontend/
│  ├─ src/
│  │  ├─ components/            # Chat UI, modals, layout
│  │  ├─ hooks/                 # API hooks, state management
│  │  ├─ styles/                # Tailwind / global styles
│  │  └─ main.tsx               # React entrypoint
│  ├─ index.html
│  ├─ package.json
│  └─ vite.config.ts
├─ backend/
│  ├─ app.py                    # FastAPI app (endpoints, RAG pipeline)
│  ├─ rag/
│  │  ├─ retriever.py           # ChromaDB integration, embeddings
│  │  ├─ chunking.py            # Text chunking utilities
│  │  └─ preprocessing.py       # HTML/PDF cleaning
│  ├─ models/
│  │  ├─ load_model.py          # QLoRA loading helpers
│  │  └─ inference.py           # generation utilities
│  └─ requirements.txt
├─ notebooks/
│  ├─ 01_data_preparation.ipynb
│  ├─ 02_rag_and_embeddings.ipynb
│  ├─ 03_qlora_training.ipynb
│  ├─ 04_evaluation_metrics.ipynb
│  └─ 05_demo_and_analysis.ipynb
├─ data/
│  ├─ raw/                      # Raw HTML/PDFs from IRCC
│  ├─ processed/                # Clean, normalized text
│  ├─ chunks/                   # Chunked text files
│  ├─ embeddings/               # Embedding caches (optional)
│  └─ datasets/                 # JSONL instruction datasets
└─ models/
   ├─ visa-buddy-model.zip      # Compressed QLoRA model
   └─ visa-buddy/               # Extracted model directory
```

---

## Demo

> **Demo GIF / Screenshot Placeholder**
>
> - Add a short GIF of the chat interface in action.
> - Example: user asks about Express Entry eligibility and the assistant answers with grounded, step‑by‑step guidance.
>
> Suggested filename: `docs/demo.gif`

---

## Limitations & Responsible Use

- **Not official advice**
  - VisaBuddy is a technical demo and educational tool.
  - It should **not** be used as a sole source for immigration decisions.

- **Data freshness**
  - Immigration policies change frequently.
  - The model and underlying documents may not reflect the latest IRCC updates.

- **Hallucinations**
  - Despite RAG and evaluation metrics, the model can still:
    - invent rules,
    - misinterpret edge cases,
    - or omit important conditions.

- **Bias and fairness**
  - Training data and model behavior may contain hidden biases.
  - Always apply human judgment, especially for high‑stakes decisions.

> **Recommendation:** For real immigration cases, cross‑check all advice with official IRCC pages and, where appropriate, a licensed immigration professional.

---

## License

This project is released under the **MIT License**.

```text
MIT License

Copyright (c) <year> <your-name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contributing

- **Bug reports / issues**
  - Use GitHub Issues to report bugs, request features, or propose improvements.
- **Pull requests**
  - Fork the repository.
  - Create a feature branch: `git checkout -b feature/my-improvement`.
  - Commit your changes with clear messages.
  - Open a pull request with a short description and context.

Please ensure that:

- You do not add private or confidential data.
- Any new evaluation or benchmarks are explained clearly in the PR description.

---

## Acknowledgements

- **IRCC (Immigration, Refugees and Citizenship Canada)** for public documentation that informs the knowledge base.  
  - Only publicly available pages and documents are used.
  - No private or confidential datasets are included.
- Open‑source ecosystems around:
  - **Transformers**, **PEFT**, **Sentence Transformers**
  - **FastAPI**, **ChromaDB**
  - **React**, **TypeScript**, **TailwindCSS**

VisaBuddy builds on this open ecosystem to demonstrate how a responsible, retrieval‑augmented immigration assistant can be implemented end‑to‑end.


