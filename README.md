## Visa Buddy – Canadian Immigration RAG + QLoRA Assistant

Visa Buddy is a full‑stack, UX‑focused Canadian immigration assistant.  
It combines:

- **Modern React + Tailwind UI** (chat experience, multi‑chat history, link configuration)
- **RAG pipeline** (crawler → cleaning → chunking → embeddings → retrieval)
- **QLoRA/LoRA fine‑tuning** on a high‑quality synthetic dataset
- **Evaluation and metrics dashboards** (loss, perplexity, RAG relevance scores)

The front‑end talks to a **Colab backend** (or other Python host) via a **Cloudflare tunnel**.  
You run the model in Colab, expose it through Cloudflare, then paste the tunnel URL into Visa Buddy’s UI.

---

## 1. Project Structure

- **Frontend (React + Vite + Tailwind)**
  - `index.html` – Vite entry
  - `src/main.tsx` / `src/App.tsx` – app bootstrap and routing
  - `src/components/ChatInterface.tsx` – main chat UI, multi‑chat, loading states
  - `src/components/SideMenu.tsx`, `ConfirmationModal.tsx`, `LinkModal.tsx` – UX components
  - `src/hooks/useVisaBuddy.ts` – manages connection state and sends messages to the backend URL
  - `src/services/visaBuddyApi.ts` – API client abstraction for the chat backend
  - `src/types/visa_buddy.ts` – shared types for messages and chats
  - `src/utils/dateFormatter.ts` – helper(s) for date display

- **Notebooks & Model Pipeline (Python)**
  - `src/notebooks/Visa Buddy jupyter with outputs and measures.ipynb`  
    End‑to‑end pipeline: crawl → clean → chunk → embed → build SFT dataset → QLoRA training → inference → evaluation + metrics dashboards.
  - `src/notebooks/Visa Buddy jupyter clean.ipynb`  
    Cleaned version of the main pipeline (no outputs).
  - `src/notebooks/visa_buddy_canada google colab backend and model.ipynb`  
    Colab‑oriented backend + model loading notebook (serving via HTTP).
  - Local data folder used by the notebooks (created at runtime): `./visa_buddy_data/…`

---

## 2. Tech Stack

- **Frontend**
  - React 19, TypeScript, Vite
  - Tailwind CSS 4
  - Framer Motion / Motion for animations
  - `lucide-react` icons

- **Backend / Modeling (notebooks)**
  - Crawler & preprocessing: `requests`, `beautifulsoup4`, `trafilatura`, `pdfplumber`, `nltk`, `re`, `pandas`
  - Vector store & RAG: `chromadb`, `sentence-transformers`
  - Training: `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `huggingface_hub`
  - Orchestration: `modal` (for remote GPU QLoRA training)

---

## 3. Getting Started – Frontend

### 3.1 Prerequisites

- Node.js 18+ and npm
- Git

### 3.2 Install & Run

```bash
git clone <YOUR_REPO_URL>.git
cd visa_buddy
npm install
npm run dev
```

The app will start on a Vite dev port (typically `http://localhost:5173`).  
You’ll see the Visa Buddy chat interface with:

- a **side menu** for chat history
- a **header** with:
  - new chat button
  - link button to configure the backend URL
- a **message input** at the bottom with typing/“Visa Buddy is thinking…” feedback.

---

## 4. Backend – Colab + Cloudflare Tunnel + Model ZIP

Visa Buddy’s model and RAG logic live in a Python backend you run yourself (typically in **Google Colab**).  
The **model ZIP** is stored in your personal cloud (e.g., Google Drive) and mounted/loaded from Colab.

### 4.1 Steps Overview

1. **Download or train the model ZIP**
2. **Upload the ZIP to a storage location** (e.g., Google Drive)
3. **Open the Colab backend notebook**
4. **Mount Drive (or other storage) and load the model ZIP**
5. **Start the HTTP server in Colab**
6. **Expose the Colab port via Cloudflare tunnel**
7. **Paste the public tunnel URL into Visa Buddy (Link modal)**

### 4.2 Model ZIP (LoRA/QLoRA Adapter)

The main notebook (`Visa Buddy jupyter with outputs and measures`) includes:

- QLoRA training on top of **`meta-llama/Llama-3.1-8B-Instruct`**
- Saving adapters and tokenizer into `./visa_buddy_data/models`
- Zipping model artifacts into **`visa-buddy-model.zip`**

Typical contents after training:

- `adapter_model.safetensors`, `adapter_config.json` (LoRA weights)
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja`
- `checkpoint-*` folders and `training_args.bin`

You should:

1. Copy `visa_buddy_data/models/visa-buddy-model.zip` to your local machine.
2. Upload the ZIP to **Google Drive** (e.g., `MyDrive/visa_buddy/visa-buddy-model.zip`), or any storage you prefer.

### 4.3 Running the Backend in Google Colab

1. **Open the Colab backend notebook**
   - Use `visa_buddy_canada google colab backend and model.ipynb` (or a copy of the main notebook adapted for Colab).

2. **Mount Google Drive (if using Drive)**

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Set paths to your model ZIP and data directory**

   ```python
   MODEL_ZIP_PATH = "/content/drive/MyDrive/visa_buddy/visa-buddy-model.zip"
   LOCAL_MODELS_DIR = "./visa_buddy_data/models"
   ```

4. **Unzip and load the model**
   - Unzip the model into `LOCAL_MODELS_DIR`
   - Load base model (`Llama-3.1-8B-Instruct`) + LoRA adapters as shown in the inference cell (`setup_inference_model`).

5. **Start an HTTP server**
   - Implement a simple FastAPI/Flask endpoint such as:
     - `POST /chat` with payload `{ "message": "user question", "history": [...] }`
   - Inside the handler:
     - run retrieval (RAG) using your chunk files & embeddings
     - call the loaded Llama model with a prompt that includes RAG context
     - return the model’s answer as JSON.

> The exact endpoint signature must match what `visaBuddyApi.ts` / `useVisaBuddy` expect (typically `POST` returning a text response or message list). If you change the payload shape, update those TypeScript files as well.

### 4.4 Exposing Colab via Cloudflare Tunnel

1. **Install Cloudflare tunnel in Colab**
   - In a Colab cell, download and run `cloudflared` (for example):

   ```bash
   !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   !sudo dpkg -i cloudflared-linux-amd64.deb
   ```

2. **Authenticate and start a tunnel**
   - From your local machine, or using a Cloudflare token, create a tunnel that forwards to the Colab server port (e.g., `localhost:8000`).
   - When you run the tunnel command in Colab, it prints a **public HTTPS URL**, e.g.:
     - `https://your-random-subdomain.trycloudflare.com`

3. **Use the tunnel URL in Visa Buddy**
   - Run `npm run dev` and open the frontend.
   - Click the **link icon** in the header to open `LinkModal`.
   - Paste your tunnel URL (e.g. `https://your-random-subdomain.trycloudflare.com`) as the API base URL.
   - Save – `useVisaBuddy` will now send chat requests to this backend.

---

## 5. How the RAG + Training Pipeline Works

The notebook `Visa Buddy jupyter with outputs and measures.ipynb` builds a full immigration‑focused RAG + QLoRA system in several stages:

### 5.1 Data Collection and Pre‑processing

- **Crawling (Cells 4–7)**
  - Starts from `https://www.canada.ca/en/immigration-refugees-citizenship.html`.
  - Restricts to `www.canada.ca`, depth = 2.
  - Focuses on Express Entry and related pages via URL filtering (`"express-entry"`).
  - Saves raw HTML to `./visa_buddy_data/raw_html`, with hashed filenames.

- **PDF Download & Extraction (Cell 5)**
  - Template functions to download immigration PDFs and extract text with `pdfplumber`.
  - Writes extracted text into `./visa_buddy_data/clean_text`.

- **HTML → Clean Text (Cells 10–11)**
  - Uses `trafilatura.extract` as primary extractor; falls back to `BeautifulSoup.get_text`.
  - Removes navigation, headers/footers, scripts, styles.
  - Saves normalized plain text to `./visa_buddy_data/clean_text`.

- **Text Normalization (Cell 12–13)**
  - Replaces non‑breaking spaces.
  - Collapses multiple spaces/newlines.
  - Produces cleaner text that’s friendly for chunking and embeddings.

### 5.2 Chunking and Embeddings

- **Chunking (Cell 7 / 15–16)**
  - Uses NLTK sentence tokenization.
  - Creates overlapping chunks with:
    - `CHUNK_SIZE = 1000` (approx. words)
    - `CHUNK_OVERLAP = 200` (words)
  - Saves chunks as individual `.txt` files in `./visa_buddy_data/chunks/<source>/`.

- **Embeddings + Chroma Ingestion (Cells 17–18)**
  - Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
  - Creates a Chroma collection `visa_buddy_chunks`.
  - Stores:
    - `ids` (chunk identifiers),
    - `embeddings`,
    - `metadatas` (source and file path),
    - `documents` (chunk text).

- **RAG Retrievers (Cell 9 / 19–20)**
  - `retrieve_top_k`, `retrieve_top_k_simple`, `retrieve_top_k_enhanced`:
    - query embedding → Chroma query → top‑k chunks with distances.
    - `retrieve_top_k_enhanced` also computes **similarity scores** and provides a ranked list.

### 5.3 Finetune Dataset Construction (SFT)

- **Similarity‑aware dataset builder (Cell 22)**
  - Templates immigration questions across topics:
    - Express Entry, study permits, work permits, PGWP, visitor visas, PR, family sponsorship, citizenship, refugees, business immigration.
  - For each (topic, question template):
    - Runs `retrieve_top_k_enhanced`.
    - Filters by `similarity_score >= 0.3`.
    - Builds a **context string** from top 1–2 chunks, annotated with similarity.
    - Computes average similarity per example.
    - Generates a **structured answer** using `generate_structured_response` and `extract_key_phrases_from_docs`.
  - Writes JSONL to `visa_buddy_data/quality_finetune_dataset.jsonl` with:
    - `instruction`, `input` (context + question), `output` (answer)
    - metadata: topic, similarity, confidence level, question type, etc.

This produces a **high‑quality synthetic SFT dataset** rooted in real IRCC content while avoiding raw copy of government text in model outputs.

### 5.4 QLoRA Training (Modal)

- **Modal function `train_and_download` (Cell 25)**
  - Uses a Modal app with a GPU (T4) and a custom image.
  - Loads dataset from `/root/quality_finetune_dataset.jsonl`.
  - Base model: `meta-llama/Llama-3.1-8B-Instruct`.
  - QLoRA configuration:
    - `r=8`, `lora_alpha=16`, `lora_dropout=0.05`
    - Targets attention matrices (`q_proj`, `v_proj`)
  - Training hyperparameters:
    - `per_device_train_batch_size=1`
    - `gradient_accumulation_steps=8`
    - `num_train_epochs=2`
    - `learning_rate=2e-4`
    - `fp16=True`
  - Returns the model as a **zip file** (`visa-buddy-model.zip`), which is then downloaded and extracted into `./visa_buddy_data/models`.

### 5.5 Inference

- **`setup_inference_model` / `generate_llama_response` (Cell 27)**
  - Loads base Llama model and LoRA adapters (if present).
  - Requires `HF_TOKEN` environment variable for auth.
  - Formats prompts consistently (user + assistant headers).
  - Provides a fallback to `microsoft/DialoGPT-medium` if Llama isn’t accessible.
  - A simple smoke‑test cell sends prompts like “What is Express Entry?” and verifies response generation.

---

## 6. Metrics and Evaluation – Explanation of the Notebook Outputs

The notebook includes a **metrics section** that simulates and visualizes training and RAG performance. This is especially useful for presentations and documentation.

### 6.1 Training Metrics (Loss & Perplexity)

- Implemented in **Cell 14** (`simulate_training_metrics`, `plot_training_metrics`):
  - **Loss**:
    - Simulated as an exponentially decaying curve with noise.
    - Represents how well the model fits the training data.
    - Lower loss → better fit. A smoothly decreasing curve indicates stable learning.
  - **Perplexity**:
    - Approximate measure of how “surprised” the model is when predicting tokens.
    - Lower perplexity → model is more confident/accurate in next‑token predictions.
    - In the notebook, perplexity follows a similar decaying pattern.
  - The plot:
    - Shows **loss vs. steps** and **perplexity vs. steps** side‑by‑side.
    - Annotates **final loss** and **final perplexity** at the right side of each curve.

**How to interpret:**

- A downward‑sloping loss curve that stabilizes suggests successful fine‑tuning.
- Perplexity decreasing and then plateauing suggests the model learned the distribution and is no longer improving significantly.
- If either curve increases or becomes extremely noisy, it can indicate:
  - learning rate too high
  - overfitting
  - poor dataset quality.

> Note: These curves are currently **simulated** for visualization. In a production run, you would replace them with real logs from `Trainer` to get actual loss/perplexity values.

### 6.2 RAG Performance Metrics

Implemented in **Cell 15** (`RAG PERFORMANCE EVALUATION DASHBOARD`).

- **Benchmark queries:**
  - 10 realistic immigration questions (Express Entry, study permits, work permits, visitor visas, PR, etc.).

- **Retrieval logic for evaluation:**
  - Uses direct chunk‑file search (`retrieve_from_chunk_files`) instead of Chroma:
    - Scores each chunk based on:
      - query phrase match
      - word overlap
      - keyword density
      - immigration term boosts (e.g., “express entry”, “study permit”).

- **Per‑query metrics:**
  - `retrieved_count` – how many chunks were returned.
  - `avg_score` – average retrieval score across returned chunks.
  - `avg_relevance` – average fraction of query words found in chunk text.
  - `max_score` – best per‑chunk score.
  - `successful_retrieval` – whether at least one chunk was found.

- **Dashboard plots:**
  - **Relevance by Query:** bar chart of `avg_relevance` per query (`Q1–Q10`).
  - **Retrieval Success Rate:** pie chart showing % of queries with at least one retrieved chunk.
  - **Score Distribution:** histogram of average retrieval scores.
  - **Overall Summary:** bar chart of:
    - average relevance,
    - success rate,
    - average score,
    - average documents retrieved per query.

In the sample run logged in the notebook:

- Average relevance ≈ **0.57**
- Average retrieval score ≈ **0.38**
- Retrieval success rate = **100%**
- Average documents per query = **3**

**How to interpret:**

- **Average relevance (~0.57):**
  - Means, on average, about 57% of the query’s words overlap with retrieved chunk text.
  - Higher values (closer to 1.0) indicate very tight matches.
  - Values in the 0.5–0.7 range are generally good for a first‑pass RAG system.

- **Success rate (100% in the example):**
  - Every test question found at least one relevant chunk.
  - This is a strong signal that the knowledge base is reasonably aligned with the test questions.

- **Score distribution:**
  - Helps you see whether most queries are “easy” (high scores) or there’s a spread, highlighting weak spots.

### 6.3 Comprehensive Evaluation Report

**Cell 17** (`COMPREHENSIVE EVALUATION REPORT`) synthesizes the above metrics into a human‑readable and machine‑readable report:

- **Artifacts produced:**
  - `evaluation_report_*.json` – structured JSON with:
    - benchmark values
    - average RAG metrics
    - topic‑level performance
    - system health
    - risk assessment and recommendations.
  - `executive_summary_*.txt` – short, presentation‑ready summary of:
    - system status
    - key KPIs (relevance, success rate, docs per query)
    - strengths and top recommendations.
  - `detailed_analysis_*.txt` – long‑form technical breakdown:
    - per‑topic relevance
    - best/worst queries
    - risk breakdown and mitigations.

This turns your notebook into a **full evaluation suite**, not just a code playground:

- You can attach these reports to a portfolio, paper, or internal review.
- They make explicit:
  - where the system meets or misses RAG benchmarks
  - what to improve next (coverage, retrieval, training, UX).

---

## 7. UX / UI Notes

- **ChatInterface**
  - Persists chats in `localStorage` using `visabuddy-chats`.
  - Automatically scrolls to the latest message.
  - Shows:
    - user messages with a highlighted bubble
    - assistant messages in a neutral style
    - a typing indicator (“Visa Buddy is thinking…”) while waiting for backend.

- **SideMenu**
  - Groups chats into **Today / Yesterday / 7 Days / Older** based on `createdAt`.
  - Allows renaming, deleting (with confirmation), and selecting chats.

- **LinkModal**
  - Manages the backend API URL and connection status (`connected`, `disconnected`, etc.).
  - This is where you paste your Cloudflare tunnel URL.

These UX decisions are tuned for a **professional, assistant‑style experience** rather than a toy demo.

---

## 8. Publishing to GitHub

Once you’re happy with the project locally and this README, you can publish it to GitHub.

### 8.1 Initialize Git (if not already)

From the project root:

```bash
cd /Users/nicholasadan/visa_buddy
git init
git add .
git commit -m "Initial commit: Visa Buddy RAG + QLoRA assistant"
```

### 8.2 Create GitHub Repo and Push

1. Create a new repository on GitHub (no README/license needed, since this repo already has one).
2. Add it as a remote and push:

```bash
git remote add origin git@github.com:<your-username>/<your-repo-name>.git
git branch -M main
git push -u origin main
```

Replace `<your-username>` and `<your-repo-name>` with your GitHub account and chosen repo name (for example, `visa-buddy`).

---

## 9. Next Steps / Ideas

- Integrate **real training logs** into the metrics notebook instead of simulated curves.
- Add **endpoint health checks** and latency measurements from the frontend.
- Extend RAG coverage to:
  - provincial nomination programs,
  - more visa categories (e.g., spouse open work permits, LMIA, etc.).
- Add feedback capture in the UI (thumbs up/down) to refine the dataset over time.

# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
