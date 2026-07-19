# InsightMind AI // Local Multimodal Agentic RAG System

[![Python Version](https://shields.io)](https://python.org)
[![Docker Image](https://shields.io)](https://docker.com)
[![VectorDB](https://shields.io)](https://qdrant.tech)
[![MLOps Tracking](https://shields.io)](https://mlflow.org)
[![CI/CD Pipeline](https://shields.io)](https://github.com)

An enterprise-grade, privacy-preserving, and offline-capable **Multimodal Agentic RAG (Retrieval-Augmented Generation)** assistant tailored for deep scientific document understanding. Driven locally by **Qwen2.5-VL:7b** and **Qdrant Vector DB**, this system autonomously orchestrates decoupled vector retrieval paths to parse complex texts, tabular evaluation metrics, and visual engineering charts without cloud dependency.

---

## 📺 Live Application Demonstration

![Application Demo](https://githubusercontent.com)
*Note: Ganti jalur placeholder ini dengan aset gambar bergerak (.gif) berdurasi 60 detik hasil rekaman layar Streamlit Anda yang menunjukkan kelancaran ketikan streaming respons, perbandingan tabel kotak, dan penemu gambar diagram jurnal.*

---

## 🚀 Key Architectural Capabilities

- **Decoupled Cross-Modal Retriever:** Leverages dual-engine vector mappings using `paraphrase-multilingual-mpnet-base-v2` (768-dim) for semantic text/tabular segments and OpenAI's `CLIP-ViT-Base-Patch32` (512-dim) for high-fidelity visual layout indexing.
- **Reverse Image Search Engine:** Autonomous pixel-to-vector distance calculations allowing users to upload an isolated chart or diagram and instantly trace back its precise source document context (**[PAPER ID]**) with high mathematical confidence metrics.
- **Dynamic Intent Classifier & Vision Routing:** Intelligently intercepts user prompts to separate lightweight text context extraction from deep vision reasoning, injecting dynamic **Base64 String Arrays** to feed the LLM's visual synapses on-demand.
- **Memory-Safe MLOps Telemetry Pipeline:** Comprehensive, production-ready logging of execution latency, tool invocation paths, and hierarchical reasoning token structures logged cleanly via **MLflow Text-Log** parameters, eliminating Windows I/O file-locking overhead (`WinError 32`).
- **Production-Ready Containerization:** Package blueprint completely isolated into a Python 3.12-slim base Docker layer utilizing explicit in-build caching mechanisms (`SentenceTransformer` model hydration during compile time) to guarantee a sub-3-second UI load time on any host environment.
- **Automated CI/CD Pipeline:** Fully integrated **GitHub Actions Automation** workflow triggered on every push to the `main` branch, handling remote Ubuntu compiler linting checks, multi-stage layer caching, and automatic image multi-tagging delivery directly to Docker Hub registries.

---

## 🛠️ System Architecture Blueprint

```text
       ┌────────────────────────────────────────────────────────┐
       │             Streamlit WebUI Interface                  │
       └─────────────────────────┬──────────────────────────────┘
                                 │
                 [ User Prompt / File Upload ]
                                 ▼
       ┌────────────────────────────────────────────────────────┐
       │      Autonomous Smart Intent Classifier / Router       │
       └─────────┬────────────────────────────────────┬─────────┘
                 │                                    │
    (Text / Table Query)                     (Visual Asset Upload)
                 ▼                                    ▼
┌─────────────────────────────────┐        ┌─────────────────────────────────┐
│  MPNet Semantic Vector Search   │        │     CLIP Image Feature Match    │
│    [768-dim Core Embedding]     │        │      [512-dim Core Tensor]      │
└────────────────┬────────────────┘        └────────────────┬────────────────┘
                 │                                    │
                 ▼                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                  Decoupled Qdrant DB Core Collections                      │
│        [scientific_texts node]           [scientific_images node]          │
└────────────────┬────────────────────────────────────┬──────────────────────┘
                 │                                    │
                 └─────────────────┬──────────────────┘
                                   │
                   [ Multimodal Context Injection ]
            (Markdown Table Syntax + Safe Base64 String)
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │   Local Inference Engine (Ollama Server - Qwen2.5-VL)  │
       └─────────────────────────┬──────────────────────────────┘
                                 │
                   [ Real-time Streamed Chunk ]
                                 ▼
       ┌────────────────────────────────────────────────────────┐
       │  Front-End Rendering UI  ◄───► MLOps Experiment Tracking│
       │    (Streamlit Client)    │   (MLflow Pipeline Port 5000)│
       └──────────────────────────┴──────────────────────────────┘
```

---

## 📦 Project Directory Structure

```text
multimodal-rag-project/
│
├── .github/
│   └── workflows/
│       └── main.yml           # Automated GitHub Actions CI/CD Workflow
│
├── data/                      # Local PDF Corpus & Processed Multi-Modal Chunks
├── logs/                      # System Runtime Log Output Storage
├── mlruns/                    # MLOps Telemetry & Experiment Vaults
├── qdrant_storage/            # Persistent Local Database Storage for Qdrant Node
│
├── src/                       # Central Processing Architecture Core
│   ├── agent/                 # Core Brain Reasoning Engines (Ollama SDK)
│   ├── extraction/            # Document Parsing & Asset Segment Layer
│   ├── ingestion/             # Text Chunk Structuring Pipeline
│   ├── utils/                 # MLOps Tracking Server Connectors (MLflow)
│   └── vector_store/          # Base Qdrant Connectivity Implementations
│
├── venv/                      # Local Isolated Python Virtual Environment
│
├── .dockerignore              # Context File Asset Exclusion Manifest
├── .gitignore                 # Native Git Resource Exclusion Blueprint
├── app.py                     # High-Performance Frontend Execution UI Client
├── docker-compose.yml         # Multi-Container Stack Production Orchestrator
├── Dockerfile                 # Multi-Stage Optimized Python 3.12 Deployment Base
├── README.md                  # Comprehensive System Enterprise Documentation
└── requirements.txt           # Verified Multi-Modal Production Dependencies List
```

---

## ⚙️ Deployment & Re-production Guide

This codebase is completely decoupled and supports dual-execution modes depending on your hardware resource boundaries. Ensure **Docker Desktop** is running and your local **Ollama** server is accessible before launching.

### Prerequisites (Global Access Clearance)
To allow the containerized application network to securely invoke your local host machine's Ollama model instance, authorize cross-origin resource requests globally on your system:
```bash
# Set environment variables inside your system parameters or cmd terminal
set OLLAMA_HOST=0.0.0.0

# Restart the local server instance to activate boundaries
ollama serve
```

---

### Option A: Standard Production Blueprint Mode (Docker Stack)
Highly recommended for production deployments on Linux clusters or cloud nodes. This fires up the database storage and frontend instances inside an isolated host bridge network in one single click.

```bash
# 1. Fire up the production database storage engine node
docker-compose up -d

# 2. Verify all multi-container services are running securely
docker ps
```
Access the interactive web UI interface by pointing your web browser to `http://localhost:8501`.

---

### Option B: High-Performance Hybrid Mode (Local Host Runtime)
Best suited for local Windows machine testing to eliminate hypervisor emulated instruction overhead (`AVX2/SIMD` tensor bottleneck inside WSL2), ensuring lightning-fast streaming text compilation times.

```bash
# 1. Start the decoupled database node cluster in the background
docker start qdrant_local

# 2. Launch the MLOps performance and telemetry tracking interface
mlflow ui --port 5000

# 3. Fire up the frontend application wrapper inside your native virtual environment
streamlit run app.py
```
Open your preferred web browser to access the frontend client at `http://localhost:8501`, or manage experiment metrics at `http://localhost:5000`.

---

## 📊 Industrial Value Propositions (The Interview X-Factor)

- **Absolute Cost-Efficiency:** Operates 100% locally and free-of-charge. By substituting paid proprietary APIs (OpenAI GPT-4V/Gemini Pro) with the highly competitive Qwen2.5-VL engine, it effectively cuts operational token spending down to zero.
