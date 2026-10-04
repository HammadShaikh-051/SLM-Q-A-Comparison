# SLM-QA-Comparison Backend

This is the FastAPI backend for the SLM-QA-Comparison project. It provides an API to run Question Answering inference using 3 fine-tuned Small Language Models:
- DistilGPT2
- Qwen2.5-0.5B-Instruct
- TinyLlama-1.1B-Chat-v1.0

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the development server:
   ```bash
   uvicorn app.main:app --reload
   ```
