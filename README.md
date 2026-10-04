# SLM Q&A Comparison

## Objective

Fine-tune and compare three Small Language Models on a Question-Answer dataset.

## Models

1. DistilGPT2
2. TinyLlama-1.1B
3. Qwen2.5-0.5B-Instruct

## Planned Pipeline

Q&A Dataset
→ Data Preprocessing
→ Train/Validation/Test Split
→ Tokenization
→ LoRA Fine-tuning
→ Evaluation
→ Model Comparison

## Evaluation Metrics

- Exact Match
- F1 Score
- ROUGE-1
- ROUGE-L
- Validation Loss

## Tools

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT / LoRA
- Google Colab
- GitHub