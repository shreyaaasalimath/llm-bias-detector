# LLM Bias Detection Pipeline

Detects demographic and political bias in LLM outputs using a 
three-signal scoring system.

## What it does
- Runs 60+ prompt variants across 12 demographic groups
- Scores responses using RoBERTa sentiment, BERT toxicity, and LLM-as-judge
- Visualizes bias disparities in a live Streamlit dashboard

## Tech stack
Python · Anthropic API · HuggingFace Transformers · Streamlit · Plotly · SQLite

## Setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Add your API key to `.env`:
Run the pipeline:
```bash
python main.py
```
Launch dashboard:
```bash
PYTHONPATH=$(pwd) streamlit run dashboard/app.py
```
