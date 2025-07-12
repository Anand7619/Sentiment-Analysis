
# Electronix AI – Sentiment Microservice

## 🔍 Overview
End-to-end microservice for binary sentiment analysis using HuggingFace Transformers. Supports prediction, fine-tuning, and frontend interface.

## 🚀 Quick Start
```bash
docker-compose up --build
```

## 🧪 API

**POST /predict**
Request:
```json
{ "text": "This is great!" }
```

Response:
```json
{ "label": "positive", "score": 0.998 }
```

## 🛠 Fine-tuning
```bash
python finetune.py -data data/data.jsonl -epochs 3 -lr 3e-5
```

Saves model to `./model` for backend to pick up.

## ⚙️ Tech Stack
- 🧠 HuggingFace Transformers
- ⚙️ Flask + React
- 🐳 Docker + Compose

## 📊 CPU vs GPU (Approx.)
| Mode | Time per Epoch |
|------|----------------|
| CPU  | ~3 mins        |
| GPU  | ~20 secs       |
