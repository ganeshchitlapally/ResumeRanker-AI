# ResumeRanker AI: Semantic Resume ↔ Job Description Matching

ResumeRanker AI ranks a folder of resumes against a target Job Description (JD), combining:
- **Semantic similarity** (Sentence-BERT embeddings, cosine)
- **Keyword/skill overlap** (weighted)
- **Rule bonuses/penalties** (e.g., years of exp, location if present)

## Why this is hire‑signal
- Realistic HRTech/NLP problem: **ranking + scoring + bias checks**.
- FAISS index for fast retrieval, **FastAPI** service, Docker-ready.
- Clean separation of concerns with tests and config.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload
# 1) POST /index_resumes  -> indexes ./data/resumes
# 2) POST /rank           -> ranks resumes for provided JD text
```

## API
- `POST /index_resumes` → build/update FAISS index from `./data/resumes`
- `POST /rank` body:
```json
{
  "jd_text": "We need an ML Engineer with Python, PyTorch, AWS...",
  "top_k": 5,
  "skills": ["python","pytorch","aws","docker","mlops"]
}
```
response: ranked list with scores and rationales.

## Files
```
ResumeRanker-AI/
  app/
    main.py
    ranker.py
    config.py
    models.py
  data/
    resumes/ (sample .txt files included)
    jds/ (sample_jd.txt)
  tests/
    test_ranker.py
  requirements.txt
  Dockerfile
  .env.example
  README.md
```

## Resume Bullets
- Built **resume–JD ranking service** using Sentence‑BERT + feature rules; shipped FAISS‑backed FastAPI API with Docker and unit tests.
- Implemented explainable scoring with semantic cosine similarity and explicit skill matches; exposed `/rank` endpoint returning **rationales** and alignment snippets.
- Added bias/leakage safeguards and configurable skill weighting for fairer ranking.
