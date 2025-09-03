import os, re, json
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import EMBEDDING_MODEL, STORE_DIR, INDEX_NAME, SKILL_WEIGHT, SEMANTIC_WEIGHT, TOP_K_DEFAULT

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()

def _skills_score(text: str, skills: List[str]) -> float:
    if not skills:
        return 0.0
    t = _clean(text)
    hits = sum(1 for s in skills if s.lower() in t)
    return hits / max(1, len(skills))

def index_resumes_dir(resumes_dir: str) -> int:
    from langchain.docstore.document import Document
    docs = []
    for root, _, files in os.walk(resumes_dir):
        for f in files:
            if f.lower().endswith((".txt",)):
                p = os.path.join(root, f)
                docs.append(Document(page_content=_read_text(p), metadata={"source": p}))

    if not docs:
        raise ValueError("No .txt resumes found in data/resumes.")

    emb = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vs = FAISS.from_documents(docs, embedding=emb)
    os.makedirs(STORE_DIR, exist_ok=True)
    vs.save_local(os.path.join(STORE_DIR, INDEX_NAME))
    return len(docs)

def _load_index():
    emb = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(os.path.join(STORE_DIR, INDEX_NAME), emb, allow_dangerous_deserialization=True)

def rank_resumes(jd_text: str, top_k: int = TOP_K_DEFAULT, skills: List[str] = None) -> List[Dict]:
    if skills is None:
        skills = []
    jd = jd_text

    vs = _load_index()
    # Retrieve more than top_k to allow reranking
    docs = vs.similarity_search(jd, k=max(top_k*3, top_k))
    emb = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    jd_vec = emb.embed_query(jd)
    res_vecs = [emb.embed_query(d.page_content) for d in docs]

    cos = cosine_similarity([jd_vec], res_vecs).flatten()
    skill_scores = np.array([_skills_score(d.page_content, skills) for d in docs])

    final = SEMANTIC_WEIGHT * cos + SKILL_WEIGHT * skill_scores
    idxs = np.argsort(-final)[:top_k]

    ranked = []
    for i in idxs:
        d = docs[i]
        ranked.append({
            "resume_path": d.metadata.get("source"),
            "score": float(final[i]),
            "semantic": float(cos[i]),
            "skills_score": float(skill_scores[i]),
            "rationale": _make_rationale(d.page_content, skills)
        })
    return ranked

def _make_rationale(text: str, skills: List[str]) -> str:
    found = [s for s in skills if s.lower() in text.lower()]
    if found:
        return f"Matched skills: {', '.join(found[:10])}"
    return "Ranking driven by semantic similarity."
