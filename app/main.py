from fastapi import FastAPI, HTTPException
from app.ranker import index_resumes_dir, rank_resumes
from app.models import RankRequest

app = FastAPI(title="ResumeRanker AI", version="1.0.0")

@app.post("/index_resumes")
def index_resumes():
    try:
        n = index_resumes_dir("data/resumes")
        return {"status": "ok", "resumes_indexed": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rank")
def rank(req: RankRequest):
    try:
        results = rank_resumes(req.jd_text, top_k=req.top_k, skills=req.skills)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
