from app.ranker import index_resumes_dir, rank_resumes
import os, shutil

def setup_module(module):
    if os.path.exists("store"):
        shutil.rmtree("store")

def test_ranker():
    n = index_resumes_dir("data/resumes")
    assert n >= 1
    out = rank_resumes("Looking for an ML Engineer with Python and AWS experience", top_k=2, skills=["python","aws"])
    assert len(out) >= 1
    assert "score" in out[0]
