import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))
SKILL_WEIGHT = float(os.getenv("SKILL_WEIGHT", "0.3"))
SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
STORE_DIR = "store"
INDEX_NAME = "resume_index"
