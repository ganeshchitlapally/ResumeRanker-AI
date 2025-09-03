from pydantic import BaseModel, Field
from typing import List

class RankRequest(BaseModel):
    jd_text: str = Field(..., description="Job description text")
    top_k: int = Field(5, ge=1, le=50)
    skills: List[str] = Field(default_factory=list, description="Target skills to emphasize")
