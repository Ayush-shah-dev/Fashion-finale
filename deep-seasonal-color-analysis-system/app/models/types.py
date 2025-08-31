from typing import List
from pydantic import BaseModel

class RankedImage(BaseModel):
    image_path: str
    score: float

class AnalyzeResponse(BaseModel):
    season: str
    hex_colors: List[str]
    prompt: str
    top_matches: List[RankedImage]