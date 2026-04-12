from pydantic import BaseModel
from typing import List, Optional

class RecommendationWeights(BaseModel):
    plot: float = 0.45
    genre: float = 0.15
    director: float = 0.10
    origin: float = 0.05
    year: float = 0.05
    concept: float = 0.10

class RecommendationRequest(BaseModel):
    title: str
    top_n: int = 10
    weights: Optional[RecommendationWeights] = None

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

class MovieRecommendation(BaseModel):
    title: str
    release_year: int
    genre: str
    origin: str
    plot: str
    similarity_score: float
    explanation: str

class RecommendationResponse(BaseModel):
    query_title: str
    recommendations: List[MovieRecommendation]

class SearchResponse(BaseModel):
    results: List[str]
