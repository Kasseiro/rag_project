from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentSnippet(BaseModel):
    id: int
    title: Optional[str] = None
    content: str
    distance: float


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Question to pass to the RAG backend")
    top_k: int = Field(3, ge=1, le=10, description="How many similar documents to retrieve")
    temperature: float = Field(1.0, ge=0, le=2, description="Sampling temperature for the LLM response")


class AskResponse(BaseModel):
    query: str
    answer: str
    documents: List[DocumentSnippet] = Field(default_factory=list)


class ClearResponse(BaseModel):
    status: str
    message: str
