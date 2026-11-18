"""FastAPI application for campaign performance analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.analytics import analyze_campaigns
from app.llm_summary import summarize_analysis

app = FastAPI(
    title="RevenueOps Campaign Insights API",
    description="CSV/JSON preparation, KPI calculation, explainable outlier flags, and optional LLM summaries.",
    version="1.0.0",
)


class Campaign(BaseModel):
    campaign: str = Field(min_length=1, max_length=120)
    spend: float = Field(ge=0)
    impressions: int = Field(ge=0)
    clicks: int = Field(ge=0)
    conversions: int = Field(ge=0)


class AnalysisRequest(BaseModel):
    campaigns: list[Campaign] = Field(min_length=1, max_length=10000)


class SummaryRequest(BaseModel):
    analysis: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalysisRequest) -> dict[str, Any]:
    try:
        return analyze_campaigns([row.model_dump() for row in request.campaigns])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a .csv file")
    raw = await file.read(2_000_001)
    if len(raw) > 2_000_000:
        raise HTTPException(status_code=413, detail="CSV file must be 2 MB or smaller")
    try:
        frame = pd.read_csv(pd.io.common.BytesIO(raw))
        return analyze_campaigns(frame.to_dict(orient="records"))
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not analyze CSV: {exc}") from exc


@app.post("/summarize")
async def summarize(request: SummaryRequest) -> dict[str, str]:
    try:
        return {"summary": await summarize_analysis(request.analysis)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Summary provider request failed: {exc}") from exc
