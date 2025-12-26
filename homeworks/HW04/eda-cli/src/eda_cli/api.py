from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import pandas as pd
from io import StringIO
from . import core

app = FastAPI(
    title="EDA Quality API",
    description="API для оценки качества датасетов на основе eda_cli.",
    version="0.1.0"
)

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    service: str

class QualityRequest(BaseModel):
    n_rows: int
    n_cols: int
    max_missing_share: float

class FlagInfo(BaseModel):
    too_few_rows: bool
    too_many_missing: bool
    has_constant_columns: bool
    has_many_zero_values: bool

class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float
    latency_ms: float
    flags: FlagInfo

class QualityFlagsResponse(BaseModel):
    flags: Dict[str, Any]
    latency_ms: float

@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    return HealthCheckResponse(
        status="healthy",
        version=app.version,
        service="EDA Quality API"
    )

@app.post("/quality", response_model=QualityResponse)
def predict_quality(request: QualityRequest):
    start_time = time.perf_counter()
    
    fake_summary = core.DatasetSummary(
        n_rows=request.n_rows,
        n_cols=request.n_cols,
        columns=[]
    )
    fake_missing_df = pd.DataFrame({'missing_share': [request.max_missing_share]})
    
    fake_data = {'dummy_col': [1] * request.n_rows}
    for _ in range(request.n_cols - 1):
        fake_data[f'dummy_col_{_}'] = [1] * request.n_rows
    fake_df = pd.DataFrame(fake_data)

    flags = core.compute_quality_flags(fake_df, fake_summary, fake_missing_df)

    latency = (time.perf_counter() - start_time) * 1000
    ok_for_model = flags.get('quality_score', 0.0) > 0.5

    flag_info_obj = FlagInfo(
        too_few_rows=flags.get('too_few_rows', False),
        too_many_missing=flags.get('too_many_missing', False),
        has_constant_columns=flags.get('has_constant_columns', False),
        has_many_zero_values=flags.get('has_many_zero_values', False),
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=flags.get('quality_score', 0.0),
        latency_ms=latency,
        flags=flag_info_obj
    )

@app.post("/quality-from-csv", response_model=QualityResponse)
async def predict_quality_from_csv(file: UploadFile = File(...)):
    start_time = time.perf_counter()

    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    summary = core.summarize_dataset(df)
    missing_df = core.missing_table(df)
    flags = core.compute_quality_flags(df, summary, missing_df)

    latency = (time.perf_counter() - start_time) * 1000
    ok_for_model = flags.get('quality_score', 0.0) > 0.5

    flag_info_obj = FlagInfo(
        too_few_rows=flags.get('too_few_rows', False),
        too_many_missing=flags.get('too_many_missing', False),
        has_constant_columns=flags.get('has_constant_columns', False),
        has_many_zero_values=flags.get('has_many_zero_values', False),
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=flags.get('quality_score', 0.0),
        latency_ms=latency,
        flags=flag_info_obj
    )

@app.post("/quality-flags-from-csv", response_model=QualityFlagsResponse)
async def get_quality_flags_from_csv(file: UploadFile = File(...)):
    start_time = time.perf_counter()

    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    summary = core.summarize_dataset(df)
    missing_df = core.missing_table(df)
    flags = core.compute_quality_flags(df, summary, missing_df)

    latency = (time.perf_counter() - start_time) * 1000

    return QualityFlagsResponse(
        flags=flags,
        latency_ms=latency
    )