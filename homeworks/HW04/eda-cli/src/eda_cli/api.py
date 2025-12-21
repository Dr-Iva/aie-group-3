# src/eda_cli/api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import pandas as pd
from io import StringIO
from . import core  # Импортируем модуль core из этого же пакета (eda_cli)

# --- Инициализация FastAPI приложения ---
app = FastAPI(
    title="EDA Quality API",
    description="API для оценки качества датасетов на основе eda_cli.",
    version="0.1.0"
)

# --- Pydantic модели для запросов и ответов ---

# Модель для ответа от эндпоинта /health
class HealthCheckResponse(BaseModel):
    status: str
    version: str
    service: str

# Модель для запроса к эндпоинту /quality
class QualityRequest(BaseModel):
    n_rows: int
    n_cols: int
    max_missing_share: float
    # Вы можете добавить другие параметры, которые хотите передавать,
    # но для простоты начнём с этих.

# Модель для информации о флагах (часть QualityResponse)
class FlagInfo(BaseModel):
    too_few_rows: bool
    too_many_missing: bool
    has_constant_columns: bool  # <-- Новый флаг из HW03
    has_many_zero_values: bool  # <-- Новый флаг из HW03
    # Добавьте другие флаги, которые вы добавляли в HW03

# Модель для ответа от эндпоинтов /quality и /quality-from-csv
class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float
    latency_ms: float
    flags: FlagInfo

# Модель для ответа от эндпоинта /quality-flags-from-csv
class QualityFlagsResponse(BaseModel):
    flags: Dict[str, Any]
    latency_ms: float

# --- Реализация эндпоинтов ---

@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    """
    Проверка состояния сервиса.
    """
    return HealthCheckResponse(
        status="healthy",
        version=app.version,
        service="EDA Quality API"
    )

@app.post("/quality", response_model=QualityResponse)
def predict_quality(request: QualityRequest):
    """
    Принимает параметры датасета и возвращает оценку его качества.
    Это упрощённая версия без чтения CSV.
    """
    start_time = time.perf_counter()

    # Здесь мы "создаём" фиктивный summary и missing_df для демонстрации.
    # В реальности, если бы мы получили все данные, мы бы их использовали.
    # Но мы будем использовать request.n_rows, n_cols и max_missing_share
    # для гипотетического вычисления флагов.
    # Однако, compute_quality_flags ожидает реальные объекты DatasetSummary и pd.DataFrame.
    # Для этого эндпоинта мы можем сымитировать поведение.
    # Но лучше использовать /quality-from-csv для реального анализа.

    # Создаём фиктивный DataFrame и Summary для вызова compute_quality_flags
    # Это не самый чистый способ для этого эндпоинта, но позволяет переиспользовать логику.
    # Более реалистично использовать /quality-from-csv.
    # Пока оставим это как есть, но логика будет приблизительной.
    fake_summary = core.DatasetSummary(
        n_rows=request.n_rows,
        n_cols=request.n_cols,
        columns=[] # Пустой список, так как у нас нет реальных колонок
    )
    fake_missing_df = pd.DataFrame({'missing_share': [request.max_missing_share]})

    # ВАЖНО: Используем сигнатуру из вашего core.py из HW03!
    # Если вы передавали df, то нужно будет передать и здесь фиктивный df.
    # Предположим, сигнатура compute_quality_flags из HW03: (df, summary, missing_df)
    # Тогда нам нужно создать фиктивный df.
    # Если сигнатура была (summary, missing_df), то используем её.

    # Проверим сигнатуру. Поскольку вы предоставили core.py, он принимает (df, summary, missing_df).
    # Нам нужно создать фиктивный df, чтобы передать в функцию.
    # Это не идеальный подход для этого эндпоинта, но позволяет использовать существующую логику.
    # Создаём минимальный df, чтобы не нарушить внутреннюю логику core.
    # Пусть будет 1 колонка, чтобы не вызвать ошибок, если core ожидает хотя бы одну.
    fake_data = {'dummy_col': [1] * request.n_rows}
    for _ in range(request.n_cols - 1):
        fake_data[f'dummy_col_{_}'] = [1] * request.n_rows
    fake_df = pd.DataFrame(fake_data)

    # Теперь вызываем compute_quality_flags с правильной сигнатурой
    flags = core.compute_quality_flags(fake_df, fake_summary, fake_missing_df)

    latency = (time.perf_counter() - start_time) * 1000

    # Возвращаем результаты
    # ok_for_model: решим, что датасет годен, если качество > 0.5
    ok_for_model = flags.get('quality_score', 0.0) > 0.5

    # Создаём объект FlagInfo, используя ключи из словаря flags
    flag_info_obj = FlagInfo(
        too_few_rows=flags.get('too_few_rows', False),
        too_many_missing=flags.get('too_many_missing', False),
        has_constant_columns=flags.get('has_constant_columns', False),
        has_many_zero_values=flags.get('has_many_zero_values', False),
        # Добавьте другие флаги, если они могут быть в словаре flags
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=flags.get('quality_score', 0.0),
        latency_ms=latency,
        flags=flag_info_obj
    )


@app.post("/quality-from-csv", response_model=QualityResponse)
async def predict_quality_from_csv(file: UploadFile = File(...)):
    """
    Принимает CSV-файл, анализирует его и возвращает оценку качества.
    """
    start_time = time.perf_counter()

    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    summary = core.summarize_dataset(df)
    missing_df = core.missing_table(df)

    # Вызываем compute_quality_flags с передачей df, как в HW03
    flags = core.compute_quality_flags(df, summary, missing_df)

    latency = (time.perf_counter() - start_time) * 1000

    # ok_for_model: решим, что датасет годен, если качество > 0.5
    ok_for_model = flags.get('quality_score', 0.0) > 0.5

    # Создаём объект FlagInfo
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

# --- Обязательный пользовательский эндпоинт из задания ---
@app.post("/quality-flags-from-csv", response_model=QualityFlagsResponse)
async def get_quality_flags_from_csv(file: UploadFile = File(...)):
    """
    Принимает CSV-файл, анализирует его и возвращает словарь флагов качества.
    Этот эндпоинт использует ваши доработки из HW03.
    """
    start_time = time.perf_counter()

    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    summary = core.summarize_dataset(df)
    missing_df = core.missing_table(df)

    # Вызываем compute_quality_flags с передачей df, как в HW03
    flags = core.compute_quality_flags(df, summary, missing_df)

    latency = (time.perf_counter() - start_time) * 1000

    return QualityFlagsResponse(
        flags=flags,
        latency_ms=latency
    )

# --- (Опционально) Дополнительный пользовательский эндпоинт ---
# Например, эндпоинт для получения сводки по колонкам
class ColumnSummaryResponse(BaseModel):
    summary: List[Dict[str, Any]]
    latency_ms: float

@app.post("/column-summary", response_model=ColumnSummaryResponse)
async def get_column_summary(file: UploadFile = File(...)):
    """
    Принимает CSV-файл, анализирует его и возвращает сводку по колонкам.
    """
    start_time = time.perf_counter()

    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    summary = core.summarize_dataset(df)

    # Конвертируем объекты ColumnSummary в словари
    summary_dicts = [col_summary.to_dict() for col_summary in summary.columns]

    latency = (time.perf_counter() - start_time) * 1000

    return ColumnSummaryResponse(
        summary=summary_dicts,
        latency_ms=latency
    )

