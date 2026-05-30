# SemDatasheet

## Описание проекта

**SemDatasheet** — сервис интеллектуального поиска характеристик электронных компонентов в PDF Datasheet.

Проект автоматически извлекает текст из технической документации производителей электронных компонентов, строит гибридный поисковый индекс (BM25 + FAISS) и позволяет находить значения характеристик по запросам на естественном языке.

Пример запроса:

```json
{
  "query": "Максимальное напряжение питания MCP6001",
  "top_k": 3
}
```

Пример ответа:

```json
{
  "results": [
    {
      "rank": 1,
      "document": "MCP6001_datasheet.pdf",
      "page": 3,
      "text_snippet": "... Operating Voltage 1.8V to 6.0V ...",
      "extracted_value": "1.8V to 6.0V",
      "unit": "V",
      "score": 0.91
    }
  ],
  "processing_time_ms": 42.7
}
```

---

# Основные возможности

* загрузка PDF Datasheet из папки `data/raw`;
* извлечение текста через PyMuPDF;
* автоматический пропуск PDF без текстового слоя;
* интеллектуальное разбиение текста на чанки;
* приоритетная обработка секций:

  * Electrical Characteristics;
  * Absolute Maximum Ratings;
  * Typical Performance Characteristics;
* построение BM25 индекса;
* построение FAISS векторного индекса;
* гибридный поиск BM25 + Vector Search;
* извлечение числовых значений и единиц измерения;
* сохранение индексов и метаданных в артефакты;
* автоматическая загрузка индексов при запуске;
* автоматическая переиндексация при отсутствии артефактов;
* логирование через structlog;
* Prometheus метрики;
* MLflow интеграция.

---

# Архитектура проекта

```text
semdatasheet/
├── src/
│   ├── api/
│   ├── core/
│   ├── data/
│   ├── indexing/
│   ├── extraction/
│   ├── services/
│   └── main.py
│
├── data/
│   ├── raw/
│   └── eval/
│
├── artifacts/
│
├── tests/
│
├── notebooks/
│
├── config.yaml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
├── report.md
└── self-checklist.md
```

---

# Компоненты системы

## Data Layer

Отвечает за:

* загрузку PDF;
* извлечение текста;
* хранение моделей данных;
* чанкование текста.

Файлы:

```text
src/data/
```

---

## Indexing Layer

Отвечает за:

* построение BM25 индекса;
* построение FAISS индекса;
* гибридный поиск;
* сохранение артефактов.

Файлы:

```text
src/indexing/
```

---

## Search Layer

Отвечает за:

* обработку поисковых запросов;
* объединение результатов BM25 и FAISS;
* извлечение найденных характеристик.

Файлы:

```text
src/services/
```

---

## API Layer

Отвечает за:

* REST API;
* валидацию запросов;
* выдачу результатов.

Файлы:

```text
src/api/
```

---

# Используемые технологии

## Backend

* Python 3.10+
* FastAPI
* Uvicorn

## Работа с PDF

* PyMuPDF (fitz)

## Поиск

* rank-bm25
* faiss-cpu

## Embeddings

* sentence-transformers
* intfloat/multilingual-e5-small

## Конфигурация

* pydantic-settings
* python-dotenv
* PyYAML

## Логирование

* structlog

## Метрики

* prometheus_client

## Эксперименты

* MLflow

## Тестирование

* pytest
* httpx

---

# Установка

## Клонирование репозитория

```bash
git clone <repository_url>
cd semdatasheet
```

## Создание виртуального окружения

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Создание файла окружения

```bash
cp .env.example .env
```

---

# Подготовка данных

Поместите PDF Datasheet в каталог:

```text
data/raw/
```

Пример:

```text
data/raw/
├── MCP6001_datasheet.pdf
├── MCP6002_datasheet.pdf
└── LM358_datasheet.pdf
```

После запуска сервиса индексы будут построены автоматически.

---

# Запуск проекта

## Локальный запуск

```bash
uvicorn src.main:app --reload
```

После запуска:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

# API

## Проверка состояния сервиса

### Запрос

```http
GET /health
```

### Ответ

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_loaded": true,
  "index_loaded": true,
  "indexed_chunks": 124
}
```

---

## Поиск характеристик

### Запрос

```http
POST /search
```

Тело запроса:

```json
{
  "query": "Максимальное напряжение питания MCP6001",
  "top_k": 3
}
```

### Ответ

```json
{
  "results": [
    {
      "rank": 1,
      "document": "MCP6001_datasheet.pdf",
      "page": 3,
      "text_snippet": "Operating Voltage 1.8V to 6.0V",
      "extracted_value": "1.8V to 6.0V",
      "unit": "V",
      "score": 0.91
    }
  ],
  "processing_time_ms": 42.7
}
```

---

## Метрики

### Запрос

```http
GET /metrics
```

Метрики Prometheus:

* search_requests_total
* search_latency_seconds
* index_size

---

# Docker

## Сборка контейнера

```bash
docker compose build
```

или

```bash
docker-compose build
```

---

## Запуск

```bash
docker compose up
```

или

```bash
docker-compose up
```

---

## Остановка

```bash
docker compose down
```

---

# MLflow

Запуск интерфейса:

```bash
mlflow ui
```

После запуска:

```text
http://127.0.0.1:5000
```

В интерфейсе отображаются:

* alpha;
* top_k;
* количество документов;
* количество чанков.

---

# Тестирование

Запуск всех тестов:

```bash
pytest tests -v
```

Ожидаемый результат:

```text
=====================
3 passed
=====================
```

---

# Артефакты

После построения индекса в папке `artifacts/` появляются:

```text
artifacts/
├── bm25.pkl
├── faiss.index
├── chunks.json
├── manifest.json
```

Назначение:

* BM25 индекс;
* FAISS индекс;
* метаданные чанков;
* конфигурация индекса.

---

# Jupyter Notebook

В папке:

```text
notebooks/
```

находится ноутбук для:

* EDA;
* анализа PDF;
* проверки чанкования;
* проверки качества поиска.

---

# Пример сценария использования

1. Поместить Datasheet в `data/raw`.
2. Запустить сервис.
3. Дождаться построения индексов.
4. Открыть Swagger UI.
5. Выполнить запрос:

```json
{
  "query": "Максимальное напряжение питания MCP6001",
  "top_k": 3
}
```

6. Получить найденную характеристику с указанием страницы документа.

---

# Автор

Учебный проект по интеллектуальному поиску характеристик электронных компонентов на основе PDF Datasheet.

