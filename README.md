# Credit Default Prediction (PD-Model)

Сквозной автоматизированный пайплайн для PD-модели предсказания дефолта клиентов на датасете UCI Credit Card.

## Структура проекта

```
├── data/
│   ├── raw/           # Исходные данные (UCI_Credit_Card.csv)
│   ├── processed/     # train.csv, test.csv
│   └── expectations/  # Правила валидации
├── models/            # Обученные модели
├── notebooks/         # EDA и эксперименты
├── src/
│   ├── data/          # make_dataset.py, validation.py
│   ├── features/      # build_features.py
│   ├── models/        # pipeline.py, train.py, predict.py
│   └── api/           # FastAPI приложение
├── tests/             # Unit-тесты
├── scripts/           # Скрипты запуска и мониторинга
├── .github/workflows/ # CI/CD
├── dvc.yaml           # DVC pipeline
├── Dockerfile
└── requirements.txt
```

## Установка

```bash
pip install -r requirements.txt
```

## Запуск пайплайна

### 1. Подготовка данных

```bash
python src/data/make_dataset.py data/raw/UCI_Credit_Card.csv data/processed/
```

### 2. Обучение модели

```bash
python src/models/train.py
```

Модель сохраняется в `models/credit_default_model.pkl`, метрики - в `metrics.json` и MLflow

### 3. DVC pipeline (опционально)

```bash
dvc init
dvc repro
```

### 4. Запуск API

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Документация: http://localhost:8000/docs

### 5. Docker

```bash
# Сначала обучите модель (шаги 1–2)
docker build -t credit-api .
docker run -p 8000:8000 credit-api
```

Или используйте скрипт: `./scripts/build_and_run.sh` (Linux/Mac) или `scripts\build_and_run.ps1` (Windows).

### 6. Пример запроса к API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 20000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_0": 2, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
    "BILL_AMT5": 0, "BILL_AMT6": 0,
    "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
    "PAY_MEAN": -1.0, "PAY_MAX": 2, "PAY_MIN": -2,
    "BILL_AMT_MEAN": 1301.0, "BILL_AMT_MAX": 3913,
    "PAY_AMT_MEAN": 114.83, "PAY_AMT_SUM": 689, "PAY_TO_BILL_RATIO": 0.088
  }'
```

### 7. Мониторинг дрифта

```bash
# Запустите API, затем:
python scripts/drift_monitor.py --api-url http://localhost:8000
```

## Тестирование

```bash
pytest tests -v
black --check src tests
flake8 src tests --max-line-length=88
```

## MLflow

Просмотр экспериментов:

```bash
mlflow ui
```

Откройте http://localhost:5000
