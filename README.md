# ml-churn

ML-сервис для прогнозирования оттока клиентов (churn prediction).

## 🎯 О проекте

Сервис предсказывает, **уйдёт ли клиент из сервиса**, на основе его поведения и характеристик.

| | |
|---|---|
| **Тип задачи** | Бинарная классификация |
| **Модели** | Logistic Regression, Random Forest (настраивается) |
| **Целевая переменная** | `churn` (0 — остался, 1 — ушёл) |
| **Датасет** | 2000 записей, 9 признаков |

## 🧠 ML Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      ML Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Данные ──→ ColumnTransformer ──→ Модель ──→ churn          │
│                  │                    │                      │
│                  │                    ├── LogisticRegression │
│                  │                    └── RandomForest       │
│                  │                                          │
│                  ├── StandardScaler (числовые)              │
│                  └── OneHotEncoder (категориальные)         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Поддерживаемые модели

| Модель | Описание | Когда использовать |
|--------|----------|-------------------|
| `logreg` | Logistic Regression | Базовая модель, быстрая, интерпретируемая |
| `random_forest` | Random Forest | Лучше работает со сложными зависимостями |

> **Результаты сравнения:** Random Forest показывает F1=0.175 против F1=0.045 у Logistic Regression (в 4 раза лучше для churn prediction).

### Признаки

**Числовые (6)** — масштабируются через `StandardScaler`:

| Признак | Описание |
|---------|----------|
| `monthly_fee` | Ежемесячная плата за сервис |
| `usage_hours` | Часы использования сервиса |
| `support_requests` | Количество обращений в поддержку |
| `account_age_months` | Возраст аккаунта в месяцах |
| `failed_payments` | Количество неудачных платежей |
| `autopay_enabled` | Включён ли автоплатёж (0/1) |

**Категориальные (3)** — кодируются через `OneHotEncoder`:

| Признак | Значения |
|---------|----------|
| `region` | america, europe, asia, africa |
| `device_type` | mobile, desktop, tablet |
| `payment_method` | card, paypal, crypto |

## 📊 Метрики

| Метрика | Описание |
|---------|----------|
| **Accuracy** | Доля правильных предсказаний среди всех |
| **F1-score** | Гармоническое среднее precision и recall |
| **ROC-AUC** | Площадь под ROC-кривой, качество разделения классов |

> **Почему F1 важен?** Датасет несбалансирован (~80% остались, ~20% ушли). Accuracy может вводить в заблуждение — модель может просто предсказывать "не уйдёт" и получить 80%, но не ловить реальных churn-клиентов.

> **Зачем ROC-AUC?** Показывает, насколько хорошо модель разделяет классы до выбора порога. Полезно для сравнения моделей между собой.

## 🚀 Быстрый старт

### Обучение модели с дефолтным конфигом (LogisticRegression)

```bash
curl -X POST http://127.0.0.1:8000/model/train
```

**Ответ:**
```json
{
  "accuracy": 0.79,
  "f1": 0.04,
  "roc_auc": 0.61
}
```

### Обучение с Random Forest и гиперпараметрами

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "random_forest",
    "hyperparameters": {"n_estimators": 100}
  }'
```

**Ответ:**
```json
{
  "accuracy": 0.79,
  "f1": 0.17,
  "roc_auc": 0.59
}
```

### Проверка статуса модели

```bash
curl http://127.0.0.1:8000/model/status
```

**Ответ:**
```json
{
  "is_trained": true,
  "trained_at": "2026-08-02T07:59:52.699997Z",
  "accuracy": 0.79,
  "f1": 0.17,
  "roc_auc": 0.59,
  "model_type": "random_forest",
  "hyperparameters": {"n_estimators": 100}
}
```

### История обучений и метрики

```bash
# Все записи истории
curl http://127.0.0.1:8000/model/metrics

# Фильтрация по типу модели
curl "http://127.0.0.1:8000/model/metrics?model_type=logreg"

# Ограничить количество записей
curl "http://127.0.0.1:8000/model/metrics?limit=5"
```

**Ответ:**
```json
{
  "latest": {
    "timestamp": "2026-08-02T09:40:02.526865Z",
    "model_type": "random_forest",
    "hyperparameters": {"n_estimators": 50},
    "accuracy": 0.78,
    "f1": 0.14,
    "roc_auc": 0.57
  },
  "history": [
    {
      "timestamp": "2026-08-02T09:40:01.393021Z",
      "model_type": "logreg",
      "hyperparameters": {},
      "accuracy": 0.79,
      "f1": 0.04,
      "roc_auc": 0.61
    },
    {
      "timestamp": "2026-08-02T09:40:02.526865Z",
      "model_type": "random_forest",
      "hyperparameters": {"n_estimators": 50},
      "accuracy": 0.78,
      "f1": 0.14,
      "roc_auc": 0.57
    }
  ]
}
```

> **Как это работает:** При каждом `/model/train` создаётся запись в истории обучений (сохраняется в `models/training_history.json`). Эндпоинт `/model/metrics` позволяет сравнивать результаты разных обучений и фильтровать по типу модели.

### Предсказание для нового клиента

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_fee": 49.99,
    "usage_hours": 5.0,
    "support_requests": 8,
    "account_age_months": 2,
    "failed_payments": 5,
    "region": "africa",
    "device_type": "mobile",
    "payment_method": "crypto",
    "autopay_enabled": 0
  }'
```

**Ответ:**
```json
{
  "prediction": 1,
  "probability_churn_0": 0.48,
  "probability_churn_1": 0.52,
  "features": {...}
}
```

### Получение схемы признаков

```bash
curl http://127.0.0.1:8000/model/schema
```

**Ответ:**
```json
{
  "features": [
    {"name": "monthly_fee", "type": "numeric", "description": "Ежемесячная плата за сервис"},
    {"name": "usage_hours", "type": "numeric", "description": "Часы использования сервиса"},
    {"name": "support_requests", "type": "numeric", "description": "Количество обращений в поддержку"},
    {"name": "account_age_months", "type": "numeric", "description": "Возраст аккаунта в месяцах"},
    {"name": "failed_payments", "type": "numeric", "description": "Количество неудачных платежей"},
    {"name": "autopay_enabled", "type": "numeric", "description": "Включён ли автоплатёж (0/1)"},
    {"name": "region", "type": "categorical", "description": "Регион клиента (america, europe, asia, africa)"},
    {"name": "device_type", "type": "categorical", "description": "Тип устройства (mobile, desktop, tablet)"},
    {"name": "payment_method", "type": "categorical", "description": "Метод оплаты (card, paypal, crypto)"}
  ]
}
```

> **Зачем нужен `/model/schema`?** Клиенты могут получить список признаков, их типы и описания, чтобы правильно формировать запросы к `/predict`.

### Как это работает

1. Загружаются данные из `data/churn_dataset.csv`
2. Данные разделяются на train (80%) и test (20%) с стратификацией
3. Pipeline обучается на train: предобработка → модель
4. Метрики вычисляются на test (честная оценка)
5. Модель сохраняется в `models/churn_model.joblib` вместе с конфигом
6. При перезапуске сервиса модель автоматически загружается

## 📡 API Endpoints

### Модель

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/model/train` | Обучение модели с конфигурацией |
| GET | `/model/status` | Статус модели (тип, метрики, гиперпараметры) |
| GET | `/model/metrics` | История обучений и метрики (фильтрация по `model_type`, `limit`) |
| GET | `/model/schema` | Схема признаков модели (название, тип, описание) |

### Предсказания

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/predict` | Предсказание для одного клиента или списка |

### Датасет

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/dataset/preview` | Превью данных (n строк) |
| GET | `/dataset/info` | Информация о датасете |
| GET | `/dataset/split-info` | Информация о train/test split |

### Основные

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Проверка работоспособности |
| GET | `/docs` | Swagger UI документация |

## 🛠 Стек

- **Python 3.13**
- **FastAPI** — веб-фреймворк
- **scikit-learn** — ML библиотека
- **pandas** — работа с данными
- **uvicorn** — ASGI сервер
- **uv** — менеджер зависимостей

## 📦 Установка

```bash
uv sync
make install
```

<details>
<summary>Если нет make</summary>

- **macOS:** `xcode-select --install`
- **Linux:** `sudo apt install make`
- **Windows:** `winget install GnuWin32.Make` или через Chocolatey: `choco install make`

</details>

## ▶️ Запуск

```bash
make run
```

Сервис будет доступен на `http://127.0.0.1:8000`.

## 🐳 Docker

```bash
make docker-build && make docker-up
```

## ⚙️ Команды Makefile

| Команда          | Описание                          |
|------------------|-----------------------------------|
| `make install`   | Установить зависимости            |
| `make run`       | Запустить сервер                  |
| `make dev`       | Запустить сервер с автоперезагрузкой |
| `make test`      | Запустить тесты                   |
| `make lint`      | Проверить код (ruff)              |
| `make typecheck` | Проверить типы (mypy)             |
| `make format`    | Форматировать код (ruff format)   |
| `make check`     | lint + typecheck + test           |
| `make docker-build` | Собрать Docker-образ           |
| `make docker-up` | Запустить контейнеры              |
| `make docker-down` | Остановить контейнеры           |

## 📁 Структура проекта

```
src/
├── main.py                    # FastAPI приложение, lifespan, exception handler
├── schemas.py                 # Pydantic модели (схемы)
├── api/
│   ├── dependencies.py        # Dependency injection (get_dataset_repo, get_loaded_model_repo, get_training_history_repo)
│   └── routes/
│       ├── dataset.py         # Эндпоинты для работы с датасетом
│       ├── model.py           # Эндпоинты для обучения, статуса и метрик модели
│       └── predict.py         # Эндпоинт для предсказаний
├── dataset/
│   ├── preprocessing.py       # Предобработка данных, train/test split
│   └── repository.py          # Загрузка и хранение датасета
└── model/
    ├── training.py            # ML pipeline, обучение модели, ModelMetrics, TrainingHistoryEntry
    ├── repository.py          # Сохранение и загрузка модели
    └── history_repository.py  # История обучений (абстрактный репозиторий, InMemory, JSON)

tests/
├── conftest.py                # Общие fixtures
├── test_schemas.py            # Тесты Pydantic моделей
├── test_main.py               # Тесты root endpoint
├── test_dataset.py            # Тесты dataset (unit + API)
├── test_model.py              # Тесты model (unit + API)
├── test_predict.py            # Тесты predict (API)
└── test_training_history.py   # Тесты истории обучений (unit + API)

models/                        # Сохранённые модели и история (gitignored)
├── churn_model.joblib         # Обученная модель с конфигом
└── training_history.json      # История обучений

data/
└── churn_dataset.csv          # Датасет (2000 записей)
```
