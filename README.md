# ml-churn

ML-сервис для прогнозирования оттока клиентов.

## Стек

- Python 3.13
- FastAPI
- uvicorn
- uv (менеджер зависимостей)

## Установка

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

## Запуск

```bash
make run
```

Сервис будет доступен на `http://127.0.0.1:8000`.

## Команды Makefile

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

## Docker

```bash
make docker-build && make docker-up
```

Сервис будет доступен на `http://127.0.0.1:8000`.
