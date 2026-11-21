# AuditGym

AuditGym — учебная среда для ИИ-аудитора (кейс GENESIS 2026 FinTech).

Как это работает:
1. **Генератор** создаёт по номеру `seed` вымышленную компанию за один месяц: бухгалтерскую книгу, счета, платежи и курсы валют. В данные специально подмешиваются ошибки четырёх типов.
2. **Агент** (ваша программа или модель) читает эти данные через несколько инструментов, ищет ошибки и сдаёт список находок с доказательствами.
3. **Оценщик** в отдельном процессе сверяет находки с правильными ответами и выставляет награду от −1 до 1.

Правильные ответы лежат в закрытой папке `private/`. Агент её не видит.

Полная техническая спецификация — [SPEC.md](SPEC.md). Правила для ИИ-агентов, которые пишут код, — [AGENTS.md](AGENTS.md).

---

## 1. Что нужно установить

- **Python 3.11**
- **uv** — менеджер пакетов: https://docs.astral.sh/uv/getting-started/installation/
- **Docker** — только если нужен запуск в контейнере (необязательно)

GPU, интернет и платные API не нужны.

## 2. Установка

```shell
git clone <адрес репозитория>
cd genesis-hackaton
uv sync
```

`uv sync` создаёт `.venv` и ставит точные версии зависимостей из `uv.lock`.

## 3. Быстрый старт

Весь конвейер одной командой: сгенерировать данные, запустить тесты, прогнать политики и собрать отчёт.

```shell
uv run auditgym all
```

Быстрая версия: маленькие эпизоды, без тестов.

```shell
uv run auditgym all --config easy --n 2 --skip-tests
```

В конце должно появиться `all: done`. Этапы, которые ещё не реализованы, выводятся как `skipped` (см. раздел 9).

---

## 4. Основные команды

| Команда | Что делает |
|---|---|
| `uv run auditgym generate --split train --n 10 --config medium` | Генерирует 10 эпизодов в `data/train/` |
| `uv run auditgym run --policy <политика> --split dev` | Прогоняет политику на эпизодах, пишет результаты в `runs/` |
| `uv run auditgym validate --data data/` | Независимая проверка сгенерированных данных |
| `uv run auditgym transform --src ... --dst ... --permute-seed 7 --id-salt x` | Перемешивает строки и переименовывает ID эпизода (метаморфные проверки) |
| `uv run auditgym report --runs runs/` | Строит `reports/baselines.md` |
| `uv run auditgym all` | Всё по порядку |

Полезные параметры:
- `--split train|dev` выбирает набор эпизодов: `train` использует seed 10000–10009, `dev` — seed 20000–20009.
- `--config easy|medium|hard|default` задаёт сложность, настройки лежат в [configs/](configs/).
- `--mode eval` в `run` запускает режим проверки: агент всегда получает награду 0, а настоящая награда пишется только в лог оценщика `logs/grader.jsonl`.

Справка по любой команде: `uv run auditgym <команда> --help`.

---

## 5. Как выглядит эпизод

```
data/dev/ep_3f9a1c27b6d0/
├── open/              ← видит агент
│   ├── manifest.json  описание: период, валюты, размеры таблиц
│   ├── policy.json    правила учёта: какой курс брать, как нормализовать номера счетов и т.д.
│   ├── ledger.csv     бухгалтерская книга (все суммы в KZT, в тиынах)
│   ├── invoices.jsonl счета и кредит-ноты
│   ├── payments.jsonl платежи клиентов
│   └── fx.csv         курсы валют к KZT
└── private/           ← только для оценщика
    ├── truth.json     правильные ответы
    └── provenance.json как и какие ошибки были внедрены
```

Все деньги хранятся целыми числами в минимальных единицах валюты: `4500000` KZT = 45 000,00 ₸.

**Четыре типа ошибок, которые нужно найти:**

| Тип | Что это |
|---|---|
| `duplicate_invoice` | Один и тот же счёт проведён дважды под чуть изменённым номером (`INV-1042` и `INV1042`) |
| `payment_misallocation` | Платёж попал не на тот счёт или осел на «нераспределённых» (2150) |
| `fx_rate_error` | Сумма пересчитана в KZT по неправильному курсу |
| `cutoff_error` | Операция попала не в тот отчётный месяц |

Подробности — в SPEC §7.5.

---

## 6. Как подключить свою политику (агента)

Политика — это класс с двумя методами:

```python
# my_agent.py
class MyAgent:
    name = "my_agent"

    def reset(self, first_obs: dict, *, seed: int) -> None:
        """Вызывается в начале эпизода. first_obs содержит manifest, policy и список инструментов."""
        self.episode_id = first_obs["result"]["manifest"]["episode_id"]
        self.step = 0

    def act(self, obs: dict) -> dict:
        """Получает последнее наблюдение и возвращает следующее действие."""
        self.step += 1
        if self.step == 1:
            return {"tool": "list_records", "args": {"table": "invoices", "limit": 50}}
        return {"tool": "submit", "args": {"findings": {
            "episode_id": self.episode_id,
            "findings": [],
        }}}
```

Запуск из папки, где лежит `my_agent.py`:

```shell
uv run auditgym run --policy my_agent:MyAgent --split dev --n 10
```

Результаты:
- `runs/<имя>/trajectories.jsonl` — каждый шаг: действие, наблюдение, награда;
- `runs/<имя>/summary.jsonl` — итог по каждому эпизоду: precision, recall, F1, награда, число шагов.

Можно работать и напрямую из Python, без CLI:

```python
from pathlib import Path
from auditgym.env.core import AuditEnv
from auditgym.grader import GraderClient

if __name__ == "__main__":            # обязательно на Windows: оценщик запускается в отдельном процессе
    data = Path("data/dev")
    with GraderClient(data) as grader:
        env = AuditEnv(data, mode="train", grader=grader)
        obs, info = env.reset(20000, "medium")
        obs, reward, terminated, truncated, info = env.step(
            {"tool": "list_records", "args": {"table": "ledger"}})
        ...
```

Есть и обёртка под Gymnasium: [src/auditgym/env/gym_adapter.py](src/auditgym/env/gym_adapter.py).

### Инструменты агента

| Инструмент | Аргументы | Что возвращает |
|---|---|---|
| `list_records` | `table` (`ledger`/`invoices`/`payments`/`fx`), `cursor`, `limit` ≤ 50 | Страницу записей и `next_cursor` для следующей |
| `get_record` | `table`, `id` | Одну запись |
| `get_document` | `doc_id` | Счёт или платёж по `doc_id` |
| `calculator` | `expression` (например `"1000000*512.34/100"`) | Точный результат строкой |
| `submit` | `findings` | Сдаёт ответ и завершает эпизод |

На эпизод даётся не больше **100 шагов**, `submit` тоже считается шагом. Если шаги кончились без `submit`, ответ считается пустым.

### Формат находки

```json
{"episode_id": "ep_3f9a1c27b6d0",
 "findings": [{
   "type": "duplicate_invoice",
   "affected_ids": ["doc:D-k9mq1b"],
   "claimed_amount_minor": 512340000,
   "currency": "KZT",
   "evidence_ids": ["doc:D-7q2x9a", "doc:D-k9mq1b", "ledger:R-c3z8aa"],
   "explanation": "Тот же клиент, номер и сумма через 3 дня"
 }]}
```

ID в находках пишутся с префиксом: `doc:D-…` для документа, `ledger:R-…` для строки книги, `fx:2026-03-31:USD:KZT` для курса.

Находка засчитывается, только если совпадают тип, затронутые документы и валюта, сумма верна (для `fx_rate_error` допускается расхождение в 1 единицу), а доказательства включают хотя бы один полный правильный набор.

### Награда

```
reward = 2·F1 − 1 − 0.002 · (число шагов до submit)
```

- Все ошибки найдены, ложных нет — награда почти 1.
- Ничего не найдено, хотя ошибки есть, — −1.
- В чистом эпизоде без ошибок лучший ответ — пустой список. Ложное обвинение даёт −1.
- Ответ, нарушающий формат, даёт −1.
- Каждый лишний шаг немного снижает награду.

Как система реагирует на попытки её обмануть, видно в [tests/attacks/README.md](tests/attacks/README.md).

---

## 7. Тесты

```shell
uv run pytest -q
```

Полный прогон занимает около 4 минут. Отдельные наборы:

```shell
uv run pytest -q tests/env          # среда
uv run pytest -q tests/generator    # генератор
uv run pytest -q tests/grader tests/attacks tests/leakage   # оценщик, атаки, утечки
```

> **Windows:** если `uv run pytest` пишет `uv trampoline failed to canonicalize script path`, используйте `uv run python -m pytest -q`.

---

## 8. Docker

```shell
make docker
```

Команда собирает образ и запускает `auditgym all` без сети (`--network none`), с 2 CPU и 4 ГБ памяти, то есть в условиях проверки жюри.

Внутри контейнера процессы работают от разных пользователей:
- оценщик работает под пользователем `grader` и только он может читать `private/`;
- среда и политика работают под `agent`.

Даже если код агента попробует открыть файлы с ответами, операционная система ему не даст.

Другие цели Makefile: `make test` (тесты), `make all` (конвейер без Docker), `make docker-build` (только сборка).

---

## 9. Структура и роли

| Папка | Что внутри | Роль |
|---|---|---|
| `src/auditgym/world/`, `books/`, `injectors/`, `episode/serialize.py`, `configs/` | Генератор компании, проводки, внедрение ошибок | 1 Генератор |
| `src/auditgym/env/`, `logging/`, `cli.py`, `Dockerfile`, `Makefile` | Среда, инструменты, траектории, CLI | 2 Среда |
| `src/auditgym/grader/`, `schemas/` | Оценщик, схемы, сканер утечек | 3 Оценщик |
| `src/auditgym/validate/`, `policies/`, `episode/transform.py`, `reports/` | Валидатор, базовые политики, отчёты | 4 Интегратор |
| `src/auditgym/common/`, `SPEC.md` | Общий код и спецификация, правки только по согласованию | Все |

**Текущий статус.** Готовы генератор, среда, оценщик и Docker. Ещё не реализованы (роль 4): независимый валидатор, `transform`, базовые политики `oracle`/`random`/`flag_all`/`reconciler` и отчёты. Пока их нет, `auditgym all` пропускает эти этапы с пометкой `skipped`, а свою политику уже можно запускать через `--policy модуль:Класс`.

---

## 10. Частые вопросы

**Почему повторный запуск даёт те же данные?** Вся случайность выводится из `seed`, поэтому одинаковый `seed` и конфиг дают побайтово одинаковые файлы. Чтобы данные нельзя было заранее подобрать перебором seed, задайте секретную соль: `AUDITGYM_SALT=... uv run auditgym generate ...`.

**Куда пишутся логи оценщика?** В `logs/grader.jsonl`. Агенту этот файл недоступен.

**Как проверить, что в траекториях нет утечек?**
`uv run python -m auditgym.grader.leakage data/dev/<episode_id> runs/<run>/trajectories.jsonl`

## Лицензия

Apache-2.0.
