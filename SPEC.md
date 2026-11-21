# AuditGym — техническая спецификация (v1.0)

> Единственный источник правды для людей и ИИ-агентов (Codex, Claude Code).
> Если код расходится со SPEC.md — прав SPEC.md. Изменения спецификации — только через PR с пометкой `spec:` и согласие всей команды.

---

## 0. Суть

AuditGym — воспроизводимая RL-среда для ИИ-аудитора (кейс GENESIS 2026 FinTech).

1. **Генератор** по `seed + config` создаёт вымышленную компанию за один отчётный период: проводки, счета, платежи, курсы валют. В данные внедряются 0–12 ошибок четырёх типов. Правильные ответы пишутся в закрытые файлы.
2. **Среда** (`reset/step`, JSON-контракт в стиле Gymnasium) даёт агенту узкий набор инструментов только для чтения.
3. **Оценщик** в отдельном процессе сопоставляет выводы агента с эталоном и считает награду.
4. **Базовые политики** (oracle, random, flag_all, reconciler) показывают, что оценка честная.

**Цели (= критерии жюри):** контракт и изоляция, корректность данных и эталона, устойчивость награды, воспроизводимость и обобщение, инженерия.

**Не цели:** обучение модели, UI, реальные финансовые данные, сетевые API. Обучение весов баллов не даёт.

---

## 1. Жёсткие ограничения

| Параметр | Значение |
|---|---|
| Язык | Python 3.11 |
| Зависимости | `uv`, `uv.lock` обязателен; runtime: `numpy`, `jsonschema`, `pyyaml`; dev: `pytest`, `hypothesis`; опционально `gymnasium` |
| Ресурсы | 2 CPU, 4 ГБ RAM, без GPU, ≤1 ГБ на решение, ≤60 с на эпизод |
| Сеть | Запрещена при проверке (`docker run --network none`) |
| Шаги | ≤100 на эпизод, включая `submit` |
| Лицензия | Apache-2.0 |

### Правила кода (обязательны для всех агентов)

- Деньги — только `int` в минимальных единицах. Курсы — только `decimal.Decimal`, создаваемый из строки. `float` для денег и курсов запрещён.
- Округление — `ROUND_HALF_EVEN`, ровно в одном месте: `common/money.py::convert()`.
- Запрещены `hash()`, итерация по `set`/`dict` без `sorted()`, `datetime.now()`, глобальный `random`, `uuid4()`.
- Вся случайность идёт только из `common/rng.py` (именованные подпотоки).
- JSON: `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`. CSV: фиксированный порядок колонок, `\n`, UTF-8.
- Время — ISO-8601 с явным смещением (`2026-03-31T23:59:00+05:00`). Только фиксированные смещения, без tzdata.
- Сообщения об ошибках для агента не содержат путей, трейсбеков, ID из эталона и любых данных из `private/`.

---

## 2. Структура репозитория

```
auditgym/
├── SPEC.md  AGENTS.md  CLAUDE.md  README.md  DESIGN.md  LICENSE
├── pyproject.toml  uv.lock  Makefile  Dockerfile  .github/workflows/ci.yml
├── configs/
│   ├── default.yaml
│   └── difficulty/{easy,medium,hard}.yaml
├── schemas/                       # JSON Schema (draft 2020-12)
│   ├── manifest.schema.json  policy.schema.json  ledger_row.schema.json
│   ├── invoice.schema.json   payment.schema.json fx_row.schema.json
│   ├── action.schema.json    observation.schema.json
│   ├── findings.schema.json  truth.schema.json   trajectory.schema.json
├── src/auditgym/
│   ├── common/        # ОБЩЕЕ — правки только по согласованию
│   │   ├── types.py   # dataclasses: TruthItem, Mutation, GradeResult, Episode
│   │   ├── rng.py     # SeedSequence, именованные подпотоки
│   │   ├── money.py   # exponent(), convert(), parse_rate()
│   │   ├── ids.py     # генерация непрозрачных ID, префиксы
│   │   ├── timeutil.py
│   │   └── io.py      # канонический JSON/CSV, sha256 каталога
│   ├── world/         # Роль 1: компания, контрагенты, календарь, FX-рынок, события
│   ├── books/         # Роль 1: документы, проводки, инварианты
│   ├── injectors/     # Роль 1: base.py registry.py + 4 типа + interactions/ + distractors/
│   ├── episode/       # Роль 1: serialize.py; Роль 4: transform.py
│   ├── env/           # Роль 2: core.py tools.py pagination.py calculator.py gym_adapter.py
│   ├── grader/        # Роль 3: schema.py normalize.py matcher.py reward.py service.py client.py
│   ├── validate/      # Роль 4: independent_checker.py
│   ├── policies/      # Роль 4: oracle.py random_explorer.py flag_all.py reconciler.py
│   ├── logging/       # Роль 2: trajectory.py
│   └── cli.py         # Роль 2
├── tests/{common,generator,env,grader,attacks,leakage,metamorphic,validate,e2e}/
├── data/{train,dev}/ep_XX/{open/,private/}
└── reports/{baselines.md,failure_analysis.md}
```

---

## 3. Предметная модель

### 3.1 Компания и период

- Одна компания, `base_currency = "KZT"` (exponent 2), `timezone = "+05:00"`.
- Период — один календарный месяц, задаётся в config (по умолчанию `2026-03-01..2026-03-31`).
- Граница периода: `[period_start 00:00:00, period_end 23:59:59.999]` в TZ компании.
- **Хвост:** в `ledger.csv`, `invoices.jsonl` и `payments.jsonl` включаются записи первых `tail_days = 3` дней следующего периода (нужно, чтобы были видны cut-off ошибки в обе стороны). Это объявлено в `policy.json`.

### 3.2 Валюты

| Код | Exponent |
|---|---|
| KZT, USD, EUR | 2 |
| JPY | 0 |
| KWD | 3 |

### 3.3 План счетов

| Код | Название | Нормальное сальдо |
|---|---|---|
| 1010 | Cash | Дт |
| 1200 | Accounts Receivable | Дт |
| 2150 | Unapplied Cash | Кт |
| 4000 | Revenue | Кт |
| 4100 | Sales Returns | Дт |
| 6100 | Bank Fees | Дт |
| 7100 | FX Gain/Loss | Дт/Кт |

### 3.4 Правила проводок (корректная бухгалтерия)

Все суммы в `ledger.csv` — **в базовой валюте**, колонка `currency` всегда равна `base_currency`. Конвертация: `convert(amount_minor, src, dst, rate)` по курсу `fx_rate_date` (см. `policy.json`).

| Событие | Проводки (база) | Ссылки |
|---|---|---|
| Счёт выставлен | Дт 1200 / Кт 4000 на `convert(total, rate(issued_date))` | `invoice_id`, `entity_id` |
| Платёж получен | Дт 1010 на `convert(amount, rate(pay_date))` | `payment_id` |
| Распределение платежа на счёт | Кт 1200 на `convert(alloc, rate(invoice_issued_date))` | `invoice_id`, `payment_id` |
| Курсовая разница | Дт/Кт 7100 на разницу между Дт 1010 и Σ Кт 1200 + Кт 2150 | `payment_id` |
| Нераспределённый остаток (переплата) | Кт 2150 | `payment_id` |
| Банковская комиссия | Дт 6100 / уменьшение Дт 1010 (платёж приходит на сумму − fee) | `payment_id` |
| Возврат / кредит-нота | Дт 4100 / Кт 1200 | `invoice_id` (исходного счёта) |

`timestamp` проводки = время события документа, переведённое в TZ компании.

### 3.5 Инварианты чистой книги (assert в генераторе и независимо в валидаторе)

- **I1:** для каждой группы проводок (по `payment_id`, иначе по `invoice_id`) Σ debit = Σ credit.
- **I2:** Σ всех debit = Σ всех credit.
- **I3:** для каждого счёта: сальдо 1200 по `invoice_id` = `convert(total)` − Σ распределений − Σ кредит-нот ≥ 0.
- **I4:** для каждого платежа: `amount` = Σ `invoice_refs[].amount_minor` + unapplied + fee (в валюте платежа).
- **I5:** каждый документ проведён ровно один раз, `timestamp` проводки лежит в том же периоде, что и дата документа (в TZ компании).
- **I6:** каждая ссылка (`invoice_id`, `payment_id`, `entity_id`) существует или явно `null`.
- **I7:** для каждой нужной даты есть курс согласно правилу выходных.

---

## 4. Формат эпизода на диске

```
ep_XX/
├── open/      # видит агент (через env)
│   ├── manifest.json
│   ├── policy.json
│   ├── ledger.csv
│   ├── invoices.jsonl
│   ├── payments.jsonl
│   └── fx.csv
└── private/   # видит только грейдер
    ├── truth.json
    └── provenance.json
```

### 4.1 manifest.json

```json
{
  "schema_version": "1.0",
  "config_version": "medium-1.0",
  "generator_version": "1.0.0",
  "episode_id": "ep_3f9a1c27b6d0",
  "base_currency": "KZT",
  "currency_exponents": {"KZT": 2, "USD": 2, "EUR": 2, "JPY": 0, "KWD": 3},
  "period_start": "2026-03-01T00:00:00+05:00",
  "period_end": "2026-03-31T23:59:59+05:00",
  "timezone": "+05:00",
  "counts": {"ledger": 812, "invoices": 64, "payments": 51, "fx": 155},
  "rounding": "ROUND_HALF_EVEN"
}
```

`episode_id` = `"ep_" + sha256(seed‖config_hash‖version‖salt)[:12]`. Количество ошибок в manifest **не** указывается.

### 4.2 policy.json

```json
{
  "fx_rate_date": "document_date_in_company_tz",
  "fx_weekend_rule": "last_available_on_or_before",
  "fx_direction": "rate = units of target per 1 unit of source",
  "period_boundary_tz": "+05:00",
  "tail_days": 3,
  "ledger_currency": "base",
  "finding_amount_currency": "base",
  "duplicate_rule": "the later-issued of two invoices with same entity, normalized number and amount is the duplicate",
  "invoice_number_normalization": "uppercase, remove spaces/dashes/underscores, strip leading zeros of numeric part",
  "rounding": "ROUND_HALF_EVEN",
  "amount_tolerance_minor": {"fx_rate_error": 1, "default": 0}
}
```

### 4.3 ledger.csv

Колонки строго в этом порядке:

```
row_id,timestamp,account,debit_minor,credit_minor,currency,entity_id,invoice_id,payment_id
R-8k2m1q,2026-03-04T10:15:00+05:00,1200,4500000,0,KZT,E-19ad,INV-1042,
```

- Ровно одно из `debit_minor` / `credit_minor` > 0, второе = 0.
- Пустое поле в CSV = `null`.

### 4.4 invoices.jsonl

```json
{"doc_id":"D-7q2x9a","invoice_id":"INV-1042","entity_id":"E-19ad","issued_at":"2026-03-04T10:15:00+05:00","due_at":"2026-04-03T23:59:59+05:00","currency":"USD","total_minor":1000000,"line_items":[{"description":"Subscription March","quantity":1,"unit_price_minor":1000000,"service_period":"2026-03"}],"kind":"invoice","original_invoice_id":null}
```

- `kind ∈ {"invoice","credit_note"}`; для кредит-ноты `original_invoice_id` заполнен, `total_minor > 0` (знак задаёт `kind`).
- Σ `line_items[].quantity * unit_price_minor` = `total_minor`.

### 4.5 payments.jsonl

```json
{"doc_id":"D-k1p0zz","payment_id":"P-3x7a","entity_id":"E-19ad","timestamp":"2026-03-10T09:00:00-04:00","currency":"USD","amount_minor":995000,"fee_minor":5000,"invoice_refs":[{"invoice_id":"INV-1042","amount_minor":1000000}]}
```

- `invoice_refs` — намерение клиента (remittance). Это первичный документ, по нему проверяется распределение в ledger.
- `amount_minor` — сколько дошло до банка: Σ `invoice_refs` − `fee_minor` + переплата.

### 4.6 fx.csv

```
date,source_currency,target_currency,rate
2026-03-31,USD,KZT,512.340000
```

- Только рабочие дни, 6 знаков после точки, строка.
- Только пары `X→KZT`.
- Ключ курса в доказательствах: `fx:2026-03-31:USD:KZT`.

### 4.7 truth.json (private)

```json
{
  "episode_id": "ep_3f9a1c27b6d0",
  "issues": [
    {
      "issue_id": "iss_01",
      "type": "duplicate_invoice",
      "affected_ids": ["doc:D-k9mq1b"],
      "expected_amount_minor": 512340000,
      "currency": "KZT",
      "evidence_sets": [
        ["doc:D-7q2x9a", "doc:D-k9mq1b", "ledger:R-c3z8aa"],
        ["doc:D-7q2x9a", "doc:D-k9mq1b", "ledger:R-d8y1bb"]
      ]
    }
  ]
}
```

### 4.8 provenance.json (private)

```json
{
  "seed": 42, "config": {}, "config_hash": "…", "generator_version": "1.0.0",
  "stream_keys": ["world","docs","ids","shuffle","inj.duplicate_invoice", "..."],
  "mutations": [
    {"issue_ids": ["iss_01"], "injector": "duplicate_invoice",
     "causal_note": "Invoice INV-1042 re-entered 3 days later as INV1042",
     "diff": {"added": ["doc:D-k9mq1b","ledger:R-c3z8aa","ledger:R-d8y1bb"], "removed": [], "modified": []},
     "decomposition": null}
  ],
  "distractors": [{"kind": "recurring_subscription", "ids": ["doc:…","doc:…"]}]
}
```

---

## 5. ID и префиксы

| Сущность | Формат на диске | Формат в выводах |
|---|---|---|
| Строка книги | `R-` + 6 символов `[a-z0-9]` | `ledger:R-…` |
| Документ | `D-` + 6 символов | `doc:D-…` |
| Номер счёта | `INV-` + 4 цифры (вариации только у дублей) | — (не используется в выводах) |
| Платёж | `P-` + 4 символа | — |
| Контрагент | `E-` + 4 символа | — |
| Курс | — | `fx:YYYY-MM-DD:SRC:DST` |

- ID генерируются только из подпотока `ids`, без коллизий (повтор при коллизии).
- В выводах (`affected_ids`, `evidence_ids`) допустимы только префиксы `ledger:`, `doc:`, `fx:`.
- ID не должны раскрывать ни порядок генерации, ни факт инъекции.

---

## 6. Детерминизм и seed

```python
# common/rng.py
root = np.random.SeedSequence([seed, crc32(config_hash), crc32(GENERATOR_VERSION), crc32(salt)])
def stream(name: str) -> np.random.Generator:
    key = int.from_bytes(sha256(name.encode()).digest()[:4], "big")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(root.entropy, spawn_key=(key,))))
```

- Подпотоки: `world`, `events`, `docs`, `ids`, `shuffle`, `distract`, `inj.<type>`, `inter.<name>`. Включение или выключение одного инжектора не меняет данные других подпотоков.
- `salt = os.environ.get("AUDITGYM_SALT", "")`. В открытых данных соль пустая. Жюри может задать свою, тогда перебор seed по публичному генератору становится бесполезен.
- **Требование R1:** одинаковые `seed + config + version + salt` дают побайтово одинаковые файлы (sha256 каталога).
- **Требование R2:** одинаковые эпизод и последовательность действий дают одинаковые наблюдения и награды.

---

## 7. Генератор (Роль 1)

### 7.1 Публичный API

```python
def generate_episode(seed: int, config: dict, out_dir: Path) -> Path:
    """Создаёт out_dir/{open,private}. Возвращает out_dir. Детерминирован."""
```

### 7.2 Конвейер

```
config → validate_config
       → world.build(rng.world)            # компания, контрагенты (tz, валюта), FX-рынок
       → world.events(rng.events)          # календарь счетов/платежей/возвратов/комиссий
       → books.post(events, policy)        # документы + проводки
       → books.check_invariants()          # I1–I7, при провале — исключение
       → distractors.apply(rng.distract)   # корректные «подозрительные» случаи, инварианты сохраняются
       → books.check_invariants()
       → injectors.apply(plan, rng.inj.*)  # мутации + truth_items + provenance
       → check_declared_discrepancies()    # каждое объявленное расхождение = фактическая дельта
       → serialize(rng.ids, rng.shuffle)   # непрозрачные ID, перемешивание строк, запись файлов
```

### 7.3 Мир

- Контрагенты: `n_entities` (5–40). У каждого: валюта из `config.currencies`, tz из `{-08:00,-05:00,-04:00,-03:00,+00:00,+01:00,+03:00,+05:00,+09:00}` с учётом `tz_spread`.
- FX: для каждой пары `X→KZT` — геометрическое блуждание от базового курса (USD 500, EUR 545, JPY 3.4, KWD 1630), дневная волатильность 0.4%, 6 знаков. Выходные пропускаются. Курсы генерируются с запасом на 10 дней до периода и `tail_days` после.

### 7.4 Интерфейс инжектора

```python
class Injector(Protocol):
    type: str                                            # "duplicate_invoice"
    def eligible(self, books: Books) -> list[Candidate]: ...
    def apply(self, cand: Candidate, books: Books, rng: Generator) -> Mutation: ...

@dataclass(frozen=True)
class TruthItem:
    issue_id: str
    type: str
    affected_ids: tuple[str, ...]          # отсортировано
    expected_amount_minor: int             # всегда в base_currency
    currency: str                          # = base_currency
    evidence_sets: tuple[tuple[str, ...], ...]

@dataclass(frozen=True)
class Mutation:
    truth_items: tuple[TruthItem, ...]
    diff: dict
    causal_note: str
    decomposition: dict | None
```

Регистрация: `@register("duplicate_invoice")`. Каждый инжектор независимо включается через `config.enabled_types`.

### 7.5 Четыре типа ошибок

Во всех случаях `currency` в эталоне = `base_currency`, а сумма считается в базе.

#### T1 `duplicate_invoice`

- **Кандидат:** счёт `kind=invoice`.
- **Мутация:** новый документ с тем же `entity_id`, `currency`, `total_minor`, `line_items`. `invoice_id` — вариация номера (`INV-1042` → `INV1042` / `inv-1042` / `INV-01042`, нормализуется в тот же номер). `issued_at` позже оригинала на 1–10 дней (в пределах периода). Плюс пара проводок Дт 1200 / Кт 4000.
- **affected_ids:** `[doc:<дубль>]`.
- **Сумма:** база по проводке дубля.
- **evidence_sets:** `[doc:<оригинал>, doc:<дубль>, ledger:<Дт 1200 дубля>]` и `[doc:<оригинал>, doc:<дубль>, ledger:<Кт 4000 дубля>]`.

#### T2 `payment_misallocation`

- **Кандидат:** платёж с ≥1 `invoice_refs`.
- **Мутация (варианты):**
  - (a) распределение на счёт X полностью ушло в Кт 2150 вместо Кт 1200;
  - (b) на Кт 1200 по X проведено меньше, чем в `invoice_refs`, разница в Кт 2150.
- **affected_ids:** `[doc:<платёж>, doc:<счёт X>]`.
- **Сумма:** нераспределённая часть в базе.
- **evidence_sets:** `[doc:<платёж>, doc:<счёт X>, ledger:<строка Кт 2150>]`.

#### T3 `fx_rate_error`

- **Кандидат:** валютный документ (счёт или платёж), предпочтительно у контрагента, чья локальная дата отличается от даты в TZ компании.
- **Мутация (варианты):**
  - (a) курс по дате в TZ контрагента;
  - (b) курс соседнего рабочего дня;
  - (c) инвертированный курс (1/rate).
- Пересчитываются суммы проводок документа. Баланс группы сохраняется: разница уходит в 7100 для платежа или пропорционально меняет обе стороны для счёта.
- **affected_ids:** `[doc:<документ>]`.
- **Сумма:** `|posted_base − correct_base|` по строке 1200 (для счёта) или 1010 (для платежа).
- **evidence_sets:** `[doc:<документ>, ledger:<строка 1200/1010>, fx:<правильная дата>:<CCY>:KZT]`.

#### T4 `cutoff_error`

- **Кандидат:** документ в последние 2 дня периода или в хвосте.
- **Мутация:** `timestamp` проводок документа переносится через границу периода (внутрь из хвоста или из периода в хвост), при этом документ не меняется.
- **affected_ids:** `[doc:<документ>]`.
- **Сумма:** база по строке 4000 (счёт) или 1010 (платёж).
- **evidence_sets:** `[doc:<документ>, ledger:<строка 4000 или 1010>]`.

### 7.6 Взаимодействующие ошибки

Для каждого сценария нужен отдельный модуль в `injectors/interactions/` и правило `decomposition` в provenance.

**IA `fx_partial_boundary`:** валютный частичный платёж. В TZ клиента это 31-е число, в TZ компании уже 1-е число следующего месяца. В книге платёж проведён в отчётном периоде (cut-off) по курсу последнего дня периода (FX).
- Эталон: `cutoff_error` с суммой = `convert(amount, правильный курс)` и `fx_rate_error` с суммой = `|convert(amount, неверный) − convert(amount, правильный)|`.
- Оба `affected_ids = [doc:<платёж>]`. Правило декомпозиции описано в DESIGN.md.

**IB `duplicate_with_payment`:** создаётся дубль T1, и платёж клиента (`invoice_refs` указывает на оригинал) в книге распределён на `invoice_id` дубля.
- Эталон: `duplicate_invoice` (как в T1) и `payment_misallocation` с `affected_ids = [doc:<платёж>, doc:<оригинал>]`, суммой = распределённая сумма в базе, доказательствами `[doc:<платёж>, doc:<оригинал>, ledger:<Кт 1200 с invoice_id дубля>]`.

### 7.7 Отвлекающие случаи (корректны, ошибками не являются)

| Вид | Что генерируется |
|---|---|
| `recurring_subscription` | Одинаковая сумма тому же контрагенту, разные `service_period` и номера |
| `split_payment` | Два платежа закрывают один счёт |
| `multi_invoice_payment` | Один платёж на несколько счетов |
| `partial_open` | Честный частичный платёж, счёт остаётся открытым |
| `overpayment` | Переплата корректно в Кт 2150, отражена в `amount > Σ refs` |
| `bank_fee` | `fee_minor > 0`, Дт 6100 |
| `credit_note` | Возврат со ссылкой на исходный счёт |
| `boundary_correct` | Документы у границы периода, проведённые правильно, клиенты в других TZ |
| `exotic_exponent` | Документы в JPY и KWD |

`distractor_rate` (0–1) задаёт долю документов, участвующих в отвлекающих случаях. В **чистых** эпизодах (`n_issues = 0`) отвлекающие случаи обязательны.

### 7.8 Сериализация

- Присвоить итоговые ID из подпотока `ids`.
- Перемешать порядок строк каждой таблицы (`shuffle`), затем отсортировать ledger по `(timestamp, row_id)`. Так порядок не выдаёт инъекции.
- Внедрённые записи по формату неотличимы от обычных: та же точность, тот же генератор времени (минуты кратны 1, секунды 00).

### 7.9 Config

```yaml
config_version: "medium-1.0"
period: {start: "2026-03-01", end: "2026-03-31"}
n_entities: [10, 25]          # диапазон, конкретное значение из rng.world
n_invoices: [40, 120]
n_ledger_target: [400, 1500]  # итог обязан попасть в 200–2000
n_issues: [0, 8]              # 0–12
clean_probability: 0.2
enabled_types: [duplicate_invoice, payment_misallocation, fx_rate_error, cutoff_error]
interactions: {fx_partial_boundary: 0.3, duplicate_with_payment: 0.3}
distractor_rate: 0.25
currencies: [KZT, USD, EUR, JPY, KWD]
foreign_share: 0.4
tz_spread: 0.6
```

Пресеты: `easy` (1–2 валюты, без взаимодействий, `distractor_rate` 0.1), `medium` (выше), `hard` (все валюты, `distractor_rate` 0.4, взаимодействия 0.6, до 12 ошибок).

---

## 8. Среда (Роль 2)

### 8.1 API

```python
class AuditEnv:
    def __init__(self, data_root: Path, mode: Literal["train", "eval"] = "train",
                 grader: GraderClient | None = None, max_steps: int = 100): ...
    def reset(self, seed: int, config: dict | str) -> tuple[dict, dict]: ...
    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]: ...
    def close(self) -> None: ...
```

- `reset` генерирует эпизод (или берёт из кэша `data_root/<episode_id>`), загружает `open/` в память только для чтения и полностью сбрасывает состояние.
- `config` — dict или имя пресета.

### 8.2 Первое наблюдение

```json
{"ok": true, "error": null, "steps_used": 0, "steps_left": 100,
 "result": {
   "task": "Audit the books for the period. Find errors of types [...]. Submit findings with evidence.",
   "manifest": {"...": "manifest.json"},
   "policy": {"...": "policy.json"},
   "tables": {"ledger": 812, "invoices": 64, "payments": 51, "fx": 155},
   "tools": ["list_records","get_record","get_document","calculator","submit"],
   "finding_types": ["duplicate_invoice","payment_misallocation","fx_rate_error","cutoff_error"]
 }}
```

`info` после reset: `{"episode_id": "...", "schema_version": "1.0", "config_version": "..."}`, без данных эталона.

### 8.3 Инструменты

| tool | args | result |
|---|---|---|
| `list_records` | `table ∈ {ledger,invoices,payments,fx}`, `cursor: str\|null`, `limit: int (1..50, по умолчанию 50, >50 обрезается)` | `{"items":[...], "next_cursor": str\|null, "total": int}` |
| `get_record` | `table`, `id` (для ledger — `row_id`, для fx — ключ `YYYY-MM-DD:SRC:DST`, для invoices/payments — `doc_id`) | `{"record": {...}}` |
| `get_document` | `doc_id` | `{"document": {...}, "table": "invoices"\|"payments"}` |
| `calculator` | `expression: str` | `{"value": "123.45"}` (строка Decimal) |
| `submit` | `findings: {...}` (формат §9.1) | `{"accepted": true}` |

- **Курсор:** `base64url(json({"t": table, "o": offset, "e": episode_id[:8]}))`. Чужой или битый курсор → `BAD_CURSOR`. Порядок выдачи — порядок файла, он стабилен.
- **Калькулятор:** разбор через `ast.parse(mode="eval")`. Разрешены только `Expression`, `BinOp(Add|Sub|Mult|Div)`, `UnaryOp(USub|UAdd)` и `Constant(int|float-literal → Decimal(str))`. Длина ≤ 200, глубина ≤ 20, `|value| ≤ 1e18`, точность контекста 28. Всё остальное, деление на 0 и переполнение → `CALC_ERROR`.

### 8.4 Ошибки

`{"ok": false, "result": null, "error": {"code": C, "message": M}, ...}`, шаг расходуется.

| code | Когда | message (фиксированный) |
|---|---|---|
| `UNKNOWN_TOOL` | неизвестный tool | `"unknown tool"` |
| `BAD_ARGS` | не та схема args, неизвестная table | `"invalid arguments"` |
| `UNKNOWN_ID` | id не найден | `"record not found"` |
| `BAD_CURSOR` | битый курсор | `"invalid cursor"` |
| `CALC_ERROR` | калькулятор | `"invalid expression"` |
| `EPISODE_DONE` | step после завершения или до reset | `"episode is not active"` |

### 8.5 Завершение и награда

- `submit` → `terminated=True`. Оценщик получает `(episode_id, findings, inspection_steps)`, где `inspection_steps = steps_used − 1`.
- Если после шага `steps_used == max_steps` и `submit` не было → `truncated=True`, оценщик получает `findings=None` (пустой ответ, F1 по правилам пустого множества).
- Награда по ходу эпизода всегда 0.0. Терминальная награда: в `train` — значение от оценщика, в `eval` — 0.0 (реальная только в логе оценщика).
- `info` в терминальном шаге: в `train` — `{"grade": {tp, fp, fn, precision, recall, f1, reward, valid}}`, в `eval` — `{}`.

### 8.6 Gymnasium-обёртка (опционально, `gym_adapter.py`)

`gymnasium.Env` с `observation_space = action_space = Text(max_length=200_000)`. Действия и наблюдения передаются как JSON-строки. Обёртка должна проходить `gymnasium.utils.env_checker.check_env`.

---

## 9. Оценщик (Роль 3)

### 9.1 findings.json

```json
{"episode_id": "ep_3f9a1c27b6d0",
 "findings": [{"type": "duplicate_invoice", "affected_ids": ["doc:D-k9mq1b"],
   "claimed_amount_minor": 512340000, "currency": "KZT",
   "evidence_ids": ["doc:D-7q2x9a","doc:D-k9mq1b","ledger:R-c3z8aa"],
   "explanation": "Same entity, number and amount re-issued 3 days later."}]}
```

**Схема:** `additionalProperties: false`, `findings` ≤ 30, `evidence_ids` ≤ 10, `affected_ids` 1–10, `explanation` ≤ 1000 символов, `claimed_amount_minor` int ≥ 0, `currency` ∈ известных, `type` ∈ 4 типов, `episode_id` совпадает с текущим. Любое нарушение → `valid=False`, `reward=-1.0`, ошибка пишется в лог оценщика.

### 9.2 Нормализация

`strip()` и Unicode NFC для всех строк. ID — префикс в нижнем регистре, остаток как есть. `affected_ids` и `evidence_ids` превращаются в `frozenset`.

### 9.3 Сопоставление

```
eligible(t, f) = f.type == t.type
             and set(f.affected_ids) == set(t.affected_ids)
             and f.currency == t.currency
             and |f.claimed_amount_minor − t.expected_amount_minor| <= tol(t.type)
             and any(set(E) <= set(f.evidence_ids) for E in t.evidence_sets)
```

- Двудольный граф truth × findings по рёбрам `eligible`. Максимальное паросочетание (алгоритм Куна, ≤ 12 × 30). Обход вершин в порядке `issue_id`, затем индекса finding — это даёт детерминированный tie-break.
- `TP = |M|`, `FP = |F| − TP`, `FN = |T| − TP`. Повторные и несопоставленные выводы считаются FP.
- Так как TP бинарный, награда не зависит от выбора среди максимальных паросочетаний (это нужно отразить в DESIGN.md).

### 9.4 Метрики и награда

```
P  = TP/(TP+FP) if TP+FP>0 else 0
R  = TP/(TP+FN) if TP+FN>0 else 0
F1 = 1                        if |T|=0 and |F|=0
     0                        if |T|=0 and |F|>0
     2TP/(2TP+FP+FN)          otherwise
reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)
invalid structure → reward = -1
```

Расширения формулы допустимы только как отдельное поле `experimental_reward`. Основная формула не меняется.

### 9.5 Изоляция

- `GraderClient` поднимает оценщик в отдельном процессе (`multiprocessing.get_context("spawn")`) и общается по `Pipe`. Передаются только `{"episode_id","findings","inspection_steps"}`.
- Только процесс оценщика открывает `private/`. Процесс env и политики не получают путей к `private/`.
- Docker: `private/` принадлежит UID `grader`, права `0400`, env работает под UID `agent`.
- Лог оценщика: `logs/grader.jsonl` с результатами и ошибками валидации. Агенту он недоступен.

### 9.6 GradeResult

```python
@dataclass(frozen=True)
class GradeResult:
    episode_id: str; valid: bool; error: str | None
    tp: int; fp: int; fn: int
    precision: float; recall: float; f1: float
    inspection_steps: int; reward: float
    matches: tuple[tuple[str, int], ...]   # (issue_id, finding_index)
```

---

## 10. Базовые политики (Роль 4)

Интерфейс:

```python
class Policy(Protocol):
    name: str
    def reset(self, first_obs: dict, *, seed: int) -> None: ...
    def act(self, obs: dict) -> dict: ...
```

| Политика | Поведение | Ожидаемая награда |
|---|---|---|
| `oracle` | Получает truth через `OracleChannel` (только из процесса оценщика при запуске из CLI, не через env). Делает минимальное число шагов: по `get_document` на каждый ID из первого evidence_set, затем `submit` | ≈ 1 − 0.002·k |
| `random_explorer` | Фиксированный seed, случайные `list_records`/`get_record` до шага 60, затем `submit` со случайными выводами из увиденных ID (0–3 штуки) | ≈ −1 |
| `flag_all` | Читает первые страницы, сдаёт 30 выводов всех типов на увиденные документы | ≈ −0.6…−1 |
| `reconciler` | Читает все страницы в пределах бюджета. Ищет дубли по точному номеру и сумме, сравнивает `invoice_refs` с Кт 1200, сверяет курс по `issued_at[:10]` (наивно, без учёта TZ), проверяет период по `timestamp`. Специально наивен | заметно между ними |

Адаптер для внешних политик: `auditgym run --policy pkg.module:Class`.

---

## 11. Траектории

`trajectories.jsonl`, одна строка на шаг:

```json
{"episode_id":"ep_…","policy":"reconciler","step":3,"action":{"tool":"list_records","args":{"table":"ledger","cursor":"…","limit":50}},"observation":{"ok":true,"…":"…"},"reward":0.0,"terminated":false,"truncated":false,"schema_version":"1.0","config_version":"medium-1.0"}
```

- Шаг 0 — запись reset с `action: null`.
- Наблюдения пишутся ровно в том виде, в каком их видела политика. Ни одного поля из `private/`. Это проверяет сканер утечек.
- Итоги по эпизодам: `runs/<run_id>/summary.jsonl` (`episode_id`, `policy`, P, R, F1, reward, steps, wall_time_s).

---

## 12. Независимый валидатор (Роль 4)

`validate/independent_checker.py`. **Не импортирует** `world/`, `books/`, `injectors/`. Пишется с нуля по SPEC.

Для каждого эпизода:
1. Схемы всех файлов и счётчики в `manifest`.
2. Диапазоны: 200–2000 проводок, 20–200 документов, 0–12 ошибок.
3. Инварианты I1–I7 на книге с **откатом** мутаций из `provenance.diff`.
4. Для каждого issue и каждого `evidence_set`: все ID существуют в `open/`, а сумма `expected_amount_minor` пересчитывается **только** из записей этого набора и `policy.json`.
5. В эталоне нет требований к фактам, которых нет в `open/`.

Выход: `reports/validation.json`. Любая ошибка означает, что тест падает.

---

## 13. Преобразования и метаморфные проверки (Роль 4)

`episode/transform.py`:

```python
def transform(src: Path, dst: Path, *, permute_seed: int, id_salt: str) -> dict[str, str]:
    """Перемешивает строки всех таблиц и переименовывает row_id/doc_id биекцией.
       Переписывает все ссылки, truth.json и provenance.json. Возвращает отображение старый→новый ID."""
```

Свойства (тестируются на ≥100 seed):
- **M1:** валидатор проходит на преобразованном эпизоде.
- **M2:** награда oracle на преобразованном эпизоде = награде на исходном.
- **M3:** P/R/F1 политики reconciler совпадают, если она не зависит от порядка строк.
- **M4:** ответ oracle, переведённый по отображению, получает ту же награду.

---

## 14. Тесты

| Набор | Обязательные проверки |
|---|---|
| `tests/common` | `convert` для exp 0/2/3, HALF_EVEN, разбор курса из строки, канонический JSON |
| `tests/generator` | R1 (sha256 при повторе), I1–I7 на 1000 seed (hypothesis), диапазоны размеров, каждый тип включается и выключается независимо, чистые эпизоды содержат отвлекающие случаи, оба сценария взаимодействия встречаются |
| `tests/env` | Схемы obs/info, JSON round-trip, пагинация (≤50, стабильность, конец, битый курсор), каждый код ошибки, калькулятор (≥30 кейсов, включая `**`, `__import__`, длину), шаг до reset и после конца, truncated на 100, изоляция двух reset подряд, R2, режим eval возвращает 0.0 |
| `tests/grader` | Схема (≥10 невалидных вариантов → −1), нормализация, matcher на неоднозначностях, F1 в пустых случаях, формула награды |
| `tests/attacks` | (1) полный ответ > молчание; (2) молчание > ложное обвинение на чистых; (3) повтор TP не даёт выгоды; (4) flag_all < честного частичного; (5) 10 и 90 шагов отличаются на 0.16; (6) угаданные ID дают 0 TP; (7) без полного evidence_set — не TP; (8) неверная валюта и сумма вне допуска — не TP, граница допуска — TP; (9) повторный submit — EPISODE_DONE; (10) тайм-аут — пустой ответ; (11) перестановка findings не меняет награду; (12) hypothesis: монотонность по TP и антимонотонность по FP |
| `tests/leakage` | Рекурсивный скан obs/info/ошибок всех политик на 50 эпизодах: нет `issue_id`, `iss_`, `inject`, `truth`, `private`, `provenance`, seed, путей ФС; сообщения UNKNOWN_ID одинаковы для существующего-удалённого и выдуманного ID |
| `tests/metamorphic` | M1–M4 |
| `tests/validate` | Валидатор на всех `data/train` и `data/dev`, плюс на 100 свежих seed |
| `tests/e2e` | `auditgym all` на маленьком конфиге < 120 с |

CI: `uv run pytest -q` и `docker build` на каждый PR.

---

## 15. CLI, Make, Docker

```
auditgym generate --split train|dev --n 10 --config medium --out data/
auditgym run --policy oracle|random|flag_all|reconciler|pkg.mod:Class --split dev [--mode eval]
auditgym validate --data data/
auditgym transform --src data/dev/ep_00 --dst /tmp/ep_00_t --permute-seed 7 --id-salt x
auditgym report --runs runs/ --out reports/baselines.md
auditgym all        # generate → validate → pytest → 4 политики → report
```

Seed-пространства: train `10_000–10_009`, dev `20_000–20_009`, собственные проверки `90_000+`.

```makefile
all:      ; uv run auditgym all
test:     ; uv run pytest -q
docker:   ; docker build -t auditgym . && docker run --rm --network none --cpus 2 --memory 4g auditgym all
```

```dockerfile
FROM python:3.11-slim@sha256:<закрепить>
ENV PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 UV_NO_CACHE=1
RUN pip install --no-cache-dir uv==<закрепить> && useradd -m agent && useradd -m grader
WORKDIR /app
COPY . .
RUN uv sync --frozen --no-dev || uv sync --frozen
ENTRYPOINT ["uv", "run", "auditgym"]
CMD ["all"]
```

---

## 16. Отчёты и документы (Роль 4)

- `reports/baselines.md`: таблица по политикам × split (P, R, F1, reward mean ± std, steps, time) и разбивка по типам ошибок.
- `reports/failure_analysis.md`: на каких типах и ловушках ошибается reconciler, с измеренными долями и примерами `episode_id`.
- `DESIGN.md` (2–4 стр.): модель мира и инварианты, таблица ошибок с доказательствами, правило декомпозиции сочетаний, matcher и допуски, почему награда инвариантна к tie-break, модель угроз утечек (включая перебор seed и соль), результаты, ограничения (синтетичность, калибровка штрафа за шаги на больших эпизодах, неполнота типов ошибок).
- `README.md`: одна команда, структура, как подключить свою политику, таблица «атака → награда».

---

## 17. Владельцы и порядок работ

| Роль | Владеет | Отдаёт другим в первый час |
|---|---|---|
| 1 Генератор | `world/ books/ injectors/ episode/serialize.py configs/` | Эпизод-заглушка на диске по §4 |
| 2 Среда | `env/ logging/ cli.py Dockerfile Makefile` | `AuditEnv` с `list_records` и `submit` |
| 3 Оценщик | `grader/ schemas/ tests/{grader,attacks,leakage}` | `GraderClient` с интерфейсом §9.6 (можно заглушку) |
| 4 Интегратор | `validate/ policies/ episode/transform.py tests/{metamorphic,validate,e2e} reports/ README DESIGN` | Политика oracle, `run_episode` |
| Все | `common/ SPEC.md` | — |

**Вехи:**
- **M0:** контракт заморожен, заглушки.
- **M1:** E2E на дубле, oracle ≈ 1.
- **M2:** все 4 типа ошибок, атаки зелёные.
- **M3:** сочетания, ловушки, метаморфика, утечки, Docker.
- **M4:** 10+10 эпизодов, отчёты, документы, демо.
- **M5:** запуск с нуля на чужой машине, сдача.

**Определение готовности (DoD) для любого PR:** тесты на новое поведение, `uv run pytest -q` зелёный, нет нарушений §1, правки только в своих папках или с апрувом владельца.
