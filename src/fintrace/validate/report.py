"""Trusted offline measurement and reporting for FinTrace (§11, §16)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from fintrace.grader.normalize import Finding, IssueType
from fintrace.grader.schema import parse_findings
from fintrace.grader.service import grade
from fintrace.logging.trajectory import EpisodeSummary, read_jsonl

SAMPLES = (("train", 10_000, 10), ("dev", 20_000, 10), ("fresh", 90_000, 20))
POLICIES = ("oracle", "reconciler", "flag_all", "random")
INTERACTIONS = ("fx_partial_boundary", "duplicate_with_payment")
DISTRACTORS = ("recurring_subscription", "split_payment", "multi_invoice_payment", "partial_open",
               "overpayment", "bank_fee", "credit_note", "boundary_correct", "exotic_exponent")


@dataclass(frozen=True)
class Case:
    kind: str
    ids: frozenset[str]


@dataclass(frozen=True)
class Measurement:
    split: str
    config_version: str
    generator_version: str
    summary: EpisodeSummary
    issues: dict[str, str]
    findings: tuple[Finding, ...]
    matches: tuple[tuple[str, int], ...]
    mutations: tuple[Case, ...]
    distractors: tuple[Case, ...]
    valid: bool


def _terminals(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["terminated"] or row["truncated"]:
                yield row


def _measurements(runs: Path, data: Path) -> list[Measurement]:
    locations: dict[str, tuple[Path, str, str]] = {}
    for path in sorted(data.glob("*/*/open/manifest.json")):
        manifest = json.loads(path.read_bytes())
        episode_id = manifest["episode_id"]
        if episode_id in locations:
            raise ValueError("duplicate dataset episode")
        locations[episode_id] = (path.parent.parent, manifest["config_version"], manifest["generator_version"])
    result = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(runs.glob("*/summary.jsonl")):
        ends = _terminals(path.with_name("trajectories.jsonl"))
        for raw, end in zip(read_jsonl(path), ends, strict=True):
            summary = EpisodeSummary(**raw)
            if summary.reward is None:
                raise ValueError("report requires scored train summaries")
            identifier = summary.episode_id
            key = (summary.policy, identifier)
            if key in seen:
                raise ValueError("duplicate policy/episode measurement; select one run per policy")
            seen.add(key)
            if identifier not in locations:
                raise ValueError("episode for report missing from dataset")
            if (identifier != end["episode_id"] or summary.policy != end["policy"]
                    or summary.steps != end["observation"]["steps_used"]):
                raise ValueError("trajectory/summary mismatch")
            episode, config_version, generator_version = locations[identifier]
            payload = end["action"]["args"]["findings"] if end["terminated"] else None
            scored = grade(episode, payload, summary.steps - int(end["terminated"]))
            actual = (scored.precision, scored.recall, scored.f1, scored.reward)
            if actual != (summary.precision, summary.recall, summary.f1, summary.reward):
                raise ValueError("summary does not agree with independent grader replay")
            truth = json.loads((episode / "private/truth.json").read_bytes())["issues"]
            provenance = json.loads((episode / "private/provenance.json").read_bytes())
            findings = parse_findings(payload, identifier) if scored.valid and payload is not None else ()
            result.append(Measurement(
                episode.parent.name, config_version, generator_version, summary,
                {t["issue_id"]: t["type"] for t in truth},
                findings, scored.matches,
                tuple(Case(m["injector"], frozenset(m["issue_ids"])) for m in provenance["mutations"]),
                tuple(Case(d["kind"], frozenset(d["ids"])) for d in provenance["distractors"]),
                scored.valid,
            ))
    if not result:
        raise ValueError("no completed measurements")
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _counts(rows: Sequence[Measurement], kind: str) -> tuple[int, int, int]:
    truth = sum(sum(row.issues[k] == kind for k in sorted(row.issues)) for row in rows)
    reported = sum(sum(f.type == kind for f in row.findings) for row in rows)
    tp = sum(sum(row.issues[k] == kind for k, _ in row.matches) for row in rows)
    return tp, reported - tp, truth - tp


def _splits(rows: Sequence[Measurement]) -> list[str]:
    present = {r.split for r in rows}
    preferred = [s for s, _, _ in SAMPLES if s in present]
    return preferred + sorted(present.difference(preferred))


def _baseline(rows: Sequence[Measurement]) -> str:
    configs = ", ".join(sorted({r.config_version for r in rows}))
    versions = ", ".join(sorted({r.generator_version for r in rows}))
    lines = ["# Базовые политики", "", f"Конфигурация — {configs}. Версия генератора — {versions}.", "",
             "Каждая строка усредняет эпизоды с одинаковым весом. Разброс награды — "
             "стандартное отклонение по всей указанной выборке. Шаги включают submit. "
             "Время измерено от загрузки открытых таблиц до терминальной оценки, включая "
             "запись траектории и обмен с оценщиком. Генерация и запуск процессов исключены.", "",
             "| Выборка | Политика | N | P | R | F1 | Награда ± σ | Шаги | Время, с |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    details = ["## По типам ошибок", "",
               "Счётчики объединены внутри выборки. P, R и F1 рассчитаны из этих счётчиков. "
               "При отсутствии примеров типа в обеих колонках метрика помечена прочерком.", "",
               "| Выборка | Политика | Тип | TP | FP | FN | P | R | F1 |",
               "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for split in _splits(rows):
        group = [r for r in rows if r.split == split]
        for name in sorted({r.summary.policy for r in group}):
            cohort = [r for r in group if r.summary.policy == name]
            means = [fmean(getattr(r.summary, field) for r in cohort)
                     for field in ("precision", "recall", "f1", "reward", "steps", "wall_time_s")]
            sd = pstdev(r.summary.reward for r in cohort if r.summary.reward is not None)
            p, recall, f1, reward, steps, wall = means
            lines.append(f"| {split} | {name} | {len(cohort)} | {p:.3f} | {recall:.3f} | {f1:.3f} "
                         f"| {reward:.3f} ± {sd:.3f} | {steps:.1f} | {wall:.3f} |")
            for kind in IssueType:
                tp, fp, fn = _counts(cohort, kind.value)
                values = (f"{_ratio(tp, tp+fp):.3f} | {_ratio(tp, tp+fn):.3f} | "
                          f"{_ratio(2*tp, 2*tp+fp+fn):.3f}") if tp + fp + fn else "— | — | —"
                details.append(f"| {split} | {name} | {kind} | {tp} | {fp} | {fn} | {values} |")
    lines += ["", "P и R пустого правильного ответа равны 0, F1 равен 1 по §9.4. "
              "Поэтому средние P и R даже у oracle могут быть меньше единицы.", "",
              "## Порядок наград", ""]
    for split in _splits(rows):
        group = [r for r in rows if r.split == split]
        averages = {name: fmean(r.summary.reward for r in group if r.summary.policy == name)
                    for name in sorted({r.summary.policy for r in group})}
        order = sorted(averages, key=lambda name: (-averages[name], name))
        lines.append(f"- {split} — " + ", ".join(f"{name} {averages[name]:.3f}" for name in order) + ".")
    random = [r for r in rows if r.summary.policy == "random_explorer"]
    if random:
        silent_clean = sum(not r.issues and not r.findings and r.valid for r in random)
        lines += ["", f"У random_explorer пустая сдача на чистой книге встретилась "
                  f"{silent_clean}/{len(random)} раз. После 60 инспекций такой эпизод приносит 0.88. "
                  "Эта ветвь §9.4 может поднять random выше flag_all, даже если ни одна "
                  "случайная находка не засчитана."]
    lines += ["", f"Невалидных пакетов находок — {sum(not r.valid for r in rows)}/{len(rows)}. "
              "Результат каждого эпизода повторно рассчитан по терминальному действию.",
              "", *details, ""]
    return "\n".join(lines)


def _example(ids: Sequence[str]) -> str:
    return ", ".join(f"`{key}`" for key in sorted(set(ids))[:2]) or "—"


def _failure(rows: Sequence[Measurement]) -> str:
    cohort = [r for r in rows if r.summary.policy == "reconciler"]
    lines = ["# Где ошибается reconciler", "",
             f"Выборка содержит {len(cohort)} эпизодов. Каждая пропущенная ошибка — FN, "
             "каждая несопоставленная находка — FP. Знаменатель для доли пропусков — "
             "число ошибок в эталоне, для доли ложных находок — число поданных находок этого типа.", "",
             "| Тип | TP | FN / ошибок | FP / находок | Пример FN | Пример FP |",
             "|---|---:|---|---|---|---|"]
    for kind in IssueType:
        tp, fp, fn = _counts(cohort, kind.value)
        misses, false = [], []
        for r in cohort:
            t, f = {k for k, _ in r.matches}, {i for _, i in r.matches}
            if any(r.issues[k] == kind and k not in t for k in sorted(r.issues)):
                misses.append(r.summary.episode_id)
            if any(row.type == kind and i not in f for i, row in enumerate(r.findings)):
                false.append(r.summary.episode_id)
        lines.append(f"| {kind} | {tp} | {fn}/{tp+fn} | {fp}/{tp+fp} | "
                     f"{_example(misses)} | {_example(false)} |")
    lines += ["", "## Сочетания", "", "Единица подсчёта — одна парная мутация. Полное обнаружение "
              "требует двух совпадений в паросочетании.", "",
              "| Сценарий | Случаев | Оба | Один | Ни одного | Неполных / всего | Пример неполного |",
              "|---|---:|---:|---:|---:|---|---|"]
    for kind in INTERACTIONS:
        found: Counter[int] = Counter()
        missed = []
        for r in cohort:
            matched = {k for k, _ in r.matches}
            for m in r.mutations:
                if m.kind == kind:
                    hits = len(m.ids & matched)
                    found[hits] += 1
                    if hits < 2:
                        missed.append(r.summary.episode_id)
        total = found.total()
        lines.append(f"| {kind} | {total} | {found[2]} | {found[1]} | {found[0]} "
                     f"| {total-found[2]}/{total} | {_example(missed)} |")
    lines += ["", "## Корректные отвлекающие случаи", "",
              "Числитель — число размеченных случаев, где хотя бы один affected_id ложной находки "
              "совпал с документом случая. Один FP может затронуть несколько случаев. "
              "Пересечение показывает место ложного обвинения, причинность из него не следует.", "",
              "| Ловушка | Затронуто / размечено | Доля | Пример FP | Пример случая |",
              "|---|---:|---:|---|---|"]
    for kind in DISTRACTORS:
        touched, cases = [], []
        for r in cohort:
            matched = {i for _, i in r.matches}
            wrong = {key for i, f in enumerate(r.findings) if i not in matched for key in sorted(f.affected_ids)}
            for d in r.distractors:
                if d.kind == kind:
                    cases.append(r.summary.episode_id)
                    if wrong & d.ids:
                        touched.append(r.summary.episode_id)
        rate = f"{len(touched)/len(cases):.1%}" if cases else "—"
        lines.append(f"| {kind} | {len(touched)}/{len(cases)} | {rate} | "
                     f"{_example(touched)} | {_example(cases)} |")
    lines += ["", "## Объяснение ошибок", "",
              "Reconciler сравнивает номера счетов буквально. Генератор меняет написание "
              "дубликата, поэтому одинаковый экономический счёт может получить два разных ключа. "
              "В duplicate_with_payment распределение иногда удаётся доказать независимо от дубля.", "",
              "Для FX и границы периода политика берёт дату из текста документа. В "
              "fx_partial_boundary перевод в +05:00 меняет день. Такой платёж может выглядеть "
              "согласованным одновременно по неверному курсу и неверному месяцу. Тот же выбор "
              "даты может создать лишнюю находку у корректно проведённого документа.", "",
              "При обработке платежей политика сопоставляет invoice_refs с Кт 1200 и ищет "
              "строку, объясняющую недостачу. Ошибка даты курса исходного счёта меняет "
              "и ожидаемую сумму распределения. Отсутствие FP на одной из ловушек "
              "описывает только наблюдавшуюся выборку.", ""]
    return "\n".join(lines)


def build_report(runs: Path, out: Path, data: Path | None = None) -> None:
    """Replay recorded terminal answers as an evaluator and write both Markdown reports."""
    rows = _measurements(runs, data if data is not None else runs.parent / "data")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_baseline(rows), encoding="utf-8")
    if any(r.summary.policy == "reconciler" for r in rows):
        out.with_name("failure_analysis.md").write_text(_failure(rows), encoding="utf-8")


def main() -> None:
    """Reproduce the medium 10+10+20 evaluation with one fresh policy process per split."""
    from fintrace.cli import run_cached_episodes
    from fintrace.env.core import ensure_episode, load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--runs", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--out", type=Path, default=Path("reports/baselines.md"))
    parser.add_argument("--measure", nargs=2, metavar=("SPLIT", "POLICY"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config("medium")
    for split, start, count in SAMPLES:
        if args.measure and args.measure[0] != split:
            continue
        root = args.data / split
        paths = [ensure_episode(root, seed, config) for seed in range(start, start + count)]
        if args.measure:
            run_cached_episodes(root, paths, args.measure[1], args.runs, split=split)
            return
        for policy in POLICIES:
            subprocess.run([sys.executable, "-m", "fintrace.validate.report", "--data", str(args.data),
                            "--runs", str(args.runs), "--measure", split, policy], check=True)
    build_report(args.runs, args.out, args.data)


if __name__ == "__main__":
    main()
