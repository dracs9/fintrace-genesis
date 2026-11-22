"""Baseline policies exercise the real audited tools without sharing a key with the env."""

import json
from pathlib import Path

import pytest

from auditgym.env.core import AuditEnv
from auditgym.grader.client import GraderClient
from auditgym.grader.schema import parse_findings
from auditgym.policies import Policy, load_policy, run_episode
from auditgym.policies.base import Record, submit
from auditgym.policies.oracle import OracleChannel, OraclePolicy
from auditgym.policies.reconciler import Reconciler

TINY = Path(__file__).resolve().parents[1] / 'env' / 'fixtures' / 'ep_tiny'
CLEAN = Path(__file__).resolve().parents[1] / 'generator' / 'fixtures' / 'step0'


class ExternalPolicy:
    name = 'external_empty'

    def reset(self, first_obs: Record, *, seed: int) -> None:
        self.episode = first_obs['result']['manifest']['episode_id']

    def act(self, obs: Record) -> Record:
        return submit(self.episode, [])


def _trace(env: AuditEnv, policy: Policy, seed: int = 0) -> tuple[list[Record], list[Record]]:
    obs, _ = env.reset_from_dir(TINY)
    policy.reset(obs, seed=seed)
    actions: list[Record] = []
    observations = [obs]
    while True:
        action = policy.act(obs)
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        assert obs['ok'] and not truncated
        if terminated:
            return actions, observations


def test_oracle_reads_each_first_evidence_once(tmp_path: Path) -> None:
    with OracleChannel(TINY) as channel, GraderClient(TINY, log_dir=tmp_path) as grader:
        env = AuditEnv(tmp_path, grader=grader)
        actions, _ = _trace(env, OraclePolicy(channel))
        assert actions[:-1] == [
            {'tool': 'get_document', 'args': {'doc_id': 'D-7q2x9a'}},
            {'tool': 'get_document', 'args': {'doc_id': 'D-k9mq1b'}},
            {'tool': 'get_record', 'args': {'id': 'R-c3z8aa', 'table': 'ledger'}},
        ]
        run = run_episode(env, load_policy('oracle', channel=channel), env.reset_from_dir(TINY)[0])
        assert run.terminated and run.steps == 4 and run.reward == pytest.approx(0.994)
        assert run.info['grade']['valid'] and run.info['grade']['tp'] == 1
        assert run.findings == actions[-1]['args']['findings']
        with pytest.raises(ValueError, match='^invalid arguments$'):
            channel.findings('not-an-episode')
        with pytest.raises(ValueError, match='^invalid arguments$'):
            channel.findings('ep_000000000000')


def test_oracle_requires_injection_and_submits_only_complete_proofs() -> None:
    with OracleChannel(CLEAN) as channel:
        env = AuditEnv(CLEAN)
        first, _ = env.reset_from_dir(CLEAN)
        with pytest.raises(ValueError, match='^invalid arguments$'):
            OraclePolicy().reset(first, seed=0)
        run = run_episode(env, OraclePolicy(channel), first)
        assert run.steps == 1 and run.findings == {'episode_id': env.episode_id, 'findings': []}
    with OracleChannel(TINY) as channel:
        env = AuditEnv(TINY, max_steps=2)
        run = run_episode(env, OraclePolicy(channel), env.reset_from_dir(TINY)[0])
        assert run.steps == 2 and run.terminated and run.findings is not None
        assert run.findings['findings'] == []


def test_random_reads_only_seen_ids_and_replays() -> None:
    env = AuditEnv(TINY)
    policy = load_policy('random_explorer')
    actions, obs = _trace(env, policy, seed=378)
    assert _trace(env, policy, seed=378) == (actions, obs)
    assert _trace(env, policy, seed=2)[0] != actions
    assert len(actions) == 61
    seen: set[tuple[str, str]] = set()
    evidence: set[str] = set()
    tools = {a['tool'] for a in actions}
    assert tools == {'list_records', 'get_record', 'submit'}
    for action, after in zip(actions[:-1], obs[1:-1], strict=True):
        table = action['args']['table']
        if action['tool'] == 'list_records':
            for record in after['result']['items']:
                if table == 'fx':
                    name = f"{record['date']}:{record['source_currency']}:{record['target_currency']}"
                    prefix = 'fx'
                elif table == 'ledger':
                    name, prefix = record['row_id'], 'ledger'
                else:
                    name, prefix = record['doc_id'], 'doc'
                seen.add((table, name))
                evidence.add(f'{prefix}:{name}')
        else:
            assert (table, action['args']['id']) in seen
    payload = actions[-1]['args']['findings']
    assert len(parse_findings(payload, payload['episode_id'])) <= 3
    for row in payload['findings']:
        assert evidence.issuperset(row['affected_ids'] + row['evidence_ids'])


def test_flag_all_uses_first_pages_and_reserves_submit() -> None:
    env = AuditEnv(TINY)
    actions, obs = _trace(env, load_policy('flag_all'))
    assert [a['args']['table'] for a in actions[:-1]] == ['invoices', 'payments']
    payload = actions[-1]['args']['findings']
    rows = parse_findings(payload, payload['episode_id'])
    docs = {'doc:' + r['doc_id'] for o in obs[1:-1] for r in o['result']['items']}
    assert len(rows) == 16 and {r.type.value for r in rows} == set(obs[0]['result']['finding_types'])
    assert all(docs.issuperset(r.affected_ids | r.evidence_ids) for r in rows)
    for name in ('random', 'reconciler', 'flag_all'):
        limited = AuditEnv(TINY, max_steps=1)
        first, _ = limited.reset_from_dir(TINY)
        run = run_episode(limited, load_policy(name), first)
        assert run.terminated and run.steps == 1


def test_external_adapter_uses_zero_argument_class() -> None:
    policy = load_policy(f'{__name__}:ExternalPolicy')
    env = AuditEnv(CLEAN)
    run = run_episode(env, policy, env.reset_from_dir(CLEAN)[0])
    assert run.steps == 1 and run.terminated
    with pytest.raises(ValueError, match='^invalid arguments$'):
        load_policy('builtins:dict')


def test_reconciler_exact_number_fx_cutoff_and_misallocation() -> None:
    first, _ = AuditEnv(TINY).reset_from_dir(TINY)
    policy = Reconciler()
    policy.reset(first, seed=0)
    def stamp(day: str) -> str:
        return day + 'T11:00:00+05:00'
    def sale(doc: str, number: str, day: str) -> Record:
        return {'doc_id': doc, 'entity_id': 'E-a', 'invoice_id': number, 'currency': 'USD',
                'total_minor': 100, 'kind': 'invoice', 'issued_at': stamp(day)}
    def line(i: int, account: str, day: str, debit: int, credit: int, inv: str | None,
             pay: str | None = None) -> Record:
        return {'row_id': f'R-{i:06}', 'timestamp': stamp(day), 'account': account,
                'debit_minor': debit, 'credit_minor': credit, 'invoice_id': inv, 'payment_id': pay, 'entity_id': 'E-a'}
    policy.rows['invoices'] = [sale('D-000001', 'A', '2026-03-01'), sale('D-000002', 'A', '2026-03-02'),
                              sale('D-000003', 'B', '2026-03-03')]
    policy.rows['payments'] = [{'doc_id': 'D-000004', 'payment_id': 'P-x', 'timestamp': stamp('2026-04-01'),
                                'currency': 'USD', 'amount_minor': 80, 'entity_id': 'E-a',
                                'invoice_refs': [{'invoice_id': 'B', 'amount_minor': 80}]}]
    policy.rows['fx'] = [{'date': '2026-02-27', 'source_currency': 'USD', 'target_currency': 'KZT', 'rate': '2'}]
    policy.rows['ledger'] = [line(1, '1200', '2026-03-01', 200, 0, 'A'), line(2, '1200', '2026-03-02', 200, 0, 'A'),
                             line(3, '1200', '2026-03-03', 300, 0, 'B'), line(4, '1010', '2026-03-31', 160, 0, None, 'P-x'),
                             line(5, '1200', '2026-03-31', 0, 50, 'B', 'P-x'), line(6, '2150', '2026-03-31', 0, 110, None, 'P-x')]
    findings = policy.findings()
    assert {r['type']: r['claimed_amount_minor'] for r in findings} == {
        'duplicate_invoice': 200, 'fx_rate_error': 100, 'payment_misallocation': 110, 'cutoff_error': 160}
    assert next(r for r in findings if r['type'] == 'payment_misallocation')['evidence_ids'] == [
        'doc:D-000003', 'doc:D-000004', 'ledger:R-000006']
    for table in sorted(policy.rows):
        policy.rows[table].reverse()
    assert policy.findings() == findings
    assert 'issue_id' not in json.dumps(findings)


def test_random_stream_varies_by_public_episode_and_repeats() -> None:
    env = AuditEnv(TINY)
    policy = load_policy('random')

    def sample() -> list[int]:
        sizes = []
        for n in range(32):
            first, _ = env.reset_from_dir(TINY)
            first['result']['manifest']['episode_id'] = f'ep_{n:012x}'
            policy.reset(first, seed=0)
            obs = first
            while True:
                action = policy.act(obs)
                if action['tool'] == 'submit':
                    sizes.append(len(action['args']['findings']['findings']))
                    break
                obs, _, _, _, _ = env.step(action)
        return sizes

    sizes = sample()
    assert set(sizes) == {0, 1, 2, 3} and sizes == sample()


def test_same_time_exact_numbers_have_no_unique_later_invoice() -> None:
    env = AuditEnv(TINY)
    policy = Reconciler()
    _trace(env, policy)
    original, later = sorted((r for r in policy.rows['invoices'] if r['currency'] == 'KZT'),
                             key=lambda r: r['issued_at'])
    duplicate_number = later['invoice_id']
    for row in policy.rows['ledger']:
        if row['invoice_id'] == duplicate_number:
            row['invoice_id'] = original['invoice_id']
    later['invoice_id'] = original['invoice_id']
    assert any(item['type'] == 'duplicate_invoice' for item in policy.findings())
    twin = dict(original, doc_id='D-000001')
    policy.rows['invoices'].append(twin)
    for _ in range(2):
        assert all(item['type'] != 'duplicate_invoice' for item in policy.findings())
        policy.rows['invoices'].reverse()


def test_oracle_reads_a_shared_fx_proof_once() -> None:
    class SharedProof:
        def findings(self, episode_id: str) -> list[Record]:
            return [dict(type=kind, affected_ids=['doc:D-u5s0b3'], currency='KZT',
                         claimed_amount_minor=1, explanation='Inspect the same posting.',
                         evidence_ids=['doc:D-u5s0b3', 'ledger:R-m2v7c1', 'fx:2026-03-04:USD:KZT'])
                    for kind in ('fx_rate_error', 'cutoff_error')]

    actions, _ = _trace(AuditEnv(TINY), OraclePolicy(SharedProof()))
    assert [action['tool'] for action in actions] == ['get_document', 'get_record', 'get_record', 'submit']
    assert actions[1]['args'] == {'table': 'fx', 'id': '2026-03-04:USD:KZT'}
    assert len(actions[-1]['args']['findings']['findings']) == 2
