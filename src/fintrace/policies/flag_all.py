"""Indiscriminate accusations illustrate the precision/evidence penalty."""

from ._pages import PagePolicy
from .base import Record, finding


class FlagAll(PagePolicy):
    name = 'flag_all'
    tables = ('invoices', 'payments')
    first_pages_only = True

    def findings(self) -> list[Record]:
        """Cycle through the four issue types on the first returned documents."""
        documents = self.rows['invoices'] + self.rows['payments']
        if not documents:
            return []
        findings = []
        for i in range(min(30, len(documents) * len(self.types))):
            doc = documents[i % len(documents)]
            handle = 'doc:' + doc['doc_id']
            kind = self.types[(i % len(documents) + i // len(documents)) % len(self.types)]
            findings.append(finding(kind, [handle], 0, self.manifest['base_currency'], [handle],
                                    'Flagged for review without reconciliation.'))
        return findings
