import assert from "node:assert/strict";
import test from "node:test";
import { amount, wire } from "../web/ui.js";
import { findingError } from "../web/findings.js";

test("display minor units using the manifest exponent, including large integers", () => {
  const exponents = { KZT: 2, USD: 2, EUR: 2, JPY: 0, KWD: 3 };
  assert.equal(amount(5, "KWD", exponents), "0.005 KWD");
  assert.equal(amount(12345, "JPY", exponents), "12,345 JPY");
  assert.equal(amount(-5, "EUR", exponents), "−0.05 EUR");
  assert.equal(amount(900719925474099301n, "USD", exponents), "9,007,199,254,740,993.01 USD");
});

test("wire and raw JSON preserve exact integer claims and escape user text", () => {
  const value = { amount: 900719925474099301n, ids: [], note: 'Text "quoted"\nline' };
  assert.equal(
    wire(value),
    '{"amount":900719925474099301,"ids":[],"note":"Text \\"quoted\\"\\nline"}',
  );
  assert.match(wire(value, "  "), /"amount": 900719925474099301,/);
});

test("builder limits match the grader at each boundary", () => {
  const draft = {
    affected_ids: ["doc:D-000001"],
    evidence_ids: [],
    claimed_amount_minor: "0",
    explanation: "🧾".repeat(1000),
  };
  assert.equal(findingError(draft, 29, false), "");
  assert.equal(findingError(draft, 30, true), "");
  assert.ok(findingError(draft, 30, false));
  for (const changes of [
    { affected_ids: [] },
    { evidence_ids: Array(11) },
    { affected_ids: Array(11) },
    { explanation: draft.explanation + "x" },
    { claimed_amount_minor: "1.1" },
    { claimed_amount_minor: "-1" },
  ]) {
    assert.ok(findingError({ ...draft, ...changes }, 0, false));
  }
});
