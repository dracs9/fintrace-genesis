import { $, amount, button, el, pretty } from "./ui.js";

const empty = (type, currency) => ({
  type,
  affected_ids: [],
  claimed_amount_minor: "",
  currency,
  evidence_ids: [],
  explanation: "",
});

export function findingError(draft, count, editing) {
  if (count >= 30 && !editing) return "The submission already contains 30 findings.";
  if (!draft.affected_ids.length) return "Select an affected record in the explorer.";
  if (draft.affected_ids.length > 10 || draft.evidence_ids.length > 10)
    return "Each ID field supports up to 10 records.";
  if (!/^\d+$/.test(draft.claimed_amount_minor)) return "Enter a nonnegative integer amount.";
  if (Array.from(draft.explanation).length > 1000) return "Shorten the explanation to 1,000 characters.";
  return "";
}

function select(values, current, change) {
  const node = el("select");
  for (const value of values) node.add(new Option(value, value));
  node.value = current;
  node.onchange = () => change(node.value);
  return node;
}

function field(title, control) {
  const label = el("label", title);
  label.append(control);
  return label;
}

export class Findings {
  constructor(submit) {
    this.submit = submit;
  }

  reset(first) {
    this.manifest = first.manifest;
    this.types = first.finding_types;
    this.rows = [];
    this.clear();
  }

  clear() {
    this.editing = -1;
    this.dirty = false;
    this.draft = empty(this.types[0], this.manifest.base_currency);
  }

  attach(key, id) {
    const list = this.draft[key];
    if (!list.includes(id)) {
      if (list.length >= 10) throw Error("A finding supports up to 10 IDs per field.");
      list.push(id);
      this.dirty = true;
    }
    this.render();
  }

  render(readOnly = false, submitted = null) {
    const root = $("builder");
    root.replaceChildren();
    const rows = submitted ?? this.rows;
    $("finding-count").textContent = rows.length + " / 30";
    if (readOnly) {
      root.append(
        el("div", submitted ? "Submitted findings" : "Read-only trajectory", "form-head"),
      );
      if (!rows.length) root.append(el("p", "No findings submitted.", "hint"));
      rows.forEach((row, i) => {
        const details = el("details");
        details.append(el("summary", i + 1 + ". " + row.type), pretty(row));
        root.append(details);
      });
      return;
    }
    if (rows.length) {
      const list = el("div", null, "finding-list");
      rows.forEach((row, i) => {
        const entry = el("div", null, "finding-entry");
        entry.append(
          button(
            i + 1 + ". " + row.type,
            () => {
              this.editing = i;
              this.draft = structuredClone(row);
              this.draft.claimed_amount_minor = String(row.claimed_amount_minor);
              this.dirty = true;
              this.render();
            },
            "text-btn",
          ),
          button("Remove", () => {
            this.rows.splice(i, 1);
            this.clear();
            this.render();
          }),
        );
        list.append(entry);
      });
      root.append(list);
    }
    const d = this.draft;
    root.append(
      el(
        "div",
        this.editing >= 0 ? "Edit finding " + (this.editing + 1) : "New finding",
        "form-head",
      ),
    );
    const form = el("div", null, "finding-form");
    const save = button(
      this.editing < 0 ? "Add finding" : "Save changes",
      () => {
        const row = { ...structuredClone(d), claimed_amount_minor: BigInt(d.claimed_amount_minor) };
        if (this.editing >= 0) this.rows[this.editing] = row;
        else this.rows.push(row);
        this.clear();
        this.render();
      },
      "full",
    );
    const send = button("Submit findings", () => this.submit(this.rows), "primary full");
    const discard = button(
      "Discard draft",
      () => {
        this.clear();
        this.render();
      },
      "text-btn",
    );
    const feedback = el("div", null, "hint");
    const update = () => {
      const error = findingError(d, rows.length, this.editing >= 0);
      save.disabled = Boolean(error);
      send.disabled = this.dirty;
      discard.hidden = !this.dirty;
      feedback.textContent = this.dirty ? error : "";
    };
    const change = (name, value) => {
      d[name] = value;
      this.dirty = true;
      update();
    };
    form.append(
      field(
        "Finding type",
        select(this.types, d.type, (v) => change("type", v)),
      ),
    );
    for (const [key, label] of [
      ["affected_ids", "Affected IDs"],
      ["evidence_ids", "Evidence IDs"],
    ]) {
      const header = el("div", null, "label-line");
      header.append(el("span", label), el("span", d[key].length + " / 10", "mono"));
      const chips = el("div", null, "id-field");
      if (!d[key].length) chips.textContent = "Select from open books";
      for (const id of d[key]) {
        const chip = button(
          id + " ×",
          () => {
            d[key] = d[key].filter((x) => x !== id);
            this.dirty = true;
            this.render();
          },
          "chip",
        );
        chip.setAttribute("aria-label", "Remove " + id + " from " + label);
        chips.append(chip);
      }
      const group = el("div", null, "stack tight");
      group.append(header, chips);
      form.append(group);
    }
    const integer = el("input");
    integer.inputMode = "numeric";
    integer.placeholder = "0";
    integer.value = d.claimed_amount_minor;
    integer.className = "mono";
    const converted = el("small", null, "mono muted");
    const showAmount = () => {
      converted.textContent = /^\d+$/.test(integer.value)
        ? amount(integer.value, d.currency, this.manifest.currency_exponents)
        : "";
    };
    integer.oninput = () => {
      change("claimed_amount_minor", integer.value);
      showAmount();
    };
    const pair = el("div", null, "pair");
    const currency = select(
      Object.keys(this.manifest.currency_exponents).sort(),
      d.currency,
      (v) => {
        change("currency", v);
        showAmount();
      },
    );
    pair.append(field("Claimed minor units", integer), field("Currency", currency));
    form.append(pair, converted);
    showAmount();
    const note = el("textarea");
    note.rows = 2;
    note.value = d.explanation;
    note.placeholder = "Explain the discrepancy…";
    const count = el("small", null, "muted mono");
    const countChars = () => {
      count.textContent = Array.from(note.value).length + " / 1000";
    };
    note.oninput = () => {
      change("explanation", note.value);
      countChars();
    };
    countChars();
    const noteField = field("Explanation", note);
    noteField.append(count);
    form.append(noteField, feedback, save, discard);
    root.append(form, el("hr", null, "rule"), send);
    update();
  }
}

export function verdict(grade, envReward) {
  const root = $("verdict");
  root.replaceChildren();
  if (!grade) {
    root.append(
      el("span", "Awaiting terminal step", "pending"),
      el("p", "Submit findings or exhaust the step budget to view the score.", "hint"),
    );
    return;
  }
  root.append(
    el(
      "div",
      grade.valid ? "Valid submission" : "Invalid submission",
      grade.valid ? "status" : "error",
    ),
  );
  const metrics = el("div", null, "measure-grid");
  for (const [label, key] of [
    ["TP", "tp"],
    ["FP", "fp"],
    ["FN", "fn"],
    ["Precision", "precision"],
    ["Recall", "recall"],
    ["F1", "f1"],
  ]) {
    const item = el("div", label);
    item.append(el("b", key.length > 2 || key === "f1" ? grade[key].toFixed(3) : grade[key]));
    metrics.append(item);
  }
  root.append(metrics, el("div", "Reward " + String(grade.reward), "reward"));
  root.append(
    el(
      "p",
      grade.valid
        ? "clip(2 × " + grade.f1 + " − 1 − 0.002 × " + grade.inspection_steps + ", −1, 1)"
        : "Invalid structure · reward fixed at −1",
      "formula",
    ),
  );
  root.append(
    el(
      "p",
      "Environment reward " + envReward + " · " + grade.inspection_steps + " inspection steps",
      "hint",
    ),
  );
  root.append(el("hr", null, "rule"), el("div", "Matches", "form-head"));
  if (!grade.matches.length) root.append(el("p", "No matched issues.", "hint"));
  else {
    const table = el("table"),
      head = table.createTHead().insertRow();
    head.append(el("th", "Issue ID"), el("th", "Finding index"));
    const body = table.createTBody();
    grade.matches.forEach(([issue, finding]) => {
      const row = body.insertRow();
      row.insertCell().textContent = issue;
      row.insertCell().textContent = finding;
    });
    root.append(table);
  }
}
