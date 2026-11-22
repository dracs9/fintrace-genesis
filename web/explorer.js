import { $, amount, bind, button, el, modal, pretty, tell } from "./ui.js";

const tables = {
  ledger: "Ledger",
  invoices: "Invoices",
  payments: "Payments",
  fx: "FX rates",
  meta: "Policy & manifest",
};
const columns = {
  ledger: [
    "row_id",
    "account",
    "debit_minor",
    "credit_minor",
    "invoice_id",
    "payment_id",
    "timestamp",
    "entity_id",
  ],
  invoices: [
    "doc_id",
    "invoice_id",
    "total_minor",
    "kind",
    "issued_at",
    "due_at",
    "entity_id",
    "original_invoice_id",
  ],
  payments: ["doc_id", "payment_id", "amount_minor", "fee_minor", "timestamp", "entity_id"],
  fx: ["date", "source_currency", "target_currency", "rate", "key"],
};
const names = {
  row_id: "Row ID",
  doc_id: "Document ID",
  invoice_id: "Invoice",
  payment_id: "Payment",
  account: "Account",
  debit_minor: "Debit",
  credit_minor: "Credit",
  total_minor: "Total",
  amount_minor: "Amount",
  fee_minor: "Fee",
  kind: "Kind",
  issued_at: "Issued",
  due_at: "Due",
  timestamp: "Timestamp",
  entity_id: "Entity",
  original_invoice_id: "Original invoice",
  date: "Date",
  source_currency: "Source",
  target_currency: "Target",
  rate: "FX rate",
  key: "Evidence ID",
};

function handle(row) {
  if (row.row_id) return "ledger:" + row.row_id;
  if (row.doc_id) return "doc:" + row.doc_id;
  return "fx:" + [row.date, row.source_currency, row.target_currency].join(":");
}

export class Explorer {
  constructor(step, editable, attach) {
    this.step = step;
    this.editable = editable;
    this.attach = attach;
    for (const [id, title] of Object.entries(tables)) {
      const tab = button(title, () => this.open(id));
      tab.dataset.table = id;
      $("tabs").append(tab);
    }
    $("search").oninput = () => this.render();
    bind("page-prev", () => this.previous());
    bind("page-next", () => this.next());
    bind("find-next", () => this.searchMore());
  }

  reset(first) {
    this.first = first;
    this.table = "ledger";
    this.books = {};
    for (const t of Object.keys(columns))
      this.books[t] = { pages: new Map(), cursor: null, individual: new Map() };
    $("search").value = "";
  }

  ingest(frame, follow = false) {
    if (!frame.action || !frame.obs.ok) return;
    const { tool, args } = frame.action,
      result = frame.obs.result;
    if (tool === "list_records") {
      const book = this.books[args.table];
      book.pages.set(args.cursor ?? null, result);
      book.cursor = args.cursor ?? null;
      if (follow) this.table = args.table;
    } else if (tool === "get_document" || tool === "get_record") {
      const table = result.table || args.table;
      const row = result.document || result.record;
      this.books[table].individual.set(handle(row), row);
      if (follow) this.table = table;
    }
  }

  async open(table) {
    this.table = table;
    $("search").value = "";
    this.render();
    if (table !== "meta" && !this.books[table].pages.size && this.editable())
      await this.load(table, null);
  }

  async load(table, cursor) {
    const frame = await this.step("list_records", { table, cursor, limit: 50 });
    return frame?.obs.ok ? frame.obs.result : null;
  }

  async go(cursor) {
    const book = this.books[this.table];
    if (book.pages.has(cursor)) {
      book.cursor = cursor;
      this.render();
    } else if (this.editable()) await this.load(this.table, cursor);
  }

  async next() {
    const book = this.books[this.table];
    const next = book.pages.get(book.cursor)?.next_cursor;
    if (next) await this.go(next);
  }

  async previous() {
    const book = this.books[this.table];
    for (const [cursor, page] of book.pages) {
      if (page.next_cursor === book.cursor) {
        await this.go(cursor);
        break;
      }
    }
  }

  matches(row) {
    const query = $("search").value.trim().toLowerCase();
    return (
      !query ||
      Object.values(row).some((v) => typeof v === "string" && v.toLowerCase().includes(query)) ||
      handle(row).toLowerCase().includes(query)
    );
  }

  async searchMore() {
    const book = this.books[this.table];
    let cursor = book.pages.get(book.cursor)?.next_cursor;
    while (cursor && (book.pages.has(cursor) || this.editable())) {
      await this.go(cursor);
      const page = book.pages.get(cursor);
      if (!page || page.items.some((row) => this.matches(row))) return;
      cursor = page.next_cursor;
    }
    tell("No further match in the available pages.");
  }

  async linked(table, key, value) {
    this.table = table;
    $("search").value = value;
    const book = this.books[table];
    let cursor = null;
    do {
      const page =
        book.pages.get(cursor) ?? (this.editable() ? await this.load(table, cursor) : null);
      if (!page) break;
      const doc = page.items.find((row) => row[key] === value);
      if (doc) {
        book.cursor = cursor;
        this.render();
        await this.inspect(table, doc);
        return;
      }
      cursor = page.next_cursor;
    } while (cursor);
    this.render();
    tell("No matching document in the available pages.");
  }

  async inspect(table, row) {
    const id = handle(row);
    if (this.editable()) {
      const frame = row.doc_id
        ? await this.step("get_document", { doc_id: row.doc_id })
        : await this.step("get_record", { table, id: id.slice(id.indexOf(":") + 1) });
      if (!frame?.obs.ok) return;
      row = frame.obs.result.document || frame.obs.result.record;
    }
    const actions = el("div", null, "actions");
    if (this.editable()) {
      for (const [key, text] of [
        ["affected_ids", "+ Affected ID"],
        ["evidence_ids", "+ Add to evidence"],
      ]) {
        actions.append(
          button(text, () => {
            this.attach(key, id);
            $("detail").close();
          }),
        );
      }
    }
    modal(id, actions, pretty(row));
  }

  cell(row, key) {
    const value = key === "key" ? handle(row) : row[key];
    const cell = el("td");
    if (value == null) {
      cell.append(el("span", "—", "muted"));
      return cell;
    }
    if (key.endsWith("_minor")) {
      cell.className = "numcol";
      cell.append(
        el("span", amount(value, row.currency, this.first.manifest.currency_exponents), "money"),
        el("span", value + " minor", "minor"),
      );
    } else if (
      [
        "row_id",
        "doc_id",
        "key",
        "invoice_id",
        "payment_id",
        "original_invoice_id",
        "entity_id",
      ].includes(key)
    ) {
      const table = this.table;
      const click = async () => {
        if (key === "entity_id") {
          $("search").value = value;
          this.render();
        } else if (
          key === "original_invoice_id" ||
          (key === "invoice_id" && table !== "invoices")
        ) {
          await this.linked("invoices", "invoice_id", value);
        } else if (key === "payment_id" && table !== "payments") {
          await this.linked("payments", "payment_id", value);
        } else await this.inspect(table, row);
      };
      cell.append(button(value, click, "id-link"));
      if (["row_id", "doc_id", "key"].includes(key) && this.editable()) {
        const quick = button("+", () => this.attach("evidence_ids", handle(row)), "quick-add");
        quick.title = "Add " + handle(row) + " to evidence";
        quick.setAttribute("aria-label", quick.title);
        cell.append(quick);
      }
    } else cell.textContent = value;
    return cell;
  }

  render() {
    if (!this.first) return;
    for (const tab of $("tabs").children) {
      tab.classList.toggle("active", this.table === tab.dataset.table);
      tab.setAttribute("aria-pressed", String(this.table === tab.dataset.table));
    }
    const meta = this.table === "meta";
    for (const id of ["book-tools", "pager", "cursor-row"]) $(id).hidden = meta;
    const parent = $("records");
    parent.replaceChildren();
    if (meta) {
      $("total").textContent = "2 documents";
      const list = el("div", null, "meta-list");
      for (const key of ["manifest", "policy"]) {
        const detail = el("details");
        detail.open = key === "manifest";
        detail.append(el("summary", key + ".json"), pretty(this.first[key]));
        list.append(detail);
      }
      parent.append(list);
      return;
    }
    const book = this.books[this.table],
      page = book.pages.get(book.cursor);
    const items = page?.items ?? [...book.individual.values()];
    const matches = items.filter((row) => this.matches(row));
    $("total").textContent = this.first.tables[this.table] + " records";
    const table = el("table"),
      header = table.createTHead().insertRow();
    for (const key of columns[this.table])
      header.append(el("th", names[key], key.endsWith("_minor") ? "numcol" : ""));
    const body = table.createTBody();
    for (const row of matches) {
      const tr = body.insertRow();
      for (const key of columns[this.table]) tr.append(this.cell(row, key));
    }
    parent.append(table);
    if (!matches.length)
      parent.append(
        el(
          "p",
          items.length ? "No matching records on this page." : "No records inspected at this step.",
          "empty",
        ),
      );
    const keys = [...book.pages.keys()];
    $("page-range").textContent = page
      ? "Page " +
        (keys.indexOf(book.cursor) + 1) +
        " · " +
        matches.length +
        " / " +
        items.length +
        " rows"
      : items.length + " inspected";
    $("cursor").textContent = page ? (page.next_cursor ?? "null · end of table") : "—";
    $("page-prev").disabled =
      book.cursor === null || ![...book.pages.values()].some((p) => p.next_cursor === book.cursor);
    $("page-next").disabled =
      !page?.next_cursor || (!this.editable() && !book.pages.has(page.next_cursor));
    $("find-next").disabled = !$("search").value.trim() || $("page-next").disabled;
  }
}
