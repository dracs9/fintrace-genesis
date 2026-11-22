export const $ = (id) => document.getElementById(id);

export function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

export function button(text, run, className, disabled = false) {
  const node = el("button", text, className);
  node.type = "button";
  node.disabled = disabled;
  node.onclick = () => task(run);
  return node;
}

export function tell(message = "") {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}

let working = false;
export async function task(run) {
  if (working) return;
  working = true;
  document.querySelector("main").inert = true;
  document.querySelector("main").setAttribute("aria-busy", "true");
  try {
    await run();
  } catch (error) {
    tell(error.message || "Request failed.");
  } finally {
    working = false;
    document.querySelector("main").inert = false;
    document.querySelector("main").setAttribute("aria-busy", "false");
  }
}

export function bind(id, run) {
  $(id).onclick = () => task(run);
}

export function modal(title, ...content) {
  $("detail-title").textContent = title;
  $("detail-body").replaceChildren(...content);
  if (!$("detail").open) $("detail").showModal();
}

// Serialize arbitrarily large integer claims as JSON numbers on all supported browsers.
export function wire(value, indent = "", depth = 0) {
  if (typeof value === "bigint") return value.toString();
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  const array = Array.isArray(value);
  const entries = array
    ? value.map((item) => wire(item, indent, depth + 1))
    : Object.keys(value)
        .sort()
        .map(
          (key) =>
            JSON.stringify(key) + (indent ? ": " : ":") + wire(value[key], indent, depth + 1),
        );
  const line = indent ? "\n" + indent.repeat(depth + 1) : "";
  const end = indent ? "\n" + indent.repeat(depth) : "";
  const inner = entries.length ? line + entries.join("," + line) + end : "";
  return array ? "[" + inner + "]" : "{" + inner + "}";
}

export function pretty(value) {
  return el("pre", wire(value, "  "));
}

export function amount(value, currency, exponents) {
  const n = BigInt(value);
  const places = exponents[currency];
  if (!Number.isInteger(places)) return String(value) + " minor";
  const digits = (n < 0n ? -n : n).toString().padStart(places + 1, "0");
  const integer = places ? digits.slice(0, -places) : digits;
  return (
    (n < 0n ? "−" : "") +
    integer.replace(/\B(?=(\d{3})+(?!\d))/g, ",") +
    (places ? "." + digits.slice(-places) : "") +
    " " +
    currency
  );
}

export function icons() {
  const paths = {
    grid: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
    chart: "M3 3v17h18 M7 15v-4 M12 15V7 M17 15v-6",
    reset: "M3 11a9 9 0 1 1 2 7 M3 4v7h7",
    sliders: "M3 6h5m5 0h8 M3 17h10m5 0h3 M8 3h5v6H8z M13 14h5v6h-5z",
    play: "m7 4 13 8-13 8z",
    activity: "M2 12h4l3-8 5 16 3-8h5",
    books: "M3 4h5v16H3z M10 4h4v16h-4z M15 6l4-1 3 14-4 1z",
    search: "M16 16l5 5 M18 10.5a7.5 7.5 0 1 1-15 0 7.5 7.5 0 0 1 15 0",
    calc: "M5 2h14v20H5z M8 6h8 M8 11h2m4 0h2 M8 15h2m4 0h2 M8 18h2m4 0h2",
    file: "M5 2h10l4 4v16H5z M14 2v6h5 M8 12h8 M8 16h6",
    scale: "M12 2v18 M5 21h14 M3 6h18 M5 7l-3 7h6z M19 7l-3 7h6z",
  };
  for (const node of document.querySelectorAll("[data-icon]")) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("class", "icon");
    svg.setAttribute("aria-hidden", "true");
    const path = document.createElementNS(svg.namespaceURI, "path");
    path.setAttribute("d", paths[node.dataset.icon]);
    svg.append(path);
    node.replaceWith(svg);
  }
}
