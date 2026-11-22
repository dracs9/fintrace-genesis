import { $, bind, button, el, icons, modal, pretty, task, tell, wire } from "./ui.js";
import { Explorer } from "./explorer.js";
import { Findings, verdict } from "./findings.js";

icons();
let revision = '"0"';
let frames = [],
  index = 0,
  grade = null,
  policy = "manual",
  timer = null;
let measurements = [],
  selectedGroup = "";
const first = () => frames[0]?.obs.result;
const current = () => frames[index];
const active = () =>
  policy === "manual" && current() && !current().terminated && !current().truncated;
const builder = new Findings((rows) =>
  step("submit", { findings: { episode_id: first().manifest.episode_id, findings: rows } }),
);
const explorer = new Explorer(step, active, (key, id) => builder.attach(key, id));

async function api(path, body, force = false) {
  const headers = {};
  const options = { headers, cache: "no-store" };
  if (body !== undefined) {
    options.method = "POST";
    headers["Content-Type"] = "application/json";
    if (!force) headers["If-Match"] = revision;
    options.body = wire(body);
  }
  const response = await fetch("/api/" + path, options);
  const json = JSON.parse(await response.text(), (key, value, context) =>
    typeof value === "number" &&
    Number.isInteger(value) &&
    !Number.isSafeInteger(value) &&
    context?.source &&
    /^-?\d+$/.test(context.source)
      ? BigInt(context.source)
      : value,
  );
  if (!response.ok) {
    if (response.status === 409)
      throw Error("Another tab changed this workspace. Start a new episode to take control.");
    throw Error(json.error?.message || "Request failed.");
  }
  if (body !== undefined) revision = response.headers.get("ETag");
  return json;
}

function settings() {
  const seed = $("seed").value.trim();
  if (!/^\d+$/.test(seed)) throw Error("Enter a nonnegative integer seed.");
  return { seed: BigInt(seed), config: $("preset").value, mode: $("mode").value };
}

function stop() {
  clearInterval(timer);
  timer = null;
  $("play").textContent = "Play";
}

function start(firstFrame, name) {
  stop();
  frames = [firstFrame];
  index = 0;
  policy = name;
  grade = null;
  explorer.reset(first());
  builder.reset(first());
  $("calc-result").textContent = "";
}

async function reset(force = true) {
  const config = settings();
  tell("Generating episode…");
  const response = await api("env/reset", config, force);
  start({ ...response, action: null, terminated: false, truncated: false, reward: 0 }, "manual");
  $("policy").value = "manual";
  view(false);
  tell();
  render();
  await explorer.open("ledger");
}

async function step(tool, args) {
  if (!active()) throw Error("The episode has ended.");
  const action = { tool, args };
  const response = await api("env/step", action);
  const { grade: result, ...frame } = response;
  frames.push({ ...frame, action });
  index = frames.length - 1;
  grade = result || null;
  explorer.ingest(current());
  tell(frame.obs.ok ? "" : frame.obs.error.message);
  render();
  return frame;
}

function showReplay(result) {
  start(result.trajectory[0], result.policy);
  frames = result.trajectory;
  grade = result.grade;
  $("policy").value = policy === "random_explorer" ? "random" : policy;
  seek(0);
}

function seek(position) {
  if (position < 0 || position >= frames.length) return;
  index = position;
  explorer.reset(first());
  let calc = "";
  for (const frame of frames.slice(1, index + 1)) {
    explorer.ingest(frame, true);
    if (frame.action.tool === "calculator")
      calc = frame.obs.ok ? frame.obs.result.value : frame.obs.error.message;
  }
  $("calc-result").textContent = calc;
  render();
}

function trajectory() {
  const log = $("log");
  const earlier = log.scrollTop;
  log.replaceChildren();
  $("log-count").textContent = frames.length;
  frames.forEach((frame, i) => {
    const row = button(
      "",
      () => {
        if (policy !== "manual") {
          stop();
          seek(i);
        }
        const { action, ...response } = frame;
        modal(
          "Step " + i + " · " + (action?.tool || "reset"),
          el("div", "Action", "eyebrow"),
          pretty(action),
          el("div", "Environment response", "eyebrow"),
          pretty(response),
        );
      },
      "logrow" + (index === i ? " current" : "") + (frame.obs.ok ? "" : " error"),
    );
    const action = frame.action;
    const description = action
      ? frame.obs.error?.code ||
        action.args.table ||
        action.args.doc_id ||
        (action.tool === "submit" ? "terminal submission" : action.args.expression)
      : "episode initialized";
    const text = el("span");
    text.append(el("strong", action?.tool || "reset"), el("small", description));
    row.append(el("span", String(i).padStart(3, "0"), "num"), text);
    log.append(row);
  });
  if (policy === "manual") log.scrollTop = log.scrollHeight;
  else {
    log.scrollTop = earlier;
    const row = log.children[index];
    if (
      row &&
      row.offsetTop - log.offsetTop > log.scrollTop + log.clientHeight - row.offsetHeight
    ) {
      log.scrollTop = row.offsetTop - log.offsetTop - log.clientHeight + row.offsetHeight;
    }
  }
}

function render() {
  if (!current()) return;
  const frame = current(),
    m = first().manifest;
  $("episode").textContent = m.episode_id;
  $("period").textContent = m.period_start.slice(0, 10) + " / " + m.period_end.slice(0, 10);
  $("base").textContent = m.base_currency;
  const steps = frame.obs.steps_used,
    max = steps + frame.obs.steps_left;
  $("steps").textContent = steps + " / " + max;
  $("progress").max = max;
  $("progress").value = steps;
  const status = frame.truncated ? "TRUNCATED" : frame.terminated ? "TERMINATED" : "ACTIVE";
  $("status").textContent = (policy === "manual" ? "" : "REPLAY · ") + status;
  $("status").dataset.state = status;
  $("player").hidden = policy === "manual";
  $("position").textContent = index + " / " + (frames.length - 1);
  $("next").disabled = index === frames.length - 1;
  $("prev").disabled = index === 0;
  $("run").disabled = $("policy").value === "manual";
  $("calc").querySelector("button").disabled = !active();
  $("expression").disabled = !active();
  trajectory();
  explorer.render();
  builder.render(!active(), frame.terminated ? frame.action.args.findings.findings : null);
  verdict(frame.terminated || frame.truncated ? grade : null, frame.reward);
}

function view(bench) {
  document.body.classList.toggle("benchmark-view", bench);
  $("work").hidden = bench;
  $("bench").hidden = !bench;
  const accent = el("span", bench ? "Compare policy scores." : "Trace the evidence.");
  $("title").replaceChildren(
    bench ? "Benchmark the agents." : "Audit the books.",
    el("br"),
    accent,
  );
  $("hero-description").textContent = bench
    ? "Measure precision, recall, and reward across four policies on the same episode."
    : "Explore a generated ledger, build findings, and compare how audit policies score.";
  $("nav-work").classList.toggle("active", !bench);
  $("nav-bench").classList.toggle("active", bench);
  if (bench) stop();
}

const groupKey = (row) => [row.episode_id, row.split || "CLI", row.config || ""].join(" · ");
function benchmark() {
  const groups = [...new Set(measurements.map(groupKey))].sort();
  if (!groups.includes(selectedGroup))
    selectedGroup =
      groups.find((g) => g.startsWith(first()?.manifest.episode_id)) || groups[0] || "";
  const select = $("cohort");
  select.replaceChildren();
  if (!groups.length) select.add(new Option("No completed runs", ""));
  for (const group of groups) select.add(new Option(group, group));
  select.value = selectedGroup;
  const chosen = measurements.filter((r) => groupKey(r) === selectedGroup);
  const body = $("baseline-rows");
  body.replaceChildren();
  for (const name of ["oracle", "reconciler", "flag_all", "random_explorer"]) {
    const samples = chosen.filter(
      (r) => (r.policy === "random" ? "random_explorer" : r.policy) === name,
    );
    const row = body.insertRow();
    row.insertCell().textContent = name;
    let reward;
    for (const key of ["precision", "recall", "f1", "steps", "reward"]) {
      const cell = row.insertCell();
      if (!samples.length) {
        cell.textContent = "—";
        continue;
      }
      const v = samples.reduce((sum, run) => sum + run[key], 0) / samples.length;
      cell.textContent = key === "steps" ? String(v) : v.toFixed(3);
      if (key === "reward") reward = v;
    }
    const cell = row.insertCell(),
      bar = el("div", null, "bar");
    if (reward !== undefined) {
      const fill = el("i", null, reward < 0 ? "neg" : "");
      fill.style.width = Math.abs(reward) * 50 + "%";
      fill.style.left = (reward >= 0 ? 50 : 50 + reward * 50) + "%";
      bar.append(fill);
    }
    cell.append(bar);
  }
  $("bench-note").textContent = chosen.length
    ? chosen.length +
      " completed measurements for this episode. Repeated measurements are averaged per policy."
    : "Run all four policies on the selected seed and preset, or generate CLI summaries with fintrace all.";
}

async function refresh() {
  const data = await api("runs");
  measurements = data.runs;
  benchmark();
}

bind("nav-work", () => view(false));
document.querySelector(".brand").onclick = () => view(false);
bind("nav-bench", async () => {
  view(true);
  await refresh();
});
bind("reset", () => reset());
$("policy").onchange = () => {
  $("run").disabled = $("policy").value === "manual";
};
bind("run", async () => {
  stop();
  const name = $("policy").value;
  tell("Running " + name + " to completion…");
  const result = await api("env/run-policy", { ...settings(), policy: name });
  showReplay(result);
  tell();
});
bind("prev", () => {
  stop();
  seek(index - 1);
});
bind("next", () => {
  stop();
  seek(index + 1);
});
bind("play", () => {
  if (timer) {
    stop();
    return;
  }
  if (index === frames.length - 1) seek(0);
  $("play").textContent = "Pause";
  timer = setInterval(() => {
    seek(index + 1);
    if (index === frames.length - 1) stop();
  }, 650);
});
$("calc").onsubmit = (event) => {
  event.preventDefault();
  task(async () => {
    const frame = await step("calculator", { expression: $("expression").value });
    $("calc-result").textContent = frame.obs.ok
      ? "= " + frame.obs.result.value
      : frame.obs.error.message;
  });
};
$("dismiss").onclick = () => $("detail").close();
$("detail").onclick = (event) => {
  if (event.target === $("detail")) $("detail").close();
};
$("cohort").onchange = () => {
  selectedGroup = $("cohort").value;
  benchmark();
};
bind("refresh", refresh);
bind("measure", async () => {
  const config = settings();
  stop();
  for (const name of ["oracle", "reconciler", "flag_all", "random"]) {
    tell("Measuring " + name + "…");
    const result = await api("env/run-policy", { ...config, policy: name });
    showReplay(result);
    selectedGroup = groupKey({
      episode_id: result.grade.episode_id,
      split: "studio",
      config: config.config,
    });
    await refresh();
  }
  tell();
});
task(() => reset(false));
