/* Earnings-Vol board client.
 *
 * Display only: every number shown arrives pre-computed in the bundle
 * (ScoreResult fields + render-derived display values). This file formats,
 * sorts and filters — it never derives a statistic of its own.
 *
 * Works from file:// as well as http://: the board/meta/health/flags data
 * load through <script> tags (fetch() is blocked on file:// origins), and
 * per-ticker files load lazily by injecting their .js wrapper.
 */
"use strict";

const BOARD = window.BOARD || { rows: [] };
const META = window.META || {};
const HEALTH = window.HEALTH || {};
const BOOK = window.BOOK || {};
const FLAGS = window.FLAGS || { flags: [] };
const STRATEGIES = window.STRATEGIES || {};
let MODELS = {};        /* filled when the Models area is first opened */
let MODELS_META = {};
const TICKER_DATA = window.TICKER_DATA || {};

/* How many rows the out-of-domain switch is currently hiding, under whatever
   other filters are set. Kept beside the count badge so a shrunken board says
   why it shrank rather than looking like a thin night. */
let boardHidden = 0;

const state = {
  sortKey: "event_date",
  sortDir: 1,
  strategy: "",
  gate: "",
  ticker: "",
  /* Off by default: a withheld gate is not a decision, and a name the champion
     gate was never trained on has no verdict to read. Those rows belong behind
     a switch rather than in the middle of the ones that do carry a call. */
  outOfDomain: false,
  tickerSelected: null,
};

/* ---------------------------------------------------------------- fmt */

function fmt(x, nd) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  return Number(x).toFixed(nd === undefined ? 1 : nd);
}
function pct(x, nd) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  return (Number(x) * 100).toFixed(nd === undefined ? 1 : nd) + "%";
}
function signedPct(x, nd) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  const v = Number(x) * 100;
  return (v >= 0 ? "+" : "") + v.toFixed(nd === undefined ? 1 : nd) + "%";
}
function cls(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "flat";
  return x > 0.001 ? "pos" : x < -0.001 ? "neg" : "flat";
}
function esc(s) {
  return String(s === null || s === undefined ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
function cellColor(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "";
  const v = Math.max(-0.08, Math.min(0.08, Number(x)));
  if (v >= 0) return "rgba(63,185,80," + (0.12 + (v / 0.08) * 0.5).toFixed(2) + ")";
  return "rgba(248,81,73," + (0.12 + (-v / 0.08) * 0.5).toFixed(2) + ")";
}

/* ---------------------------------------------------------------- tabs */

/* Two areas: finding trades, and understanding the models that rank them. */
const AREA_VIEWS = { trades: ["board", "explorer", "book"], models: ["modelx", "derivation", "health"] };
const ALL_VIEWS = ["board", "explorer", "book", "modelx", "derivation", "health"];

function switchTab(name) {
  document.querySelectorAll(".tab[data-tab]").forEach(
    (t) => t.classList.toggle("active", t.dataset.tab === name)
  );
  ALL_VIEWS.forEach((v) =>
    document.getElementById("view-" + v).classList.toggle("hidden", v !== name)
  );
}

function switchArea(area) {
  if (area === "models") initModelExplorer();
  document.querySelectorAll(".tab.area").forEach(
    (t) => t.classList.toggle("active", t.dataset.area === area)
  );
  Object.keys(AREA_VIEWS).forEach((a) =>
    document.getElementById("subtabs-" + a).classList.toggle("hidden", a !== area)
  );
  switchTab(AREA_VIEWS[area][0]);
}

/* ---------------------------------------------------------------- flags */

function flagText(flag) {
  if (flag.kind === "new_gate_triggers") return "new gate trigger(s): " + (flag.rows || []).join(", ");
  if (flag.kind === "earnings_date_changed") {
    return "earnings date changed: " + (flag.changes || []).map(
      (c) => c.ticker + " " + c.change + " " + (c.old || "–") + " → " + (c.new || "–")
    ).join("; ");
  }
  if (flag.kind === "calibration_drift") return "calibration drift: " + (flag.reasons || []).join("; ");
  if (flag.kind === "calendar_date_conflict") {
    const names = Object.keys(flag.tickers || {});
    return "forward sources disagree on the print date for " + names.length
      + " name(s) — both rows are shown and one is wrong: "
      + names.slice(0, 8).map((t) => t + " (" + (flag.tickers[t] || []).join(" vs ") + ")").join(", ");
  }
  if (flag.kind === "no_upcoming_events") return "no confirmed events in the horizon: " + flag.detail;
  if (flag.kind === "quota_below_reserve") return "ORATS quota below reserve (" + flag.remaining + " left)";
  if (flag.kind === "refresh_degraded") return "data refresh degraded: " + flag.detail;
  if (flag.kind === "late_backfill") return "ledger backfilled (LATE) for: " + (flag.as_ofs || []).join(", ");
  if (flag.kind === "publish_failed") return "publish failed: " + flag.detail;
  if (flag.kind === "aborted") return "nightly aborted at " + flag.step + ": " + flag.detail;
  return flag.kind + ": " + JSON.stringify(flag).slice(0, 160);
}

function renderFlagsBanner() {
  const el = document.getElementById("flags-banner");
  const flags = FLAGS.flags || [];
  if (!flags.length) { el.classList.add("hidden"); return; }
  el.classList.remove("hidden");
  el.innerHTML = flags.map(
    (f) => '<div class="flag-line">⚑ ' + esc(flagText(f)) + "</div>"
  ).join("");
}

/* ---------------------------------------------------------------- board */

/* The DECISION layer. Shown as predicted-return vs threshold, in the same
   percent units as the P&L columns, because "PASS 0.10" beside "+55.2%"
   invited reading the score as a probability — it is a predicted return,
   and the comparison against the threshold is the whole decision. */
function gatePill(row) {
  const thr = row.gate_threshold === null || row.gate_threshold === undefined
    ? "" : " " + (row.gate_pass === true ? "&ge; " : "&lt; ") + pct(row.gate_threshold, 1);
  if (row.gate_pass === true) return '<span class="pill pass">PASS ' + signedPct(row.gate_score, 1) + thr + "</span>";
  if (row.gate_pass === false) return '<span class="pill fail">fail ' + signedPct(row.gate_score, 1) + thr + "</span>";
  return '<span class="pill na">n/a</span>';
}

/* Gate passes while the model Monte-Carlo forecast is negative: two
   independent estimators of the same trade's return, opposite signs. The
   gate never sees the size model, so this is a real disagreement, not a
   rounding artefact — surfaced rather than hidden. */
function splitBadge(row) {
  if (row.gate_pass === true && row.exp_pnl_model !== null && row.exp_pnl_model !== undefined && row.exp_pnl_model < 0) {
    return ' <span class="pill warn" title="Gate passes but the model Monte-Carlo forecast is negative. The gate never sees the size model; two independent estimators disagreeing this hard is rare — treat the row with suspicion rather than as a trade.">gate vs model</span>';
  }
  return "";
}

/* Flags that say something about the EVENT rather than about this structure.
   They are identical on every sub-row of a print, so a pill each turned one
   fact into three or four; they are shown once, on the print row. */
const EVENT_FLAGS = { PROJECTED_CALENDAR: 1 };

function flagBadges(row) {
  const out = [];
  if (row.strategy === "CAL-P") out.push('<span class="pill warn">unvalidated — pending EXP-101/102</span>');
  if (row.extrapolated) out.push('<span class="pill warn">EXTRAPOLATED</span>');
  (row.flags || []).forEach((f) => {
    if (f === "UNVALIDATED_STRUCTURE") return;  /* badged above, with the detail text */
    if (f === "EXTRAPOLATED") return;
    if (EVENT_FLAGS[f]) return;                 /* carried on the print row */
    const klass = f === "LAYER_DISAGREE" ? "warn"
      : f === "THIN_ANALOGS" || f === "THIN_HISTORY" || f === "OUT_OF_DOMAIN" ? "info" : "na";
    /* NO_CHAIN says nothing on its own; the age of the newest chain says
       whether a refresh fixes this row or the name was never covered. */
    const label = f === "NO_CHAIN" && row.chain_age_days !== null && row.chain_age_days !== undefined
      ? "NO_CHAIN (newest " + row.chain_age_days + "d old)"
      : f === "NO_CHAIN" && !row.chain_last_obs ? "NO_CHAIN (never pulled)" : f;
    out.push('<span class="pill ' + klass + '" title="' + esc(row.chain_last_obs || "") + '">' + esc(label) + "</span>");
  });
  return out.join(" ");
}

/* The champion's own prediction. It needs no option chain, so it is present on
   rows whose P&L columns are not — which is most of a board more than a day or
   two ahead of its prints. */
function driverCell(r) {
  if (r.driver_prediction === null || r.driver_prediction === undefined) return "–";
  const label = r.driver_name === "abs_move" ? "|move|"
    : r.driver_name === "implied_t1" ? "T–1 implied" : (r.driver_name || "");
  /* The point estimate alone reads as more certain than it is. The band is the
     10th-90th percentile of the model's own draws. Since EXP-115 those draws
     come from the residual bucket matching this row's prediction rather than
     one pool shared by every event: measured coverage 79.3% against a nominal
     80%, up from 72.8% when the pool was flat. The width now varies by event —
     narrow where the model is reliably right, wide where it is not — which is
     the information a single shared width destroyed. */
  let band = "";
  if (r.driver_p10 !== null && r.driver_p10 !== undefined
      && r.driver_p90 !== null && r.driver_p90 !== undefined) {
    band = " <span class='band' title='10th-90th percentile of the model draws."
      + " Drawn from the residual bucket for this prediction. Nominally an 80%"
      + " band; measured coverage 79.3% (EXP-115).'>"
      + fmt(Math.max(r.driver_p10, 0), 1) + "–" + fmt(r.driver_p90, 1) + "</span>";
  }
  return fmt(r.driver_prediction, 1) + "%" + band
    + " <span class='badge'>" + esc(label) + "</span>";
}

/* What the market says, beside what the model says. The board carried the
   prediction with nothing to compare it against, which is the one comparison
   the whole programme is built on: predicted |move| against quoted implied. */
function impliedCell(r) {
  if (r.implied_move === null || r.implied_move === undefined) return "–";
  return fmt(r.implied_move, 1) + "%";
}

/* The ratio is DERIVED BY THE RENDERER, not here. It first shipped as a
   division in this file, which is a UI computation the board's rules forbid —
   and `ui_no_compute` did not catch it, because that check compares the field
   names a client reads against the ones the renderer wrote, and dividing two
   legitimate fields reads as legitimate. */
function ratioCell(r) {
  if (r.model_vs_market === null || r.model_vs_market === undefined) return "–";
  /* Not coloured. Below 1.0 means the model expects less movement than the
     option is priced for — which is bad for a long straddle and good for a
     short one, and the board carries both kinds. */
  return "×" + fmt(r.model_vs_market, 2);
}

/* The PRINT row: everything that is a property of the event rather than of a
   structure. All three forecasts belong here — how far the stock moves, what
   the market charges now, and what it will charge at T-1 — because they are
   statements about the event. That they arrive on different strategy rows is an
   artefact of which driver each strategy happens to use, not a real difference.
   Repeating event values per structure is what made two correct rows look like
   they disagreed. */
function printRow(members) {
  const head = members[0];
  const byDriver = (name) => members.find((m) => m.driver_name === name) || {};
  const move = byDriver("abs_move");
  const t1 = byDriver("implied_t1");
  const passes = members.filter((m) => m.gate_pass === true).length;
  /* PROJECTED_CALENDAR is a statement about the DATE, so it is said next to
     the date: a print no forward source has confirmed is an estimate, and the
     estimate is what everything on the row is anchored to. */
  const projected = members.some((m) => (m.flags || []).indexOf("PROJECTED_CALENDAR") !== -1)
    ? " <span class='badge' title='No forward source confirms this date yet — it is projected from the name\u2019s own print history, so it can move.'>est.</span>"
    : "";
  return '<tr class="tickerrow clickable" data-ticker="' + esc(head.ticker) + '">'
    + "<td><strong>" + esc(head.event_date) + "</strong>" + projected
    + "<br><span class='badge'>" + esc(head.session || "") + "</span></td>"
    + "<td><strong>" + esc(head.ticker) + "</strong>"
    + (passes ? " <span class='pill pass'>" + passes + "</span>" : "") + "</td>"
    + "<td colspan='2'></td>"
    + "<td>" + driverCell(move) + "</td>"
    + "<td>" + impliedCell(head) + "</td>"
    + "<td>" + ratioCell(move) + "</td>"
    + "<td>" + (t1.driver_prediction === null || t1.driver_prediction === undefined
        ? "–" : fmt(t1.driver_prediction, 1) + "%") + "</td>"
    + "<td colspan='8'><span class='badge'>" + members.length + " structure"
    + (members.length === 1 ? "" : "s") + "</span></td>"
    + "</tr>";
}

/* One structure under its print: only what genuinely differs between them —
   the gate, the two P&L layers, the win rate, and the dates it would trade.
   Cell order mirrors the grouped header: trade (strategy, dates), signal
   (blank — event-level), cost, decision (gate, rank), forecast (model PnL,
   win), evidence (analog PnL, n), flags. */
function strategyRow(r, disabled, winCell, premium) {
  return '<tr class="subrow clickable' + (disabled ? " disabled" : "") + '" data-ticker="'
    + esc(r.ticker) + '">'
    + "<td></td><td></td>"
    + "<td><span class='band'>└</span> " + esc(r.strategy) + "</td>"
    + "<td>" + esc(r.entry_date || "–") + " → " + esc(r.exit_date || "–") + "</td>"
    + "<td></td><td></td><td></td><td></td>"
    + "<td>" + premium + "</td>"
    + "<td>" + gatePill(r) + splitBadge(r) + "</td>"
    + "<td>" + (r.rank || "–") + "</td>"
    + '<td class="' + cls(r.exp_pnl_model) + '">' + signedPct(r.exp_pnl_model, 2) + "</td>"
    + "<td>" + winCell + "</td>"
    + '<td class="' + cls(r.exp_pnl_analog) + '">' + signedPct(r.exp_pnl_analog, 2) + "</td>"
    + "<td>" + (r.n_analogs === null || r.n_analogs === undefined ? "–" : r.n_analogs) + "</td>"
    + "<td>" + flagBadges(r) + "</td>"
    + "</tr>";
}

/* BMO sorts before AMC on the same date because that is the order the
   DECISIONS come due, not the order the prints do: a BMO print is entered at
   the previous session's close, an AMC print at the close of the print day
   itself. Sorting the label alphabetically would put AMC first and quietly
   invert the schedule. */
const SESSION_ORDER = { BMO: 0, AMC: 1 };

function sessionOrder(session) {
  const v = SESSION_ORDER[String(session || "").toUpperCase()];
  return v === undefined ? 2 : v;
}

/* What the board is proposing to do with the row, in the order a reader wants
   to meet it: a tradeable call first, then a call not to trade, then rows
   where no call exists. This is the gate verdict — the selection rule the
   backtests used — and not a re-ranking of any kind. */
function actionRank(row) {
  if (row.gate_pass === true) return 0;
  if (row.gate_pass === false) return 1;
  return 2;
}

function groupKey(row) {
  return row.ticker + "|" + row.event_date;
}

function isOutOfDomain(row) {
  return (row.flags || []).indexOf("OUT_OF_DOMAIN") !== -1;
}

/* The default order: when the decision is due, then what the decision is.
   `dir` flips the schedule only — passers stay at the top of their session
   either way, because reversing the date is a request to read the calendar
   backwards, not a request to bury the trades. */
function scheduleOrder(a, b, dir, groupAction) {
  const ad = String(a.event_date || ""), bd = String(b.event_date || "");
  if (ad !== bd) return ad.localeCompare(bd) * dir;
  const as = sessionOrder(a.session), bs = sessionOrder(b.session);
  if (as !== bs) return (as - bs) * dir;
  const ag = groupAction.get(groupKey(a)), bg = groupAction.get(groupKey(b));
  if (ag !== bg) return ag - bg;
  const at = String(a.ticker || ""), bt = String(b.ticker || "");
  if (at !== bt) return at.localeCompare(bt);
  return actionRank(a) - actionRank(b);
}

function boardRows() {
  let rows = BOARD.rows.slice();
  if (state.strategy) rows = rows.filter((r) => r.strategy === state.strategy);
  if (state.gate === "pass") rows = rows.filter((r) => r.gate_pass === true);
  if (state.gate === "fail") rows = rows.filter((r) => r.gate_pass === false);
  if (state.gate === "na") rows = rows.filter((r) => r.gate_pass === null || r.gate_pass === undefined);
  if (state.ticker) rows = rows.filter((r) => (r.ticker || "").toLowerCase().includes(state.ticker));
  if (state.outOfDomain) {
    boardHidden = 0;
  } else {
    const before = rows.length;
    rows = rows.filter((r) => !isOutOfDomain(r));
    boardHidden = before - rows.length;
  }

  /* A print is placed by the strongest verdict any of its structures carries,
     so a name with one tradeable structure sorts above a name with none — and
     its other structures stay with it rather than scattering down the board. */
  const groupAction = new Map();
  rows.forEach((r) => {
    const k = groupKey(r), a = actionRank(r);
    if (!groupAction.has(k) || a < groupAction.get(k)) groupAction.set(k, a);
  });

  const key = state.sortKey, dir = state.sortDir;
  rows.sort((a, b) => {
    if (key !== "event_date") {
      const x = a[key], y = b[key];
      if (x === null || x === undefined) {
        if (!(y === null || y === undefined)) return 1;
      } else if (y === null || y === undefined) {
        return -1;
      } else if (typeof x === "string") {
        const c = x.localeCompare(y) * dir;
        if (c) return c;
      } else if (x !== y) {
        return (x - y) * dir;
      }
    }
    return scheduleOrder(a, b, key === "event_date" ? dir : 1, groupAction);
  });
  return rows;
}

function renderBoard() {
  const rows = boardRows();
  document.getElementById("board-count").textContent =
    rows.length + " / " + BOARD.rows.length + " rows"
    + (boardHidden ? " · " + boardHidden + " out-of-domain hidden" : "");

  const tb = document.querySelector("#tbl-board tbody");

  /* Grouped by PRINT, not by (ticker, strategy). A ticker with two structures
     used to be two unrelated-looking rows; the things that actually belong to
     the EVENT — the date, the session, what the market is quoting today — were
     repeated on each and invited the reader to compare them as if they differed.
     Only the per-strategy numbers differ, and those are the sub-rows. */
  const groups = new Map();
  for (const r of rows) {
    const key = r.ticker + "|" + r.event_date;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }

  tb.innerHTML = Array.from(groups.values()).map((members) => {
    return printRow(members) + members.map((r) => {
      const disabled = r.strategy === "CAL-P";
      const winCell = r.win_model === null || r.win_model === undefined
        ? "–"
        : pct(r.win_model, 0) + ' <span class="badge">[' + signedPct(r.ci_low) + ", " + signedPct(r.ci_high) + "]</span>";
      const premium = r.entry_cost_pct === null || r.entry_cost_pct === undefined
        ? "–"
        : fmt(r.entry_cost_pct, 1) + "%" + (r.model_fair_pct !== null && r.model_fair_pct !== undefined
            ? ' <span class="badge">vs fair ' + fmt(r.model_fair_pct, 1) + "%</span>" : "");
      return strategyRow(r, disabled, winCell, premium);
    }).join("");
  }).join("");

  tb.querySelectorAll("tr").forEach((tr) => {
    tr.onclick = () => openExplorer(tr.dataset.ticker);
  });

  document.querySelectorAll("#tbl-board th[data-sort]").forEach((th) => {
    th.classList.remove("sorted-asc", "sorted-desc");
    if (th.dataset.sort === state.sortKey) {
      th.classList.add(state.sortDir === 1 ? "sorted-asc" : "sorted-desc");
    }
  });
}

function initBoardControls() {
  const strategies = Array.from(new Set(BOARD.rows.map((r) => r.strategy))).sort();
  const sel = document.getElementById("f-strategy");
  strategies.forEach((s) => {
    const opt = document.createElement("option");
    opt.value = s; opt.textContent = s;
    sel.appendChild(opt);
  });
  sel.onchange = () => { state.strategy = sel.value; renderBoard(); };
  document.getElementById("f-gate").onchange = (e) => { state.gate = e.target.value; renderBoard(); };
  const ood = document.getElementById("f-ood");
  ood.checked = state.outOfDomain;
  ood.onchange = (e) => { state.outOfDomain = e.target.checked; renderBoard(); };
  document.getElementById("f-ticker").oninput = (e) => {
    state.ticker = e.target.value.trim().toLowerCase(); renderBoard();
  };
  document.querySelectorAll("#tbl-board th[data-sort]").forEach((th) => {
    th.onclick = () => {
      const key = th.dataset.sort;
      if (state.sortKey === key) state.sortDir *= -1;
      else { state.sortKey = key; state.sortDir = 1; }
      renderBoard();
    };
  });
}

/* ---------------------------------------------------------------- explorer */

function loadTickerData(ticker, then) {
  if (TICKER_DATA[ticker]) { then(TICKER_DATA[ticker]); return; }
  const script = document.createElement("script");
  script.src = "data/tickers/" + encodeURIComponent(ticker) + ".js";
  script.onload = () => then(TICKER_DATA[ticker] || null);
  script.onerror = () => then(null);
  document.head.appendChild(script);
}

function openExplorer(ticker) {
  switchArea("trades");
  switchTab("explorer");
  const sel = document.getElementById("x-ticker");
  sel.value = ticker;
  renderExplorer(ticker);
}

function renderExplorer(ticker) {
  state.tickerSelected = ticker;
  const gridEl = document.getElementById("x-grid");
  const detailEl = document.getElementById("x-detail");
  detailEl.classList.add("hidden");
  detailEl.innerHTML = "";
  gridEl.innerHTML = '<span class="badge">loading ' + esc(ticker) + "…</span>";

  loadTickerData(ticker, (data) => {
    if (!data) {
      gridEl.innerHTML = '<span class="badge">no data for ' + esc(ticker) + " — run the nightly job</span>";
      return;
    }
    document.getElementById("x-asof").textContent = "as of " + (data.as_of || META.as_of || "?");
    renderStrikeGrid(data, gridEl, detailEl);
    renderHistory(data);
    renderAnalogs(data);
  });
}

function renderStrikeGrid(data, gridEl, detailEl) {
  const strategies = Array.from(new Set(data.events.flatMap((e) => e.rows.map((r) => r.strategy)))).sort();
  let html = "";
  data.events.forEach((event, eidx) => {
    html += "<h3>" + esc(event.event_date) + " <span class='badge'>" + esc(event.session || "") + "</span></h3>";
    html += '<div class="tablewrap"><table class="grid-table"><thead><tr><th>Strike</th>';
    strategies.forEach((s) => { html += "<th>" + esc(s) + "</th>"; });
    html += "</tr></thead><tbody>";

    const byStrike = {};
    event.rows.forEach((r, ridx) => {
      const key = r.strike === null || r.strike === undefined ? "atm" : Number(r.strike).toFixed(4);
      byStrike[key] = byStrike[key] || { strike: r.strike, offset: r.strike_offset, cells: {} };
      byStrike[key].cells[r.strategy] = ridx;
    });
    const ordered = Object.values(byStrike).sort((a, b) => {
      const sa = a.strike === null ? -Infinity : a.strike;
      const sb = b.strike === null ? -Infinity : b.strike;
      return sa - sb;
    });
    ordered.forEach((entry) => {
      const isAtm = entry.strike === null || entry.offset === null || entry.offset === undefined || entry.offset === 0;
      html += "<tr><td" + (isAtm ? ' class="atm"' : "") + ">"
        + (entry.strike === null || entry.strike === undefined ? "ATM" : fmt(entry.strike, 2))
        + (isAtm ? "" : ' <span class="pill warn">EXTRAPOLATED</span>') + "</td>";
      strategies.forEach((s) => {
        const ridx = entry.cells[s];
        if (ridx === undefined) { html += "<td>–</td>"; return; }
        const r = event.rows[ridx];
        const tooltip = esc(s) + " · win " + pct(r.win_model, 0)
          + " · CI [" + signedPct(r.ci_low) + ", " + signedPct(r.ci_high) + "]"
          + " · n=" + (r.n_analogs === null ? "?" : r.n_analogs)
          + (r.flags && r.flags.length ? " · " + r.flags.join(",") : "");
        html += '<td class="cell' + (isAtm ? " atm" : "") + '" title="' + tooltip + '" '
          + 'style="background:' + cellColor(r.exp_pnl_model) + '" '
          + 'data-eidx="' + eidx + '" data-ridx="' + ridx + '">'
          + '<span class="' + cls(r.exp_pnl_model) + '">' + signedPct(r.exp_pnl_model, 2) + "</span></td>";
      });
      html += "</tr>";
    });
    html += "</tbody></table></div>";
  });
  gridEl.innerHTML = html;

  gridEl.querySelectorAll("td.cell").forEach((td) => {
    td.onclick = () => {
      const event = data.events[Number(td.dataset.eidx)];
      const row = event && event.rows[Number(td.dataset.ridx)];
      if (row) renderRowDetail(row, detailEl);
    };
  });
}

function renderRowDetail(r, detailEl) {
  detailEl.classList.remove("hidden");
  const layer = (title, pnl, win, extra) =>
    '<div class="layer"><h4>' + esc(title) + "</h4>"
    + '<div class="big ' + cls(pnl) + '">' + signedPct(pnl, 2) + "</div>"
    + "<table><tbody>" + extra
    + "<tr><td>win rate</td><td>" + pct(win, 1) + "</td></tr>"
    + "</tbody></table></div>";

  const modelExtra =
    "<tr><td title='Percentiles of the TRADE RETURN, after the driver is pushed "
    + "through the payoff map.'>return p10 / p90</td><td>"
    + signedPct(r.model_p10) + " / " + signedPct(r.model_p90) + "</td></tr>"
    + "<tr><td>driver (" + esc(r.driver_name || "?") + ")</td><td>" + fmt(r.driver_prediction, 2)
    + (r.driver_p10 === null || r.driver_p10 === undefined ? ""
        : " <span class='band'>" + fmt(Math.max(r.driver_p10, 0), 2) + "–"
          + fmt(r.driver_p90, 2) + "</span>")
    + "</td></tr>"
    + "<tr><td title='The band on the driver is nominally 10th-90th (80%). Its "
    + "measured out-of-sample coverage is 79.3%, and since EXP-115 the width "
    + "varies with the prediction.'>driver band</td><td class='muted'>80% nominal, "
    + "<strong>79.3% measured</strong>, conditioned on the prediction</td></tr>"
    /* Both quotes, because they answer different questions and confusing them
       is what made two correct rows look inconsistent. */
    + "<tr><td title='What the market quotes for this print as of TODAY. The "
    + "same for every strategy on the print.'>market implied (today)</td><td>"
    + (r.implied_move === null || r.implied_move === undefined
        ? "– <span class=\'badge\'>no quote</span>" : fmt(r.implied_move, 2) + "%")
    + "</td></tr>"
    + "<tr><td title='The quote at THIS trade&#39;s own entry date — what the "
    + "model actually consumed. Differs between strategies on one print because "
    + "they enter on different days and implied move rises into a print, which "
    + "is the STR-RUNUP thesis.'>… at this trade&#39;s entry</td><td class='muted'>"
    + (r.implied_move_at_entry === null || r.implied_move_at_entry === undefined
        ? "–" : fmt(r.implied_move_at_entry, 2) + "%")
    + "</td></tr>";
  const analogExtra =
    "<tr><td>CI (bootstrap)</td><td>[" + signedPct(r.ci_low) + ", " + signedPct(r.ci_high) + "]</td></tr>"
    + "<tr><td>matched on</td><td class='mono'>"
    + esc(JSON.stringify(((r.analog_buckets || {}).buckets) || {})) + "</td></tr>"
    + "<tr><td>n analogs</td><td>" + (r.n_analogs === null ? "–" : r.n_analogs)
    + (r.analog_widened ? " (widened ×" + r.analog_widened + ")" : "") + "</td></tr>";

  const inputs = Object.keys(r.model_inputs || {});
  const inputRows = inputs.length
    ? "<div class='tablewrap'><table><thead><tr><th>Input</th><th>Value</th><th>What it is</th></tr></thead><tbody>"
      + inputs.map((k) => {
          const v = r.model_inputs[k];
          const note = (((STRATEGIES[r.strategy] || {}).model || {}).features || [])
            .concat((((STRATEGIES[r.strategy] || {}).gate || {}).features || []))
            .find((f) => f.name === k);
          return "<tr><td class='mono'>" + esc(k) + "</td><td class='" + (v === null ? "neg" : "") + "'>"
            + (v === null ? "missing" : fmt(v, 4)) + "</td><td>" + esc((note || {}).note || "") + "</td></tr>";
        }).join("")
      + "</tbody></table></div>"
    : '<span class="badge">no model inputs recorded for this row</span>';

  const versions = Object.entries(r.model_versions || {})
    .map((kv) => esc(kv[0]) + ": <span class='mono'>" + esc(kv[1]) + "</span>")
    .join("<br>") || "–";

  detailEl.innerHTML =
    "<h3>" + esc(r.strategy) + " — " + esc(r.ticker) + " " + esc(r.event_date || "")
    // Strike is deliberately NOT shown. Naming a strike from an older close is
    // where the whole quote/fill gap comes from: 34% of the time spot moves
    // enough overnight to shift the ATM strike, and buying yesterday's strike
    // costs 0.86% on those. Resolve ATM when the order is placed instead.
    + " · expiry " + esc(r.expiry || "–") + " (DTE " + (r.dte_entry === null ? "–" : r.dte_entry) + ")</h3>"
    + '<div class="badge">trade on <b>' + esc(r.entry_date || "–") + "</b>"
    + (r.quote_age_sessions ? " · premium quoted at the " + esc(r.quote_date || "–")
        + " close, <b>" + r.quote_age_sessions + " session"
        + (r.quote_age_sessions === 1 ? "" : "s") + " before</b> your fill"
      : " · premium quoted at that close")
    + " · buy whatever is ATM when you place the order</div>"
    + '<div class="detail-grid">'
    + layer("Model layer", r.exp_pnl_model, r.win_model, modelExtra)
    + layer("Analog layer", r.exp_pnl_analog, r.win_analog, analogExtra)
    + "</div>"
    + '<div class="layer" style="margin-top:10px"><h4>What went into the prediction</h4>'
    + "<div class='badge'>read as of " + esc(r.model_input_as_of || "–") + " · "
    + esc(r.driver_name || "?") + " = " + fmt(r.driver_prediction, 2) + "%</div>"
    + inputRows
    + (r.payoff && r.payoff.slope !== undefined && r.payoff.slope !== null
        ? "<div class='badge' style='margin-top:8px'>payoff line: exit/spot = "
          + fmt(r.payoff.intercept, 4) + " + " + fmt(r.payoff.slope, 4) + " x "
          + esc(r.driver_name || "driver") + " (fitted on " + (r.payoff.n || "?") + " replayed trades)</div>"
        : "")
    + "</div>"
    + '<div class="layer" style="margin-top:10px"><h4>Trade</h4><table><tbody>'
    + "<tr><td>entry → exit</td><td>" + esc(r.entry_date || "–") + " → " + esc(r.exit_date || "–") + "</td></tr>"
    + "<tr><td>entry cost</td><td>" + fmt(r.entry_cost, 2) + " (" + fmt(r.entry_cost_pct, 2) + "% of spot " + fmt(r.spot, 2) + ")</td></tr>"
    + "<tr><td>model-fair premium</td><td>" + fmt(r.model_fair_pct, 2) + "% of spot</td></tr>"
    + "<tr><td>gate</td><td>" + gatePill(r) + (r.gate_threshold !== null && r.gate_threshold !== undefined ? " (threshold " + fmt(r.gate_threshold, 2) + ")" : "") + "</td></tr>"
    + "<tr><td>fill alpha</td><td>" + fmt(r.fill, 2) + "</td></tr>"
    + "<tr><td>model versions</td><td>" + versions + "</td></tr>"
    + "<tr><td>flags</td><td>" + (flagBadges(r) || "–") + "</td></tr>"
    + (r.detail ? "<tr><td>detail</td><td>" + esc(r.detail) + "</td></tr>" : "")
    + "</tbody></table></div>";
}

function renderHistory(data) {
  const el = document.getElementById("x-history");
  if (!data.history || !data.history.length) {
    el.innerHTML = '<span class="badge">no prints in the panel</span>';
    return;
  }
  el.innerHTML = "<table><thead><tr><th>Date</th><th>Implied (oq)</th><th>Implied (ORATS)</th><th>Move</th><th>|Move|</th></tr></thead><tbody>"
    + data.history.map((h) =>
      "<tr><td>" + esc(String(h.date).slice(0, 10)) + "</td>"
      + "<td>" + fmt(h.implied_move, 1) + "%</td>"
      + "<td>" + fmt(h.or_implied, 1) + "%</td>"
      + '<td class="' + cls(h.move) + '">' + signedPct(h.move === null ? null : h.move / 100, 1) + "</td>"
      + "<td>" + fmt(h.abs_move, 1) + "%</td></tr>"
    ).join("") + "</tbody></table>";
}

function renderAnalogs(data) {
  const el = document.getElementById("x-analogs");
  if (!data.analogs || !data.analogs.length) {
    el.innerHTML = '<span class="badge">no engine-replayed trades on this name</span>';
    return;
  }
  el.innerHTML = "<table><thead><tr><th>Event</th><th>Strategy</th><th>Fill</th><th>Return</th></tr></thead><tbody>"
    + data.analogs.map((t) =>
      "<tr><td>" + esc(String(t.event_date).slice(0, 10)) + "</td>"
      + "<td>" + esc(t.strategy) + "</td>"
      + "<td>" + fmt(t.fill_alpha, 2) + "</td>"
      + '<td class="' + cls(t.ret) + '">' + signedPct(t.ret, 1) + "</td></tr>"
    ).join("") + "</tbody></table>";
}

function initExplorerControls() {
  const sel = document.getElementById("x-ticker");
  const tickers = Array.from(new Set(BOARD.rows.map((r) => r.ticker))).sort();
  tickers.forEach((t) => {
    const opt = document.createElement("option");
    opt.value = t; opt.textContent = t;
    sel.appendChild(opt);
  });
  sel.onchange = () => renderExplorer(sel.value);
}

/* --------------------------------------------------------- model explorer */

/* What each input looks like against the outcome, on the model's OWN training
   set. Marginal relationships, not attributions — the caveat travels with the
   data and is rendered, not paraphrased here. */

function corrBar(v) {
  if (v === null || v === undefined) return "";
  const w = Math.min(100, Math.abs(Number(v)) * 200);
  return '<span class="bar' + (v < 0 ? " neg" : "") + '" style="width:' + w.toFixed(0) + 'px"></span>';
}

function renderModelSummary(m) {
  const k = m.kind || {};
  document.getElementById("m-kind").textContent =
    (k.type || "?") + (k.pipeline ? " [" + k.pipeline.join(" → ") + "]" : "");
  const params = Object.entries(k.params || {})
    .map((kv) => kv[0] + "=" + kv[1]).join(", ") || "–";
  const rows = [
    ["role", m.role],
    ["predicts", m.target],
    ["applies to", m.strategy === "*" ? "all strategies" : m.strategy],
    ["model type", k.type || "–"],
    ["parameters", params],
    ["training rows", (m.n_rows || 0).toLocaleString()],
    ["outcome mean / sd", fmt(m.target_mean, 3) + " / " + fmt(m.target_std, 3)],
  ];
  if (m.sampled) {
    rows.push(["sampled", m.sampled.events
      ? m.sampled.events.toLocaleString() + " of " + m.sampled.of.toLocaleString() + " events (seed " + m.sampled.seed + ")"
      : m.sampled.rows.toLocaleString() + " of " + m.sampled.of.toLocaleString() + " rows (seed " + m.sampled.seed + ")"]);
  }
  document.getElementById("m-summary").innerHTML = '<div class="kv">'
    + rows.map((r) => "<div><div class='k'>" + esc(r[0]) + "</div><div class='v'>" + esc(String(r[1])) + "</div></div>").join("")
    + "</div>";
}

function renderModelInputs(m) {
  const el = document.getElementById("m-inputs");
  const inputs = m.inputs || [];
  if (!inputs.length) { el.innerHTML = '<span class="badge">no inputs recorded</span>'; return; }
  el.innerHTML = "<table><thead><tr><th>Input</th><th>Spearman</th><th></th><th>|distance|</th><th>Pearson</th>"
    + "<th>decile range</th><th>coverage</th><th>n</th><th>What it is</th></tr></thead><tbody>"
    + inputs.map((f) => {
        if (!f.usable) {
          return "<tr><td class='mono'>" + esc(f.name) + "</td><td colspan='7'>"
            + esc(f.reason || "not usable") + "</td><td>" + esc(f.note || "") + "</td></tr>";
        }
        return "<tr class='clickable' data-feature='" + esc(f.name) + "'>"
          + "<td class='mono'>" + esc(f.name) + "</td>"
          + '<td class="' + cls(f.spearman) + '">' + fmt(f.spearman, 3)
          + (f.monotone === false && Math.abs(f.decile_range)
                > 2 * Math.abs(f.decile_spread || 0)
              ? ' <span class="pill warn">V</span>' : "") + "</td>"
          + "<td>" + corrBar(f.spearman) + "</td>"
          + "<td>" + fmt(f.magnitude_spearman, 3) + "</td>"
          + "<td>" + fmt(f.pearson, 3) + "</td>"
          + "<td>" + fmt(f.decile_range, 3) + "</td>"
          + "<td>" + pct(f.coverage, 0) + "</td>"
          + "<td>" + (f.n || 0).toLocaleString() + "</td>"
          + "<td>" + esc(f.note || "") + "</td></tr>";
      }).join("")
    + "</tbody></table>";
  el.querySelectorAll("tr[data-feature]").forEach((tr) => {
    tr.onclick = () => {
      document.getElementById("m-feature").value = tr.dataset.feature;
      renderShape(m, tr.dataset.feature);
    };
  });
}

/* Scatter, with the straight line a linear model would fit and the decile
   means over the top. The two together are the point: where the line is flat
   and the decile curve is a V, the relationship is real and a linear reading
   cannot see it. All three come pre-computed in the bundle. */
function scatterSvg(f, m) {
  const pts = f.scatter || [];
  if (!pts.length) return "";
  const W = 560, H = 300, PAD = 44;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const xlo = Math.min.apply(null, xs), xhi = Math.max.apply(null, xs);
  const ylo = Math.min.apply(null, ys), yhi = Math.max.apply(null, ys);
  const xr = (xhi - xlo) || 1, yr = (yhi - ylo) || 1;
  const sx = (v) => PAD + ((v - xlo) / xr) * (W - PAD - 12);
  const sy = (v) => H - PAD - ((v - ylo) / yr) * (H - PAD - 14);

  let svg = '<svg viewBox="0 0 ' + W + " " + H + '" width="100%" style="max-width:' + W + 'px">';
  svg += '<rect x="' + PAD + '" y="10" width="' + (W - PAD - 12) + '" height="' + (H - PAD - 10)
       + '" fill="none" stroke="#30363d"/>';
  pts.forEach((p) => {
    svg += '<circle cx="' + sx(p[0]).toFixed(1) + '" cy="' + sy(p[1]).toFixed(1)
         + '" r="1.6" fill="#58a6ff" fill-opacity="0.35"/>';
  });
  if (f.ols) {
    const y1 = f.ols.intercept + f.ols.slope * xlo;
    const y2 = f.ols.intercept + f.ols.slope * xhi;
    svg += '<line x1="' + sx(xlo).toFixed(1) + '" y1="' + sy(y1).toFixed(1)
         + '" x2="' + sx(xhi).toFixed(1) + '" y2="' + sy(y2).toFixed(1)
         + '" stroke="#f85149" stroke-width="2"/>';
  }
  if (f.deciles && f.deciles.length) {
    const path = f.deciles.map((d, i) => (i ? "L" : "M")
      + sx((d.x_lo + d.x_hi) / 2).toFixed(1) + " " + sy(d.y_mean).toFixed(1)).join(" ");
    svg += '<path d="' + path + '" fill="none" stroke="#3fb950" stroke-width="2.5"/>';
    f.deciles.forEach((d) => {
      svg += '<circle cx="' + sx((d.x_lo + d.x_hi) / 2).toFixed(1) + '" cy="' + sy(d.y_mean).toFixed(1)
           + '" r="3.5" fill="#3fb950"/>';
    });
  }
  svg += '<text x="' + PAD + '" y="' + (H - 14) + '" fill="#8b949e" font-size="11">'
       + esc(String(fmt(xlo, 2))) + "</text>";
  svg += '<text x="' + (W - 60) + '" y="' + (H - 14) + '" fill="#8b949e" font-size="11">'
       + esc(String(fmt(xhi, 2))) + "</text>";
  svg += '<text x="6" y="18" fill="#8b949e" font-size="11">' + esc(String(fmt(yhi, 2))) + "</text>";
  svg += '<text x="6" y="' + (H - PAD) + '" fill="#8b949e" font-size="11">'
       + esc(String(fmt(ylo, 2))) + "</text>";
  svg += '<text x="' + (W / 2 - 40) + '" y="' + (H - 2) + '" fill="#8b949e" font-size="11">'
       + esc(f.name) + "</text>";
  svg += "</svg>";
  svg += "<div class='badge' style='margin-top:6px'>"
       + "<span style='color:#f85149'>—</span> the straight line a linear model fits &nbsp; "
       + "<span style='color:#3fb950'>—</span> mean " + esc(m.target) + " per decile &nbsp; "
       + "<span style='color:#58a6ff'>·</span> " + pts.length + " sampled rows of "
       + (f.n || 0).toLocaleString() + "</div>";
  return svg;
}

function shapeVerdict(f, m) {
  if (!f.deciles || f.monotone !== false) return "";
  const ratio = Math.abs(f.decile_range) / Math.max(Math.abs(f.decile_spread), 1e-9);
  if (ratio < 2) return "";
  return "<div class='flag-line'>⚑ Non-monotone: mean " + esc(m.target) + " spans "
    + fmt(f.decile_range, 2) + " across the deciles but only " + fmt(f.decile_spread, 2)
    + " end to end, so the correlations above (" + fmt(f.spearman, 3)
    + ") understate it. Distance from the middle reads "
    + fmt(f.magnitude_spearman, 3) + ".</div>";
}

function renderShape(m, featureName) {
  const f = (m.inputs || []).find((x) => x.name === featureName);
  const el = document.getElementById("m-shape");
  const note = document.getElementById("m-shape-note");
  if (!f || !f.deciles || !f.deciles.length) {
    note.textContent = "";
    el.innerHTML = '<span class="badge">no decile shape for this input</span>';
    return;
  }
  note.textContent = f.note || "";
  const ys = f.deciles.map((d) => d.y_mean);
  const head = shapeVerdict(f, m) + scatterSvg(f, m);
  const lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
  const span = (hi - lo) || 1;
  el.innerHTML = head + "<table style='margin-top:10px'><thead><tr><th>decile</th><th>" + esc(featureName) + " range</th>"
    + "<th>mean " + esc(m.target) + "</th><th></th><th>n</th></tr></thead><tbody>"
    + f.deciles.map((d) =>
        "<tr><td>" + d.bin + "</td>"
        + "<td class='mono'>" + fmt(d.x_lo, 3) + " … " + fmt(d.x_hi, 3) + "</td>"
        + '<td class="' + cls(d.y_mean) + '">' + fmt(d.y_mean, 3) + "</td>"
        + "<td>" + '<span class="bar' + (d.y_mean < 0 ? " neg" : "") + '" style="width:'
        + (((d.y_mean - lo) / span) * 160 + 4).toFixed(0) + 'px"></span>' + "</td>"
        + "<td>" + d.n.toLocaleString() + "</td></tr>").join("")
    + "</tbody></table>";
}

function renderModelExplorer(id) {
  const m = MODELS[id];
  if (!m) return;
  if (!m.available) {
    document.getElementById("m-summary").innerHTML =
      '<span class="badge">' + esc(m.reason || "not available") + "</span>";
    document.getElementById("m-inputs").innerHTML = "";
    document.getElementById("m-shape").innerHTML = "";
    return;
  }
  renderModelSummary(m);
  renderModelInputs(m);
  const sel = document.getElementById("m-feature");
  sel.innerHTML = "";
  (m.inputs || []).filter((f) => f.usable).forEach((f) => {
    const o = document.createElement("option");
    o.value = f.name; o.textContent = f.name;
    sel.appendChild(o);
  });
  sel.onchange = () => renderShape(m, sel.value);
  if (sel.options.length) renderShape(m, sel.options[0].value);
}

/* ~2 MB of per-input evidence, fetched only when someone asks for it. */
let modelsRequested = false;

function loadModels(then) {
  if (window.MODELS) { then(); return; }
  if (modelsRequested) return;
  modelsRequested = true;
  const script = document.createElement("script");
  script.src = "data/models.js";
  script.onload = () => then();
  script.onerror = () => {
    document.getElementById("m-summary").innerHTML =
      '<span class="badge">model evidence not in this bundle — build it with '
      + "`python3 -m engine.dashboard.model_evidence`</span>";
  };
  document.head.appendChild(script);
}

function initModelExplorer() {
  loadModels(() => {
    MODELS_META = window.MODELS || {};
    MODELS = MODELS_META.models || {};
    buildModelExplorer();
  });
}

function buildModelExplorer() {
  const sel = document.getElementById("m-model");
  if (sel.options.length) return;  /* already built */
  const ids = Object.keys(MODELS).sort();
  document.getElementById("m-caveat").textContent = MODELS_META.caveat || "";
  if (!ids.length) {
    document.getElementById("m-summary").innerHTML =
      '<span class="badge">no model evidence in this snapshot — build it with '
      + '`python3 -m engine.dashboard.model_evidence`</span>';
    return;
  }
  ids.forEach((id) => {
    const o = document.createElement("option");
    o.value = id; o.textContent = id + "  (" + (MODELS[id].role || "") + ")";
    sel.appendChild(o);
  });
  sel.onchange = () => renderModelExplorer(sel.value);
  renderModelExplorer(ids[0]);
}

/* ------------------------------------------------------------ derivation */

/* How a number is made. Reads `strategies.json` (the shape: structure, driver,
   champion, features) — the per-row VALUES live on each row as `model_inputs`
   and are rendered by inputsTable() in the explorer's row detail. */

function metricRow(k, v) {
  const labels = {
    n: "training rows", r: "OOS correlation", mae: "OOS mean abs error (pp)",
    rmse: "OOS RMSE (pp)", bias: "OOS bias (pp)", decile_spread: "top-vs-bottom decile spread (pp)",
    oos_years: "out-of-sample years",
  };
  return "<tr><td>" + esc(labels[k] || k) + "</td><td>" + fmt(v, k === "n" || k === "oos_years" ? 0 : 3) + "</td></tr>";
}

function modelPanel(title, m) {
  if (!m) return '<div class="layer"><h4>' + esc(title) + "</h4><span class='badge'>none registered</span></div>";
  const metrics = Object.keys(m.oos || {}).map((k) => metricRow(k, m.oos[k])).join("");
  return '<div class="layer"><h4>' + esc(title) + "</h4>"
    + "<table><tbody>"
    + "<tr><td>registry id</td><td class='mono'>" + esc(m.id) + "</td></tr>"
    + "<tr><td>predicts</td><td>" + esc(m.target || "–") + "</td></tr>"
    + "<tr><td>training</td><td>" + esc(m.train_window || "–") + "</td></tr>"
    + "<tr><td>promoted</td><td>" + esc(m.promoted || "–") + "</td></tr>"
    + (m.threshold !== null && m.threshold !== undefined
        ? "<tr><td>gate threshold</td><td>" + fmt(m.threshold, 4) + "</td></tr>" : "")
    + "<tr><td>artifact</td><td class='mono'>" + esc(m.artifact_sha256) + "…</td></tr>"
    + metrics
    + "</tbody></table>"
    + "<div class='badge' style='margin-top:8px'>" + (m.features || []).length + " inputs</div>"
    + "<div class='tablewrap' style='margin-top:6px'><table><thead><tr><th>Feature</th><th>What it is</th></tr></thead><tbody>"
    + (m.features || []).map((f) =>
        "<tr><td class='mono'>" + esc(f.name) + "</td><td>" + esc(f.note || "—") + "</td></tr>").join("")
    + "</tbody></table></div></div>";
}

/* The values a real row actually fed the model. The tab explains the shape;
   without an example beside it the reader still cannot see what went in. */
function workedExample(name) {
  const el = document.getElementById("d-example");
  const sel = document.getElementById("d-row");
  const rowId = sel.value;
  if (!rowId) {
    el.innerHTML = '<span class="badge">no scored row for this strategy in this snapshot</span>';
    return;
  }
  const board = BOARD.rows.find((r) => r.row_id === rowId);
  if (!board) { el.innerHTML = ""; return; }

  loadTickerData(board.ticker, (data) => {
    let full = null;
    (data ? data.events : []).forEach((ev) => (ev.rows || []).forEach((r) => {
      if (r.strategy === board.strategy && r.event_date === board.event_date
          && (r.strike_offset === null || r.strike_offset === undefined)) full = r;
    }));
    if (!full || !full.model_inputs) {
      el.innerHTML = '<span class="badge">no recorded inputs for that row</span>';
      return;
    }
    const notes = ((STRATEGIES[name] || {}).model || {}).features || [];
    const keys = Object.keys(full.model_inputs);
    el.innerHTML =
      "<div class='badge'>" + esc(full.ticker) + " · " + esc(full.event_date) + " · inputs read as of "
      + esc(full.model_input_as_of || "–") + " → " + esc(full.driver_name || "?") + " = "
      + fmt(full.driver_prediction, 2) + "%</div>"
      + "<div class='tablewrap' style='margin-top:8px'><table><thead><tr><th>Input</th><th>Value</th>"
      + "<th>What it is</th></tr></thead><tbody>"
      + keys.map((k) => {
          const v = full.model_inputs[k];
          const note = notes.find((f) => f.name === k) || {};
          return "<tr><td class='mono'>" + esc(k) + "</td>"
            + '<td class="' + (v === null ? "neg" : "") + '">'
            + (v === null ? "missing" : fmt(v, 4)) + "</td>"
            + "<td>" + esc(note.note || "") + "</td></tr>";
        }).join("")
      + "</tbody></table></div>";
  });
}

function initWorkedExample(name) {
  const sel = document.getElementById("d-row");
  sel.innerHTML = "";
  BOARD.rows
    .filter((r) => r.strategy === name && r.driver_prediction !== null
                   && r.driver_prediction !== undefined)
    .slice(0, 200)
    .forEach((r) => {
      const o = document.createElement("option");
      o.value = r.row_id;
      o.textContent = r.ticker + " " + r.event_date;
      sel.appendChild(o);
    });
  sel.onchange = () => workedExample(name);
  workedExample(name);
}

function renderDerivation(name) {
  const s = STRATEGIES[name];
  const el = document.getElementById("d-body");
  if (!s) { el.innerHTML = '<span class="badge">no such strategy in this snapshot</span>'; return; }

  const legs = (s.structure.legs || [])
    .map((l) => l.side + " " + (l.qty === 1 ? "" : l.qty + "x ") + l.right).join(" + ") || "–";

  let html = "";
  if (!s.enabled) {
    html += '<div class="flag-line">⚑ ' + esc(s.disabled_reason || "disabled") + "</div>";
  }
  html += '<div class="layer"><h4>The trade</h4><table><tbody>'
    + "<tr><td>structure</td><td>" + esc(legs) + "</td></tr>"
    + "<tr><td>opened</td><td>" + esc(s.structure.entry_note) + "</td></tr>"
    + "<tr><td>closed</td><td>" + esc(s.structure.exit_note) + "</td></tr>"
    + "<tr><td>the model predicts</td><td>" + esc(s.driver_note || s.driver || "–") + "</td></tr>"
    + "</tbody></table></div>";

  html += '<div class="layer" style="margin-top:10px"><h4>Two estimates, never averaged</h4>'
    + "<p>" + esc(s.layers.model) + "</p><p>" + esc(s.layers.analog) + "</p></div>";

  html += '<div class="detail-grid" style="margin-top:10px">'
    + modelPanel("Champion model", s.model)
    + modelPanel("Gate", s.gate)
    + "</div>";
  html += '<div class="layer" style="margin-top:10px"><h4>A worked example — the values a real row fed the model</h4>'
    + '<div id="d-example"></div></div>';
  el.innerHTML = html;
  initWorkedExample(name);
}

function initDerivation() {
  const sel = document.getElementById("d-strategy");
  const names = Object.keys(STRATEGIES).sort();
  names.forEach((n) => {
    const o = document.createElement("option");
    o.value = n; o.textContent = n;
    sel.appendChild(o);
  });
  sel.onchange = () => renderDerivation(sel.value);
  if (names.length) renderDerivation(names[0]);
}

/* ---------------------------------------------------------------- health */

/* The hypothetical book. States are kept distinct on purpose: an open trade
   and one that could not be priced are both "not settled", and collapsing them
   would let the book quietly drop the trades it failed to follow. */
const BK_STATE = {
  settled: ["settled", "the event happened and the replay priced it"],
  open: ["open", "the event has not happened yet — capital committed"],
  awaiting_exit: ["awaiting exit", "printed, but its exit chain is not published yet"],
  unresolvable: ["unresolvable", "the exit chain should exist and it still cannot be priced"],
};

function renderBook() {
  const kv = document.getElementById("bk-kv");
  const tbl = document.getElementById("bk-table");
  if (!kv || !tbl) return;
  const s = BOOK.summary || {};
  if (!BOOK.available || !(BOOK.rows || []).length) {
    kv.innerHTML = "<div class='badge'>No recommendations in the ledger yet — "
      + "no row has gate_pass=true.</div>";
    tbl.innerHTML = "";
    return;
  }
  const st = s.by_state || {};
  const money = (v) => (v === null || v === undefined || isNaN(v) ? "–"
    : (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 }));
  const parts = [
    ["recommendations", (s.n_recommended || 0) + " distinct trades"],
    ["contracts each", BOOK.contracts],
    ["capital committed", money(s.capital_committed)],
    ["settled", (s.n_settled || 0) + " of " + (s.n_recommended || 0)],
  ];
  if (s.n_settled) {
    parts.push(["P&L (settled)", money(s.pnl)]);
    parts.push(["return on capital", (100 * s.return_on_capital).toFixed(2) + "%"]);
    parts.push(["win rate", (100 * s.win_rate).toFixed(1) + "%"]);
  }
  Object.keys(st).forEach((k) => parts.push([(BK_STATE[k] || [k])[0], st[k]]));
  kv.innerHTML = parts
    .map((p) => esc(p[0]) + ": <span class='mono'>" + esc(String(p[1])) + "</span>")
    .join("<br>");

  const rows = (BOOK.rows || []).slice().sort((a, b) =>
    String(a.event_date).localeCompare(String(b.event_date)));
  const head = ["recommended", "ticker", "strategy", "event", "state",
                "premium", "return", "P&L"];
  let html = "<table><thead><tr>" + head.map((h) => "<th>" + esc(h) + "</th>").join("")
    + "</tr></thead><tbody>";
  rows.forEach((r) => {
    const meta = BK_STATE[r.state] || [r.state, ""];
    const ret = r.realized_pnl === null || r.realized_pnl === undefined
      ? "–" : (100 * r.realized_pnl).toFixed(2) + "%";
    html += "<tr>"
      + "<td class='mono'>" + esc(String(r.as_of || "").slice(0, 10)) + "</td>"
      + "<td>" + esc(r.ticker) + "</td>"
      + "<td>" + esc(r.strategy) + "</td>"
      + "<td class='mono'>" + esc(String(r.event_date || "").slice(0, 10)) + "</td>"
      + "<td title='" + esc(meta[1]) + "'>" + esc(meta[0]) + "</td>"
      + "<td class='mono'>" + (r.entry_cost === null || r.entry_cost === undefined
        ? "–" : fmt(r.entry_cost, 2)) + "</td>"
      + "<td class='mono'>" + esc(ret) + "</td>"
      + "<td class='mono'>" + esc(money(r.pnl)) + "</td>"
      + "</tr>";
  });
  tbl.innerHTML = html + "</tbody></table>";
}

function renderHealth() {
  const kv = document.getElementById("h-kv");
  const fresh = META.freshness || {};
  const quota = META.quota || {};
  const items = [
    ["as of", META.as_of || "–"],
    ["generated", (META.generated_at || "").replace("T", " ").slice(0, 19)],
    ["events × strategies", (META.n_events || 0) + " × " + (META.n_rows || 0) + " rows"],
    ["snapshot", '<span class="mono">' + esc((META.snapshot_hash || "–").slice(0, 16)) + "…</span>"],
    ["fill alpha", fmt(META.fill_alpha, 2)],
    ["ORATS quota remaining", quota.remaining === null || quota.remaining === undefined ? "–" : Number(quota.remaining).toLocaleString() + " (reserve " + (quota.reserve_floor || "–") + ")"],
    ["daily data through", fresh.daily_market_last_date || "–"],
    ["late backfills", (META.late_as_ofs || []).join(", ") || "none"],
  ];
  kv.innerHTML = items.map((kv2) =>
    "<div><div class='k'>" + kv2[0] + "</div><div class='v'>" + kv2[1] + "</div></div>"
  ).join("");

  const models = document.getElementById("h-models");
  const versions = META.model_versions || {};
  const names = Object.keys(versions);
  models.innerHTML = names.length
    ? "<table><thead><tr><th>Role</th><th>Champion</th></tr></thead><tbody>"
      + names.map((n) => "<tr><td>" + esc(n) + "</td><td class='mono'>" + esc(versions[n]) + "</td></tr>").join("")
      + "</tbody></table>"
    : '<span class="badge">no champions registered</span>';

  const size = HEALTH.size_model || {};
  const sizeEl = document.getElementById("h-sizemae");
  if (!size.available) {
    sizeEl.innerHTML = '<span class="badge">' + esc(size.reason || "no live MAE yet") + "</span>";
  } else {
    let html = '<div class="kv">'
      + "<div><div class='k'>model MAE</div><div class='v'>" + fmt(size.model_mae_pp, 2) + "pp (n=" + size.n + ")</div></div>"
      + "<div><div class='k'>implied baseline MAE</div><div class='v'>"
      + (size.implied_baseline_mae_pp === undefined ? "–" : fmt(size.implied_baseline_mae_pp, 2) + "pp") + "</div></div></div>";
    if (size.series && size.series.length) {
      html += "<div class='tablewrap' style='margin-top:10px'><table><thead><tr><th>as of</th><th>n</th><th>model</th><th>baseline</th></tr></thead><tbody>"
        + size.series.map((s) =>
          "<tr><td>" + esc(s.as_of) + "</td><td>" + s.n + "</td><td>" + fmt(s.model_mae_pp, 2) + "pp</td><td>"
          + (s.implied_baseline_mae_pp === undefined ? "–" : fmt(s.implied_baseline_mae_pp, 2) + "pp") + "</td></tr>"
        ).join("") + "</tbody></table></div>";
    }
    sizeEl.innerHTML = html;
  }

  const cal = document.getElementById("h-calibration");
  const ledger = HEALTH.ledger || {};
  if (!ledger || ledger.available === false) {
    cal.innerHTML = '<span class="badge">ledger calibration not available yet (no scored outcomes)</span>';
  } else {
    const rows = Object.entries(ledger.per_strategy || {}).map((kv) => {
      const b = kv[1] || {};
      if (!b.available) return "<tr><td>" + esc(kv[0]) + "</td><td colspan='4'>" + esc(b.reason || "n/a") + "</td></tr>";
      return "<tr><td>" + esc(kv[0]) + "</td><td>" + b.n + "</td><td>" + fmt(b.brier_skill, 3)
        + "</td><td>" + pct(b.base_rate, 0) + "</td><td>"
        + fmt(b.predicted_mean_pnl === undefined ? null : b.predicted_mean_pnl * 100, 2) + "% vs "
        + fmt(b.realized_mean_pnl === undefined ? null : b.realized_mean_pnl * 100, 2) + "%</td></tr>";
    });
    cal.innerHTML = "<table><thead><tr><th>Strategy</th><th>n</th><th>Brier skill</th><th>base rate</th><th>pred vs real PnL</th></tr></thead><tbody>"
      + (rows.join("") || "<tr><td colspan='5'>no per-strategy rows</td></tr>") + "</tbody></table>"
      + "<div class='badge' style='margin-top:8px'>predictions " + (ledger.n_predictions || 0)
      + " · scored " + (ledger.n_scored || 0) + " · drift flagged: " + ((HEALTH.calibration_drift || {}).flagged ? "YES" : "no") + "</div>";
  }

  const sc = document.getElementById("h-selfcheck");
  const last = HEALTH.last_selfcheck;
  if (!last) {
    sc.innerHTML = '<span class="badge">no self-check has run yet</span>';
  } else {
    sc.innerHTML = '<div class="kv">'
      + "<div><div class='k'>status</div><div class='v'>" + (last.ok ? '<span class="pill pass">GREEN</span>' : '<span class="pill fail">RED</span>') + "</div></div>"
      + "<div><div class='k'>rows re-scored</div><div class='v'>" + last.n_checked + " / " + last.n_board_rows + "</div></div>"
      + "<div><div class='k'>mismatches</div><div class='v'>" + (last.mismatches ? last.mismatches.length : 0) + "</div></div>"
      + "<div><div class='k'>snapshot match</div><div class='v'>" + (last.snapshot_ok ? "yes" : "DRIFT") + "</div></div>"
      + "</div><div class='badge' style='margin-top:8px'>" + esc(last.detail || "") + "</div>";
  }

  const flagsEl = document.getElementById("h-flags");
  const flags = FLAGS.flags || [];
  flagsEl.innerHTML = flags.length
    ? flags.map((f) => "<div>⚑ " + esc(flagText(f)) + "</div>").join("")
    : '<span class="badge">no flags raised on this snapshot</span>';
}

/* ---------------------------------------------------------------- init */

function init() {
  document.querySelectorAll(".tab[data-tab]").forEach((t) => {
    t.onclick = () => switchTab(t.dataset.tab);
  });
  document.querySelectorAll(".tab.area").forEach((t) => {
    t.onclick = () => switchArea(t.dataset.area);
  });

  const metaBits = [];
  if (META.as_of) metaBits.push("as of " + META.as_of);
  if (META.n_events) metaBits.push(META.n_events + " events");
  if (META.n_rows) metaBits.push(META.n_rows + " scores");
  if (META.generated_at) metaBits.push("generated " + META.generated_at.slice(0, 19).replace("T", " "));
  document.getElementById("meta-line").textContent =
    metaBits.length ? metaBits.join(" · ") : "No snapshot yet — run `python3 -m engine.dashboard.nightly`.";
  document.getElementById("foot-line").textContent =
    "render v" + (META.render_version || "?") + " · snapshot " + (META.snapshot_hash || "–").slice(0, 16)
    + " · UI renders ScoreResult fields only — it computes nothing";

  renderFlagsBanner();
  initBoardControls();
  renderBoard();
  initExplorerControls();
  initDerivation();
  renderHealth();
  renderBook();
}

init();
