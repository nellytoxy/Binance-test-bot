"""
Dashboard server for Liquidity Sweep Bot
Run: python dashboard.py
Visit: http://localhost:5000
"""

import json, threading
from flask import Flask, jsonify, request, render_template_string
from trading_bot import BotConfig, LiquiditySweepBot

app = Flask(__name__)

# ─── Paths — use /tmp on Railway (ephemeral but works) ───────────────────────
import os
CONFIG_PATH = os.environ.get("CONFIG_PATH", "/tmp/config.json")
TRADES_PATH = os.environ.get("TRADES_PATH", "/tmp/trades.json")
PORT = int(os.environ.get("PORT", 5000))

# ─── Global bot instance ─────────────────────────────────────────────────────
bot: LiquiditySweepBot = None
bot_thread = None


def load_config() -> BotConfig:
    # Start with env-var defaults (Railway injects these)
    cfg = BotConfig()
    try:
        with open(CONFIG_PATH) as f:
            d = json.load(f)
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        # Re-run post_init so env vars still win over blank saved values
        cfg.__post_init__()
    except:
        pass
    return cfg


def save_config(data: dict):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except:
        pass


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/state")
def api_state():
    global bot
    if bot is None:
        return jsonify({"status": "NOT_STARTED", "capital": 0, "stats": {}, "open_trades": [], "recent_trades": [], "scanned_pairs": []})
    return jsonify(bot.get_dashboard_state())


@app.route("/api/start", methods=["POST"])
def api_start():
    global bot, bot_thread
    if bot and bot.running:
        return jsonify({"ok": False, "msg": "Already running"})
    cfg = load_config()
    data = request.json or {}
    if "live_mode" in data:
        cfg.live_mode = data["live_mode"]
    bot = LiquiditySweepBot(cfg, trades_path=TRADES_PATH)
    bot.start()
    return jsonify({"ok": True, "msg": f"Bot started in {'LIVE' if cfg.live_mode else 'TESTNET'} mode"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global bot
    if bot:
        bot.stop()
    return jsonify({"ok": True})


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        data = request.json or {}
        save_config(data)
        return jsonify({"ok": True})
    try:
        with open(CONFIG_PATH) as f:
            return jsonify(json.load(f))
    except:
        from dataclasses import asdict
        return jsonify(asdict(BotConfig()))


@app.route("/api/close_trade", methods=["POST"])
def api_close_trade():
    global bot
    if not bot:
        return jsonify({"ok": False})
    data = request.json or {}
    symbol = data.get("symbol")
    for trade in bot.journal.open_trades():
        if trade.symbol == symbol:
            bot.executor.close_trade(trade, "MANUAL")
            trade.status = "MANUAL"
            bot.journal.update(trade)
            return jsonify({"ok": True})
    return jsonify({"ok": False, "msg": "Trade not found"})


# ─── Dashboard HTML ───────────────────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Liquidity Sweep Bot</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #050a0f;
    --panel: #0b1520;
    --border: #0d2a3a;
    --accent: #00d4ff;
    --green: #00ff88;
    --red: #ff3366;
    --yellow: #ffcc00;
    --dim: #3a5566;
    --text: #c8e0ec;
    --mono: 'Share Tech Mono', monospace;
    --head: 'Rajdhani', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--head);
    min-height: 100vh;
    background-image:
      radial-gradient(ellipse at 20% 10%, rgba(0,80,120,0.15) 0%, transparent 60%),
      radial-gradient(ellipse at 80% 90%, rgba(0,30,60,0.2) 0%, transparent 60%);
  }

  /* ─── Header ─── */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 28px;
    border-bottom: 1px solid var(--border);
    background: rgba(5,15,25,0.9);
    position: sticky; top: 0; z-index: 99;
    backdrop-filter: blur(8px);
  }
  .logo { font-size: 22px; font-weight: 700; letter-spacing: 3px; color: var(--accent); }
  .logo span { color: var(--green); }
  .mode-badge {
    font-family: var(--mono);
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 3px;
    letter-spacing: 2px;
    border: 1px solid;
  }
  .mode-live { color: var(--red); border-color: var(--red); background: rgba(255,51,102,0.1); }
  .mode-test { color: var(--yellow); border-color: var(--yellow); background: rgba(255,204,0,0.1); }
  .header-right { display: flex; gap: 10px; align-items: center; }

  /* ─── Buttons ─── */
  .btn {
    font-family: var(--head);
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 1.5px;
    padding: 8px 20px;
    border: 1px solid;
    cursor: pointer;
    border-radius: 3px;
    transition: all 0.2s;
    text-transform: uppercase;
  }
  .btn-start { color: var(--green); border-color: var(--green); background: rgba(0,255,136,0.08); }
  .btn-start:hover { background: rgba(0,255,136,0.2); }
  .btn-stop  { color: var(--red);   border-color: var(--red);   background: rgba(255,51,102,0.08); }
  .btn-stop:hover  { background: rgba(255,51,102,0.2); }
  .btn-sm {
    font-family: var(--mono);
    font-size: 10px;
    padding: 4px 10px;
    color: var(--red);
    border: 1px solid var(--red);
    border-radius: 2px;
    cursor: pointer;
    background: transparent;
    letter-spacing: 1px;
  }
  .btn-sm:hover { background: rgba(255,51,102,0.15); }
  .toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-family: var(--mono);
    color: var(--dim);
    cursor: pointer;
  }
  .toggle input { display: none; }
  .toggle-track {
    width: 40px; height: 20px;
    background: var(--border);
    border-radius: 10px;
    position: relative;
    transition: background 0.2s;
    border: 1px solid var(--dim);
  }
  .toggle input:checked + .toggle-track { background: rgba(255,51,102,0.3); border-color: var(--red); }
  .toggle-thumb {
    position: absolute;
    left: 2px; top: 2px;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: var(--dim);
    transition: left 0.2s, background 0.2s;
  }
  .toggle input:checked ~ .toggle-track .toggle-thumb,
  .toggle input:checked + .toggle-track .toggle-thumb {
    left: 22px;
    background: var(--red);
  }

  /* ─── Grid ─── */
  .main { padding: 20px 28px; }
  .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 20px; }
  .grid-3 { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 14px; margin-bottom: 20px; }

  /* ─── Cards ─── */
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 18px 20px;
  }
  .card-title {
    font-size: 10px;
    font-family: var(--mono);
    letter-spacing: 2px;
    color: var(--dim);
    text-transform: uppercase;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .card-title::before {
    content: '';
    display: inline-block;
    width: 4px; height: 4px;
    background: var(--accent);
    border-radius: 50%;
  }
  .metric-val {
    font-size: 32px;
    font-weight: 700;
    font-family: var(--mono);
    color: var(--accent);
    line-height: 1;
  }
  .metric-sub {
    font-size: 11px;
    font-family: var(--mono);
    color: var(--dim);
    margin-top: 4px;
  }

  /* ─── Status indicator ─── */
  .status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
  }
  .status-SCANNING { background: var(--yellow); }
  .status-WATCHING { background: var(--green); }
  .status-IDLE, .status-STOPPED { background: var(--dim); animation: none; }
  .status-ERROR { background: var(--red); }
  .status-MAX_TRADES { background: var(--accent); }
  @keyframes pulse {
    0%,100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  /* ─── Table ─── */
  .tbl { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 12px; }
  .tbl th {
    text-align: left;
    padding: 8px 10px;
    color: var(--dim);
    border-bottom: 1px solid var(--border);
    letter-spacing: 1px;
    font-size: 10px;
    text-transform: uppercase;
  }
  .tbl td { padding: 9px 10px; border-bottom: 1px solid rgba(13,42,58,0.5); }
  .tbl tr:last-child td { border-bottom: none; }
  .tbl tr:hover td { background: rgba(0,212,255,0.03); }
  .long  { color: var(--green); }
  .short { color: var(--red); }
  .pnl-pos { color: var(--green); }
  .pnl-neg { color: var(--red); }
  .status-OPEN { color: var(--accent); }
  .status-TP   { color: var(--green); }
  .status-SL   { color: var(--red); }
  .status-MANUAL { color: var(--dim); }

  /* ─── Scan ticker ─── */
  .pairs-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
  }
  .pair-tag {
    font-family: var(--mono);
    font-size: 11px;
    padding: 3px 8px;
    border: 1px solid var(--border);
    border-radius: 3px;
    color: var(--dim);
    background: rgba(0,0,0,0.2);
    transition: color 0.3s, border-color 0.3s;
  }
  .pair-tag.active { color: var(--accent); border-color: var(--accent); }

  /* ─── Config panel ─── */
  .cfg-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .field label {
    display: block;
    font-size: 10px;
    font-family: var(--mono);
    color: var(--dim);
    margin-bottom: 4px;
    letter-spacing: 1px;
  }
  .field input {
    width: 100%;
    background: #070f17;
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    padding: 7px 10px;
    border-radius: 3px;
    outline: none;
    transition: border-color 0.2s;
  }
  .field input:focus { border-color: var(--accent); }
  .save-btn {
    margin-top: 14px;
    width: 100%;
    padding: 10px;
    background: rgba(0,212,255,0.1);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: var(--head);
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 2px;
    cursor: pointer;
    border-radius: 3px;
    text-transform: uppercase;
  }
  .save-btn:hover { background: rgba(0,212,255,0.2); }

  /* ─── Scrollable table containers ─── */
  .tbl-wrap { max-height: 320px; overflow-y: auto; }
  .tbl-wrap::-webkit-scrollbar { width: 4px; }
  .tbl-wrap::-webkit-scrollbar-track { background: transparent; }
  .tbl-wrap::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  /* ─── Win rate bar ─── */
  .wr-bar { height: 6px; background: var(--border); border-radius: 3px; margin-top: 8px; overflow: hidden; }
  .wr-fill { height: 100%; background: linear-gradient(90deg, var(--green), var(--accent)); border-radius: 3px; transition: width 0.6s; }

  /* ─── Toast ─── */
  .toast {
    position: fixed;
    bottom: 24px; right: 24px;
    background: var(--panel);
    border: 1px solid var(--accent);
    color: var(--text);
    font-family: var(--mono);
    font-size: 12px;
    padding: 12px 20px;
    border-radius: 4px;
    transform: translateY(60px);
    opacity: 0;
    transition: all 0.3s;
    z-index: 200;
    max-width: 320px;
  }
  .toast.show { transform: translateY(0); opacity: 1; }

  @media(max-width:900px) {
    .grid-4 { grid-template-columns: 1fr 1fr; }
    .grid-3 { grid-template-columns: 1fr; }
    .grid-2 { grid-template-columns: 1fr; }
    .cfg-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<div class="header">
  <div class="logo">SWEEP<span>BOT</span></div>
  <div class="header-right">
    <label class="toggle" id="modeToggle" title="Toggle Live/Testnet">
      <input type="checkbox" id="liveModeCheck" onchange="toggleMode()">
      <span class="toggle-track"><span class="toggle-thumb"></span></span>
      <span id="modeLabel">TESTNET</span>
    </label>
    <span class="mode-badge mode-test" id="modeBadge">TESTNET</span>
    <button class="btn btn-start" id="startBtn" onclick="startBot()">▶ START</button>
    <button class="btn btn-stop"  id="stopBtn"  onclick="stopBot()" style="display:none">■ STOP</button>
  </div>
</div>

<div class="main">

  <!-- Metrics Row -->
  <div class="grid-4">
    <div class="card">
      <div class="card-title">STATUS</div>
      <div class="metric-val" id="statusVal" style="font-size:18px">
        <span class="status-dot status-IDLE" id="statusDot"></span>
        <span id="statusText">IDLE</span>
      </div>
      <div class="metric-sub" id="lastScan">—</div>
    </div>
    <div class="card">
      <div class="card-title">CAPITAL</div>
      <div class="metric-val" id="capitalVal">$0.00</div>
      <div class="metric-sub">Available USDT</div>
    </div>
    <div class="card">
      <div class="card-title">TOTAL PnL</div>
      <div class="metric-val" id="pnlVal">$0.00</div>
      <div class="metric-sub" id="pnlSub">0 trades closed</div>
    </div>
    <div class="card">
      <div class="card-title">WIN RATE</div>
      <div class="metric-val" id="wrVal">—</div>
      <div class="wr-bar"><div class="wr-fill" id="wrFill" style="width:0%"></div></div>
      <div class="metric-sub" id="wrSub">W: 0 / L: 0</div>
    </div>
  </div>

  <!-- Main content -->
  <div class="grid-3">

    <!-- Open Trades + Recent History -->
    <div style="display:flex;flex-direction:column;gap:14px">

      <div class="card">
        <div class="card-title">Open Positions</div>
        <div class="tbl-wrap">
          <table class="tbl">
            <thead>
              <tr>
                <th>Symbol</th><th>Dir</th><th>Entry</th><th>SL</th><th>TP</th><th>Notional</th><th></th>
              </tr>
            </thead>
            <tbody id="openTrades"><tr><td colspan="7" style="color:var(--dim);text-align:center;padding:20px">No open positions</td></tr></tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Trade History</div>
        <div class="tbl-wrap">
          <table class="tbl">
            <thead>
              <tr>
                <th>Symbol</th><th>Dir</th><th>Entry</th><th>Close</th><th>PnL</th><th>Status</th>
              </tr>
            </thead>
            <tbody id="recentTrades"><tr><td colspan="6" style="color:var(--dim);text-align:center;padding:20px">No closed trades yet</td></tr></tbody>
          </table>
        </div>
      </div>

    </div>

    <!-- Scanned Pairs -->
    <div class="card">
      <div class="card-title">Scanned Pairs</div>
      <div class="pairs-row" id="pairsRow">
        <span style="color:var(--dim);font-family:var(--mono);font-size:11px">Waiting for first scan…</span>
      </div>
      <div style="margin-top:20px">
        <div class="card-title">Stats</div>
        <table class="tbl">
          <tr><td style="color:var(--dim)">Total Trades</td><td id="stTotal">0</td></tr>
          <tr><td style="color:var(--dim)">Open</td><td id="stOpen">0</td></tr>
          <tr><td style="color:var(--dim)">Wins</td><td class="pnl-pos" id="stWins">0</td></tr>
          <tr><td style="color:var(--dim)">Losses</td><td class="pnl-neg" id="stLoss">0</td></tr>
          <tr><td style="color:var(--dim)">Net PnL</td><td id="stPnl">$0.00</td></tr>
        </table>
      </div>
    </div>

    <!-- Config -->
    <div class="card">
      <div class="card-title">Configuration</div>
      <div class="cfg-grid" id="cfgGrid">
        <!-- Populated by JS -->
      </div>
      <button class="save-btn" onclick="saveConfig()">Save Config</button>
    </div>

  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let isLive = false;
let cfgData = {};
let polling = null;

const cfgFields = [
  {key:"api_key",        label:"API KEY",         type:"password"},
  {key:"api_secret",     label:"API SECRET",      type:"password"},
  {key:"total_capital",  label:"CAPITAL (USDT)",  type:"number"},
  {key:"risk_per_trade", label:"RISK PER TRADE",  type:"number"},
  {key:"max_leverage",   label:"MAX LEVERAGE",    type:"number"},
  {key:"tp_ratio",       label:"TP RATIO (RR)",   type:"number"},
  {key:"pivot_len",      label:"PIVOT LENGTH",    type:"number"},
  {key:"scan_interval",  label:"SCAN INTERVAL(s)",type:"number"},
  {key:"kline_interval", label:"KLINE INTERVAL",  type:"text"},
  {key:"top_n_pairs",    label:"TOP N PAIRS",     type:"number"},
];

function toast(msg, color="#00d4ff") {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.style.borderColor = color;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 3000);
}

async function loadConfig() {
  const r = await fetch("/api/config");
  cfgData = await r.json();
  renderCfgGrid();
}

function renderCfgGrid() {
  const grid = document.getElementById("cfgGrid");
  grid.innerHTML = cfgFields.map(f => `
    <div class="field">
      <label>${f.label}</label>
      <input type="${f.type}" id="cfg_${f.key}" value="${cfgData[f.key] ?? ''}" autocomplete="off">
    </div>
  `).join("");
}

async function saveConfig() {
  cfgFields.forEach(f => {
    const el = document.getElementById("cfg_"+f.key);
    if (!el) return;
    const v = el.value;
    if (f.type === "number") cfgData[f.key] = parseFloat(v) || 0;
    else cfgData[f.key] = v;
  });
  cfgData.live_mode = isLive;
  await fetch("/api/config", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(cfgData)});
  toast("✅ Config saved");
}

function toggleMode() {
  isLive = document.getElementById("liveModeCheck").checked;
  document.getElementById("modeLabel").textContent = isLive ? "LIVE" : "TESTNET";
  const badge = document.getElementById("modeBadge");
  badge.textContent = isLive ? "LIVE" : "TESTNET";
  badge.className = "mode-badge " + (isLive ? "mode-live" : "mode-test");
  if (isLive) toast("⚠️ LIVE MODE — Real money at risk", "#ff3366");
}

async function startBot() {
  await saveConfig();
  const r = await fetch("/api/start", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({live_mode: isLive})
  });
  const d = await r.json();
  toast(d.msg, d.ok ? "#00ff88" : "#ff3366");
  if (d.ok) {
    document.getElementById("startBtn").style.display = "none";
    document.getElementById("stopBtn").style.display  = "inline-block";
    startPolling();
  }
}

async function stopBot() {
  await fetch("/api/stop", {method:"POST"});
  toast("Bot stopped", "#ffcc00");
  document.getElementById("startBtn").style.display = "inline-block";
  document.getElementById("stopBtn").style.display  = "none";
}

async function closeTrade(symbol) {
  if (!confirm(`Close ${symbol} manually?`)) return;
  const r = await fetch("/api/close_trade", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({symbol})
  });
  const d = await r.json();
  toast(d.ok ? `Closed ${symbol}` : d.msg);
}

function fmt(n, d=4) { return parseFloat(n||0).toFixed(d); }

async function poll() {
  try {
    const r = await fetch("/api/state");
    const s = await r.json();
    updateDashboard(s);
  } catch(e) {}
}

function updateDashboard(s) {
  // Status
  const dot = document.getElementById("statusDot");
  dot.className = "status-dot status-" + (s.status||"IDLE");
  document.getElementById("statusText").textContent = s.status || "IDLE";
  document.getElementById("lastScan").textContent = s.last_scan ? "Last: " + s.last_scan.split("T")[1]?.slice(0,8) + " UTC" : "—";

  // Capital
  document.getElementById("capitalVal").textContent = "$" + fmt(s.capital, 2);

  // PnL
  const pnl = s.stats?.total_pnl || 0;
  const pnlEl = document.getElementById("pnlVal");
  pnlEl.textContent = (pnl >= 0 ? "+" : "") + "$" + fmt(pnl, 4);
  pnlEl.style.color = pnl >= 0 ? "var(--green)" : "var(--red)";
  document.getElementById("pnlSub").textContent = (s.stats?.total || 0) + " trades closed";

  // Win rate
  const wr = (s.stats?.win_rate || 0) * 100;
  document.getElementById("wrVal").textContent = wr.toFixed(1) + "%";
  document.getElementById("wrFill").style.width = wr + "%";
  document.getElementById("wrSub").textContent = "W: " + (s.stats?.wins||0) + " / L: " + (s.stats?.losses||0);

  // Stats table
  document.getElementById("stTotal").textContent = s.stats?.total || 0;
  document.getElementById("stOpen").textContent  = s.stats?.open  || 0;
  document.getElementById("stWins").textContent  = s.stats?.wins  || 0;
  document.getElementById("stLoss").textContent  = s.stats?.losses|| 0;
  const sp = s.stats?.total_pnl||0;
  const spEl = document.getElementById("stPnl");
  spEl.textContent = (sp>=0?"+":"") + "$" + fmt(sp,4);
  spEl.className = sp>=0 ? "pnl-pos" : "pnl-neg";

  // Pairs
  const pr = document.getElementById("pairsRow");
  if (s.scanned_pairs && s.scanned_pairs.length) {
    pr.innerHTML = s.scanned_pairs.map(p =>
      `<span class="pair-tag active">${p}</span>`
    ).join("");
  }

  // Open trades
  const ot = document.getElementById("openTrades");
  if (s.open_trades && s.open_trades.length) {
    ot.innerHTML = s.open_trades.map(t => `
      <tr>
        <td>${t.symbol}</td>
        <td class="${t.direction=='LONG'?'long':'short'}">${t.direction}</td>
        <td>${fmt(t.entry)}</td>
        <td class="pnl-neg">${fmt(t.sl)}</td>
        <td class="pnl-pos">${fmt(t.tp)}</td>
        <td>$${fmt(t.notional,2)}</td>
        <td><button class="btn-sm" onclick="closeTrade('${t.symbol}')">CLOSE</button></td>
      </tr>
    `).join("");
  } else {
    ot.innerHTML = '<tr><td colspan="7" style="color:var(--dim);text-align:center;padding:20px">No open positions</td></tr>';
  }

  // Recent trades
  const rt = document.getElementById("recentTrades");
  const closed = (s.recent_trades||[]).filter(t => t.status !== "OPEN");
  if (closed.length) {
    rt.innerHTML = closed.map(t => {
      const pnl = t.pnl || 0;
      return `
        <tr>
          <td>${t.symbol}</td>
          <td class="${t.direction=='LONG'?'long':'short'}">${t.direction}</td>
          <td>${fmt(t.entry)}</td>
          <td>${fmt(t.close_price)}</td>
          <td class="${pnl>=0?'pnl-pos':'pnl-neg'}">${(pnl>=0?"+":"") + "$"+fmt(pnl,4)}</td>
          <td class="status-${t.status}">${t.status}</td>
        </tr>
      `;
    }).join("");
  } else {
    rt.innerHTML = '<tr><td colspan="6" style="color:var(--dim);text-align:center;padding:20px">No closed trades yet</td></tr>';
  }
}

function startPolling() {
  if (polling) clearInterval(polling);
  polling = setInterval(poll, 3000);
  poll();
}

// Init
loadConfig();
poll();
startPolling();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"\n🚀 Liquidity Sweep Bot Dashboard → http://localhost:{PORT}\n")
    app.run(debug=False, port=PORT, host="0.0.0.0")
