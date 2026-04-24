"""
Liquidity Sweep Bot — Pro Strategy v3
Concepts: ICT Kill Zones + SMC Order Blocks + BOS/CHoCH + FVG + Sweep + ATR
Multi-timeframe: 4h bias → 1h structure → 15m execution
"""
import os, time, json, math, logging, sys, threading
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd
import numpy as np

try:
    from binance.um_futures import UMFutures
    from binance.error import ClientError
except ImportError:
    raise ImportError("pip install binance-futures-connector")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("/tmp/bot.log"), logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("SweepBot")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BotConfig:
    api_key:    str   = ""
    api_secret: str   = ""
    live_mode:  bool  = False

    def __post_init__(self):
        if not self.api_key:    self.api_key    = os.environ.get("BINANCE_API_KEY","")
        if not self.api_secret: self.api_secret = os.environ.get("BINANCE_API_SECRET","")
        e = os.environ.get("LIVE_MODE","")
        if e: self.live_mode = e.lower()=="true"

    # Capital
    total_capital:    float = 30.0
    risk_per_trade:   float = 0.015   # 1.5% = ~$0.45
    max_leverage:     int   = 15
    max_open_trades:  int   = 3
    daily_loss_limit: float = 0.06    # 6% daily stop

    # Timeframes  (4h macro → 1h structure → 15m execution)
    tf_macro:  str = "4h";  lim_macro:  int = 200
    tf_struct: str = "1h";  lim_struct: int = 100
    tf_exec:   str = "15m"; lim_exec:   int = 150

    # Pivots
    pivot_len: int = 5

    # Order Block
    ob_lookback:     int   = 10   # bars to look back for OB
    ob_body_pct:     float = 0.50 # OB candle body must be >50% of range

    # FVG
    fvg_lookback: int = 8

    # Sweep quality
    wick_body_ratio: float = 1.4
    min_body_ratio:  float = 0.28

    # ATR
    atr_period:  int   = 14
    atr_sl_mult: float = 1.1   # tight SL — just beyond OB/sweep
    atr_tp_mult: float = 2.8   # 2.8R target

    # Volume
    vol_period:    int   = 20
    vol_spike_mul: float = 1.15

    # RSI
    rsi_period: int   = 14
    rsi_ob:     float = 72
    rsi_os:     float = 28

    # EMA trend
    ema_fast:  int = 21
    ema_slow:  int = 50
    ema_macro: int = 200

    # ICT Kill Zones UTC  (London 07-10, NY 13-16, Asia 00-03 avoided)
    session_filter: bool = True
    kill_zones: list = None   # set in __post_init__ below

    def __post_init__(self):
        if not self.api_key:    self.api_key    = os.environ.get("BINANCE_API_KEY","")
        if not self.api_secret: self.api_secret = os.environ.get("BINANCE_API_SECRET","")
        e = os.environ.get("LIVE_MODE","")
        if e: self.live_mode = e.lower()=="true"
        if self.kill_zones is None:
            # (start_hour, end_hour) UTC
            self.kill_zones = [(7,10),(12,16),(19,21)]

    # Confluence: minimum score to take trade (max possible ~9)
    min_score: int = 4

    # Pairs
    top_n_pairs:      int   = 10
    volume_threshold: float = 60_000_000

    # Timing
    scan_interval: int = 40
    sl_buffer:     float = 0.0012

    testnet_base_url: str = "https://testnet.binancefuture.com"
    live_base_url:    str = "https://fapi.binance.com"


@dataclass
class Trade:
    symbol:       str
    direction:    str
    entry:        float
    sl:           float
    tp:           float
    size:         float
    notional:     float
    open_time:    str
    status:       str   = "OPEN"
    close_price:  float = 0.0
    pnl:          float = 0.0
    order_id:     str   = ""
    signal_score: int   = 0
    atr:          float = 0.0
    reason:       str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
class I:
    @staticmethod
    def ema(s, n): return s.ewm(span=n, adjust=False).mean()

    @staticmethod
    def atr(df, n):
        h,l,c = df["high"],df["low"],df["close"]
        tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        return tr.ewm(span=n, adjust=False).mean()

    @staticmethod
    def rsi(s, n):
        d=s.diff(); g=d.clip(lower=0).ewm(span=n,adjust=False).mean()
        l=(-d.clip(upper=0)).ewm(span=n,adjust=False).mean()
        return 100-100/(1+g/l.replace(0,np.nan))

    @staticmethod
    def pivot_hi(s, n):
        r=pd.Series(np.nan,index=s.index); a=s.values
        for i in range(n,len(a)-n):
            w=a[i-n:i+n+1]
            if a[i]==w.max() and list(w).count(a[i])==1: r.iloc[i]=a[i]
        return r

    @staticmethod
    def pivot_lo(s, n):
        r=pd.Series(np.nan,index=s.index); a=s.values
        for i in range(n,len(a)-n):
            w=a[i-n:i+n+1]
            if a[i]==w.min() and list(w).count(a[i])==1: r.iloc[i]=a[i]
        return r

    @staticmethod
    def choch(highs, lows, direction):
        """Change of Character: first sign of reversal."""
        if direction == "BULL":
            # look for lower high after series of higher highs
            ph = highs.dropna()
            if len(ph) >= 2:
                return ph.iloc[-1] < ph.iloc[-2]
        else:
            pl = lows.dropna()
            if len(pl) >= 2:
                return pl.iloc[-1] > pl.iloc[-2]
        return False


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY: ICT / SMC FULL STACK
# ─────────────────────────────────────────────────────────────────────────────
class ProStrategy:
    """
    Signal stack (each adds to confluence score):
      [2] 4h EMA trend  (macro bias — non-negotiable direction gate)
      [1] 1h BOS or CHoCH  (market structure shift confirms reversal)
      [1] Order Block mitigation  (price returning to institutional OB)
      [1] Liquidity sweep of swing high/low
      [1] Fair Value Gap (imbalance within last N bars)
      [1] Volume spike on sweep/OB candle
      [1] RSI confluence (not overextended)
      [1] ICT Kill Zone timing
    Min score: 4  →  Takes only high-probability setups
    """

    def __init__(self, cfg: BotConfig):
        self.cfg = cfg

    # ── Kill Zone ─────────────────────────────────────────────────────────────
    def in_kill_zone(self) -> bool:
        if not self.cfg.session_filter: return True
        h = datetime.now(timezone.utc).hour
        return any(s <= h < e for s, e in self.cfg.kill_zones)

    # ── 4h Macro Bias ─────────────────────────────────────────────────────────
    def macro_bias(self, df4h: pd.DataFrame) -> str:
        if df4h.empty or len(df4h) < self.cfg.ema_macro: return "NEUTRAL"
        c    = df4h["close"]
        e21  = I.ema(c, self.cfg.ema_fast).iloc[-1]
        e50  = I.ema(c, self.cfg.ema_slow).iloc[-1]
        e200 = I.ema(c, self.cfg.ema_macro).iloc[-1]
        last = c.iloc[-1]
        if last > e21 > e50 > e200: return "BULL"
        if last < e21 < e50 < e200: return "BEAR"
        if last > e50 > e200:       return "BULL_WEAK"
        if last < e50 < e200:       return "BEAR_WEAK"
        return "NEUTRAL"

    # ── 1h Structure: BOS ─────────────────────────────────────────────────────
    def structure_bos(self, df1h: pd.DataFrame, direction: str) -> bool:
        """Break of Structure on 1h: close beyond last swing."""
        if df1h.empty or len(df1h) < 20: return False
        c = df1h["close"]
        ph = I.pivot_hi(df1h["high"], self.cfg.pivot_len).dropna()
        pl = I.pivot_lo(df1h["low"],  self.cfg.pivot_len).dropna()
        if direction == "LONG"  and not ph.empty: return c.iloc[-1] > ph.iloc[-1]
        if direction == "SHORT" and not pl.empty: return c.iloc[-1] < pl.iloc[-1]
        return False

    # ── Order Block Detection ─────────────────────────────────────────────────
    def find_order_block(self, df: pd.DataFrame,
                         direction: str) -> Optional[tuple]:
        """
        Bullish OB: last bearish candle before a strong bullish impulse.
        Bearish OB: last bullish candle before a strong bearish impulse.
        Returns (ob_high, ob_low) or None.
        """
        n   = len(df)
        lb  = min(self.cfg.ob_lookback, n - 2)
        c, o, h, l = df["close"], df["open"], df["high"], df["low"]

        for i in range(n-2, n-lb-1, -1):
            body  = abs(c.iloc[i] - o.iloc[i])
            rng   = h.iloc[i] - l.iloc[i]
            if rng == 0: continue
            body_pct = body / rng

            if direction == "LONG":
                # OB = bearish candle followed by bullish impulse
                if c.iloc[i] < o.iloc[i] and body_pct >= self.cfg.ob_body_pct:
                    # Check impulse after it
                    if i+1 < n and c.iloc[i+1] > h.iloc[i]:
                        return h.iloc[i], l.iloc[i]

            elif direction == "SHORT":
                # OB = bullish candle followed by bearish impulse
                if c.iloc[i] > o.iloc[i] and body_pct >= self.cfg.ob_body_pct:
                    if i+1 < n and c.iloc[i+1] < l.iloc[i]:
                        return h.iloc[i], l.iloc[i]
        return None

    # ── FVG ───────────────────────────────────────────────────────────────────
    def find_fvg(self, df: pd.DataFrame,
                 direction: str) -> Optional[tuple]:
        n = len(df)
        for j in range(n-1, max(n-self.cfg.fvg_lookback-2, 1), -1):
            if direction=="LONG" and j>=2:
                if df["low"].iloc[j] > df["high"].iloc[j-2]:
                    return df["high"].iloc[j-2], df["low"].iloc[j]
            elif direction=="SHORT" and j>=2:
                if df["high"].iloc[j] < df["low"].iloc[j-2]:
                    return df["high"].iloc[j], df["low"].iloc[j-2]
        return None

    # ── MAIN ANALYZE ──────────────────────────────────────────────────────────
    def analyze(self, df15: pd.DataFrame,
                df1h:  pd.DataFrame,
                df4h:  pd.DataFrame) -> Optional[dict]:

        # Minimum data
        if any(x.empty or len(x) < 50 for x in [df15, df1h]):
            return None
        if df4h.empty or len(df4h) < self.cfg.ema_macro:
            return None

        # ── Kill zone gate ────────────────────────────────────────────────────
        kz = self.in_kill_zone()

        # ── 4h Macro bias (hard gate — no counter-trend) ──────────────────────
        bias = self.macro_bias(df4h)
        if bias == "NEUTRAL": return None
        bull_ok = bias in ("BULL","BULL_WEAK")
        bear_ok = bias in ("BEAR","BEAR_WEAK")

        # ── Indicators on 15m ─────────────────────────────────────────────────
        c,h,l,o,v = df15["close"],df15["high"],df15["low"],df15["open"],df15["volume"]
        atr_s = I.atr(df15, self.cfg.atr_period)
        rsi_s = I.rsi(c,    self.cfg.rsi_period)
        vma   = v.rolling(self.cfg.vol_period).mean()

        atr = atr_s.iloc[-1]
        rsi = rsi_s.iloc[-1]

        # ── 15m pivots ────────────────────────────────────────────────────────
        ph_s = I.pivot_hi(h, self.cfg.pivot_len).dropna()
        pl_s = I.pivot_lo(l, self.cfg.pivot_len).dropna()
        if ph_s.empty or pl_s.empty: return None

        swing_hi = ph_s.iloc[-1]
        swing_lo = pl_s.iloc[-1]

        # ── Current candle ────────────────────────────────────────────────────
        i   = len(df15)-1
        hi  = h.iloc[i]; lo = l.iloc[i]
        op  = o.iloc[i]; cl = c.iloc[i]
        vol = v.iloc[i]; va = vma.iloc[i]

        body = abs(cl-op); rng = hi-lo
        if rng < 1e-10: return None

        lw = (cl-lo) if cl>op else (op-lo)
        uw = (hi-cl) if cl>op else (hi-op)

        bull_rej  = lw > uw * self.cfg.wick_body_ratio
        bear_rej  = uw > lw * self.cfg.wick_body_ratio
        bull_disp = cl > op and body > rng * self.cfg.min_body_ratio
        bear_disp = cl < op and body > rng * self.cfg.min_body_ratio
        vol_spike = va > 0 and vol >= va * self.cfg.vol_spike_mul

        bull_sweep = lo < swing_lo and cl > swing_lo and bull_rej and bull_disp
        bear_sweep = hi > swing_hi and cl < swing_hi and bear_rej and bear_disp

        def score_and_build(direction):
            score = 0; rsn = []

            # [2] Macro trend
            if bias == ("BULL" if direction=="LONG" else "BEAR"):
                score += 2; rsn.append("4h Trend ✓✓")
            else:
                score += 1; rsn.append("4h Trend(weak)")

            # [1] Sweep
            score += 1; rsn.append("Sweep+Disp")

            # [1] 1h BOS
            if self.structure_bos(df1h, direction):
                score += 1; rsn.append("1h BOS")

            # [1] Order Block
            ob = self.find_order_block(df15, direction)
            ob_hit = False
            if ob:
                ob_hi, ob_lo = ob
                if direction=="LONG"  and ob_lo <= cl <= ob_hi: ob_hit=True
                if direction=="SHORT" and ob_lo <= cl <= ob_hi: ob_hit=True
            if ob_hit:
                score += 1; rsn.append("OB mitigated")

            # [1] FVG
            fvg = self.find_fvg(df15, direction)
            if fvg:
                score += 1; rsn.append("FVG")

            # [1] Volume spike
            if vol_spike:
                score += 1; rsn.append("VolSpike")

            # [1] RSI
            if direction=="LONG"  and rsi < 50 and rsi > self.cfg.rsi_os:
                score += 1; rsn.append(f"RSI {rsi:.0f}")
            if direction=="SHORT" and rsi > 50 and rsi < self.cfg.rsi_ob:
                score += 1; rsn.append(f"RSI {rsi:.0f}")

            # [1] Kill zone bonus
            if kz:
                score += 1; rsn.append("KillZone")

            return score, rsn, ob, fvg

        # ── LONG evaluation ───────────────────────────────────────────────────
        if bull_sweep and bull_ok and rsi < self.cfg.rsi_ob:
            score, rsn, ob, fvg = score_and_build("LONG")
            if score < self.cfg.min_score: return None

            # SL: below sweep low + ATR buffer (tighter if OB present)
            ob_low = ob[1] if ob else None
            sl_base = ob_low if (ob_low and ob_low < swing_lo) else swing_lo
            sl  = sl_base - atr * self.cfg.atr_sl_mult
            sl  = min(sl, swing_lo * (1 - self.cfg.sl_buffer))
            risk = cl - sl
            if risk <= 0: return None
            tp  = cl + risk * self.cfg.atr_tp_mult

            return dict(direction="LONG", sweep_price=swing_lo, entry=cl,
                        sl=sl, tp=tp, atr=atr, score=score,
                        reason=" | ".join(rsn), bias=bias, rsi=rsi,
                        ob=ob, fvg=fvg, kz=kz)

        # ── SHORT evaluation ──────────────────────────────────────────────────
        if bear_sweep and bear_ok and rsi > self.cfg.rsi_os:
            score, rsn, ob, fvg = score_and_build("SHORT")
            if score < self.cfg.min_score: return None

            ob_high = ob[0] if ob else None
            sl_base = ob_high if (ob_high and ob_high > swing_hi) else swing_hi
            sl  = sl_base + atr * self.cfg.atr_sl_mult
            sl  = max(sl, swing_hi * (1 + self.cfg.sl_buffer))
            risk = sl - cl
            if risk <= 0: return None
            tp  = cl - risk * self.cfg.atr_tp_mult

            return dict(direction="SHORT", sweep_price=swing_hi, entry=cl,
                        sl=sl, tp=tp, atr=atr, score=score,
                        reason=" | ".join(rsn), bias=bias, rsi=rsi,
                        ob=ob, fvg=fvg, kz=kz)

        return None


# ─────────────────────────────────────────────────────────────────────────────
# MARKET DATA
# ─────────────────────────────────────────────────────────────────────────────
class MarketData:
    def __init__(self, client):
        self.client=client; self._ei=None; self._ei_ts=0

    def top_pairs(self, n, min_vol):
        try:
            tk = self.client.ticker_24hr_price_change()
            f  = [t for t in tk
                  if t["symbol"].endswith("USDT")
                  and not any(x in t["symbol"] for x in
                              ["BUSD","USDC","TUSD","USDP","DAI","FDUSD"])
                  and float(t["quoteVolume"]) >= min_vol
                  and float(t["lastPrice"])   > 0.001]
            f.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)
            syms = [t["symbol"] for t in f[:n]]
            log.info(f"Pairs: {syms}")
            return syms
        except Exception as e:
            log.error(f"top_pairs: {e}")
            return ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT"]

    def klines(self, symbol, interval, limit):
        try:
            raw = self.client.klines(symbol, interval, limit=limit)
            df  = pd.DataFrame(raw, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","quote_vol","trades",
                "tb_base","tb_quote","ignore"])
            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            return df.reset_index(drop=True)
        except Exception as e:
            log.error(f"klines {symbol}/{interval}: {e}")
            return pd.DataFrame()

    def price(self, symbol):
        try: return float(self.client.ticker_price(symbol=symbol)["price"])
        except: return 0.0

    def step_tick(self, symbol):
        now=time.time()
        if not self._ei or now-self._ei_ts>300:
            try: self._ei=self.client.exchange_info(); self._ei_ts=now
            except: return 0.001,0.01
        for s in self._ei.get("symbols",[]):
            if s["symbol"]==symbol:
                step=tick=0.0
                for f in s.get("filters",[]):
                    if f["filterType"]=="LOT_SIZE":     step=float(f["stepSize"])
                    if f["filterType"]=="PRICE_FILTER": tick=float(f["tickSize"])
                return step or 0.001, tick or 0.01
        return 0.001,0.01


# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class RM:
    def __init__(self, cfg): self.cfg=cfg

    def size(self, entry, sl, capital):
        r_usd = capital * min(self.cfg.risk_per_trade, 0.03)
        dist  = abs(entry-sl)
        if dist<1e-10: return 0,0
        qty = r_usd/dist
        notional = qty*entry
        cap = capital*self.cfg.max_leverage
        if notional>cap: qty=cap/entry; notional=cap
        return round(qty,6), round(notional,4)

    @staticmethod
    def round_step(qty, step):
        if step<=0: return qty
        return math.floor(qty/step)*step

    @staticmethod
    def p_round(price, tick):
        if tick<=0: return price
        f=1/tick; return round(round(price*f)/f,8)

    def daily_breached(self, start, now):
        return (start-now)/start >= self.cfg.daily_loss_limit


# ─────────────────────────────────────────────────────────────────────────────
# ORDER EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────
class Executor:
    def __init__(self, client, market, cfg):
        self.client=client; self.mkt=market; self.cfg=cfg; self.rm=RM(cfg)

    def set_lev(self, sym, lev):
        try:
            self.client.change_leverage(symbol=sym, leverage=lev)
            self.client.change_margin_type(symbol=sym, marginType="ISOLATED")
        except: pass

    def open(self, symbol, sig, capital) -> Optional[Trade]:
        d=sig["direction"]; entry=sig["entry"]; sl=sig["sl"]; tp=sig["tp"]
        qty, notional = self.rm.size(entry, sl, capital)
        if qty<=0 or notional<5.5:
            log.warning(f"{symbol}: notional ${notional:.2f} too small"); return None
        step,tick = self.mkt.step_tick(symbol)
        qty  = self.rm.round_step(qty, step)
        sl_r = self.rm.p_round(sl, tick)
        tp_r = self.rm.p_round(tp, tick)
        if qty<=0: return None

        side  = "BUY"  if d=="LONG" else "SELL"
        xs    = "SELL" if d=="LONG" else "BUY"
        self.set_lev(symbol, min(self.cfg.max_leverage,15))
        try:
            order = self.client.new_order(symbol=symbol,side=side,type="MARKET",quantity=qty)
            oid   = str(order.get("orderId",""))
            self.client.new_order(symbol=symbol,side=xs,type="STOP_MARKET",
                                  stopPrice=sl_r,closePosition=True,timeInForce="GTE_GTC")
            self.client.new_order(symbol=symbol,side=xs,type="TAKE_PROFIT_MARKET",
                                  stopPrice=tp_r,closePosition=True,timeInForce="GTE_GTC")
            t = Trade(symbol=symbol,direction=d,entry=entry,sl=sl_r,tp=tp_r,
                      size=qty,notional=round(notional,2),
                      open_time=datetime.utcnow().isoformat(),order_id=oid,
                      signal_score=sig.get("score",0),atr=round(sig.get("atr",0),6),
                      reason=sig.get("reason",""))
            log.info(f"✅ {d} {symbol} @ {entry:.5f} SL:{sl_r} TP:{tp_r} "
                     f"score:{sig.get('score',0)} | {sig.get('reason','')}")
            return t
        except ClientError as e:
            log.error(f"Order failed {symbol}: {e.error_message}"); return None
        except Exception as e:
            log.error(f"Order error {symbol}: {e}"); return None

    def close(self, trade, reason="MANUAL"):
        side = "SELL" if trade.direction=="LONG" else "BUY"
        try:
            self.client.new_order(symbol=trade.symbol,side=side,type="MARKET",
                                  quantity=trade.size,reduceOnly=True)
            self.client.cancel_open_orders(symbol=trade.symbol)
            log.info(f"❌ Closed {trade.symbol} ({reason})")
        except Exception as e:
            log.error(f"Close error {trade.symbol}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
class Journal:
    def __init__(self, path="/tmp/trades.json"):
        self.path=path; self.trades=[]; self._lock=threading.Lock(); self._load()

    def _load(self):
        try:
            with open(self.path) as f: data=json.load(f)
            self.trades=[Trade(**t) for t in data]
        except: self.trades=[]

    def _save(self):
        with open(self.path,"w") as f:
            json.dump([asdict(t) for t in self.trades],f,indent=2)

    def add(self, t):
        with self._lock: self.trades.append(t); self._save()

    def update(self, t):
        with self._lock: self._save()

    def open_trades(self): return [t for t in self.trades if t.status=="OPEN"]

    def stats(self):
        closed=[t for t in self.trades if t.status!="OPEN"]
        wins=[t for t in closed if t.pnl>0]
        losses=[t for t in closed if t.pnl<=0]
        wp=[t.pnl for t in wins]; lp=[t.pnl for t in losses]
        aw=sum(wp)/len(wp) if wp else 0
        al=sum(lp)/len(lp) if lp else 0
        return dict(total=len(closed),wins=len(wins),losses=len(losses),
                    win_rate=round(len(wins)/len(closed),3) if closed else 0,
                    total_pnl=round(sum(t.pnl for t in closed),4),
                    avg_win=round(aw,4),avg_loss=round(al,4),
                    realized_rr=round(abs(aw/al),2) if al else 0,
                    open=len(self.open_trades()))


# ─────────────────────────────────────────────────────────────────────────────
# BOT CORE
# ─────────────────────────────────────────────────────────────────────────────
class LiquiditySweepBot:
    def __init__(self, cfg: BotConfig, trades_path="/tmp/trades.json"):
        self.cfg=cfg; self.running=False
        self.journal=Journal(trades_path)
        self.strategy=ProStrategy(cfg)
        base = cfg.live_base_url if cfg.live_mode else cfg.testnet_base_url
        self.client=UMFutures(key=cfg.api_key,secret=cfg.api_secret,base_url=base)
        self.mkt=MarketData(self.client)
        self.executor=Executor(self.client,self.mkt,cfg)
        self.capital=cfg.total_capital
        self.start_capital=cfg.total_capital
        self.day_capital=cfg.total_capital
        self._status="IDLE"; self._last_scan=None
        self._scanned=[]; self._paused=False
        self._pause_reason=""; self._last_signals=[]
        log.info(f"SweepBot v3 PRO | {'LIVE' if cfg.live_mode else 'TESTNET'} | ${cfg.total_capital}")
        log.info("Strategy: ICT Kill Zones + SMC Order Blocks + BOS + FVG + Sweep + ATR")

    def balance(self):
        try:
            for b in self.client.balance():
                if b["asset"]=="USDT": return float(b["availableBalance"])
        except: pass
        return self.capital

    def reset_daily(self):
        now=datetime.utcnow()
        if now.hour==0 and now.minute<2:
            self.day_capital=self.capital
            self._paused=False; self._pause_reason=""
            log.info("Daily reset")

    def scan(self):
        if self._paused: self._status=f"PAUSED: {self._pause_reason}"; return
        self._status="SCANNING"
        open_t=self.journal.open_trades(); open_s={t.symbol for t in open_t}
        self.capital=self.balance()

        rm=RM(self.cfg)
        if rm.daily_breached(self.day_capital,self.capital):
            self._paused=True
            self._pause_reason=f"6% daily loss hit (${self.capital:.2f})"
            log.warning("⛔ Daily loss limit — paused")
            self._status="PAUSED"; return

        if len(open_t)>=self.cfg.max_open_trades:
            self._status="MAX_TRADES"; return

        syms=self.mkt.top_pairs(self.cfg.top_n_pairs,self.cfg.volume_threshold)
        self._scanned=syms; self._last_signals=[]

        for sym in syms:
            if sym in open_s: continue
            df15 = self.mkt.klines(sym, self.cfg.tf_exec,   self.cfg.lim_exec)
            df1h  = self.mkt.klines(sym, self.cfg.tf_struct, self.cfg.lim_struct)
            df4h  = self.mkt.klines(sym, self.cfg.tf_macro,  self.cfg.lim_macro)
            sig   = self.strategy.analyze(df15, df1h, df4h)
            if sig:
                sig["symbol"]=sym; self._last_signals.append(sig)
                log.info(f"🎯 {sym} {sig['direction']} score={sig['score']} "
                         f"bias={sig['bias']} kz={sig['kz']}")
                trade=self.executor.open(sym,sig,self.capital)
                if trade:
                    self.journal.add(trade); open_t.append(trade); open_s.add(sym)
                    if len(open_t)>=self.cfg.max_open_trades: break

        self._last_scan=datetime.utcnow().isoformat(); self._status="WATCHING"

    def monitor(self):
        for t in self.journal.open_trades():
            p=self.mkt.price(t.symbol)
            if p<=0: continue
            hit_tp=(t.direction=="LONG" and p>=t.tp) or (t.direction=="SHORT" and p<=t.tp)
            hit_sl=(t.direction=="LONG" and p<=t.sl) or (t.direction=="SHORT" and p>=t.sl)
            if hit_tp:
                pnl=abs(t.tp-t.entry)*t.size; t.status="TP"
                t.close_price=t.tp; t.pnl=round(pnl,4); self.capital+=pnl
                log.info(f"🏆 TP {t.symbol} +${pnl:.4f}"); self.journal.update(t)
            elif hit_sl:
                pnl=-abs(t.entry-t.sl)*t.size; t.status="SL"
                t.close_price=t.sl; t.pnl=round(pnl,4); self.capital+=pnl
                log.warning(f"💀 SL {t.symbol} ${pnl:.4f}"); self.journal.update(t)

    def get_dashboard_state(self):
        stats=self.journal.stats()
        open_t=self.journal.open_trades()
        recent=sorted(self.journal.trades,key=lambda x:x.open_time,reverse=True)[:25]
        return dict(mode="LIVE" if self.cfg.live_mode else "TESTNET",
                    status=self._status,capital=round(self.capital,4),
                    start_capital=round(self.start_capital,2),
                    day_capital=round(self.day_capital,2),
                    paused=self._paused,pause_reason=self._pause_reason,
                    last_scan=self._last_scan,scanned_pairs=self._scanned,
                    last_signals=self._last_signals,stats=stats,
                    open_trades=[asdict(t) for t in open_t],
                    recent_trades=[asdict(t) for t in recent],
                    in_kill_zone=self.strategy.in_kill_zone(),
                    config=dict(risk_per_trade=self.cfg.risk_per_trade,
                                max_leverage=self.cfg.max_leverage,
                                atr_tp_mult=self.cfg.atr_tp_mult,
                                tf_exec=self.cfg.tf_exec,
                                tf_struct=self.cfg.tf_struct,
                                tf_macro=self.cfg.tf_macro,
                                max_open=self.cfg.max_open_trades,
                                daily_limit=self.cfg.daily_loss_limit,
                                min_score=self.cfg.min_score))

    def run_loop(self):
        self.running=True; log.info("🚀 Bot loop started")
        while self.running:
            try:
                self.reset_daily(); self.monitor(); self.scan()
            except Exception as e:
                log.error(f"Loop error: {e}",exc_info=True); self._status="ERROR"
            time.sleep(self.cfg.scan_interval)

    def start(self): threading.Thread(target=self.run_loop,daemon=True).start()
    def stop(self):  self.running=False; self._status="STOPPED"
