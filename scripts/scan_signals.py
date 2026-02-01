"""
scan_signals.py
───────────────
Scarica dati OHLCV, rileva pattern Head & Shoulders (bearish + bullish),
e genera signals.json con i segnali attivi per il giorno successivo.

Parametri Scanner (medio-lungo termine, largo):
  Stop Loss:    3.5%
  Take Profit:  9.0%
  Trailing:     2.5% (attiva solo dopo profitto >= 2.5%)

Lista titoli:
  - Di default: i 42 MIB40 nel file (hardcoded)
  - Se esiste  data/symbols.txt  -> aggiunge i ticker contenuti (uno per riga)
"""

import json, os, sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────
# CONFIGURAZIONE SCANNER (medio-lungo termine)
# ─────────────────────────────────────────────────────────────────
STOP_LOSS_PCT    = 3.5   # %
TAKE_PROFIT_PCT  = 9.0   # %
TRAILING_PCT     = 2.5   # % – attiva dopo profitto >= questo valore

SWING_PERIOD     = 5     # period per swing detection (uguale al JS)
NECKLINE_WINDOW  = 20    # max candle dopo spalla DX per cercare rottura
SHOULDER_TOL     = 0.15  # tolleranza spalle (15%)
HEAD_MIN_DIFF    = 0.05  # testa deve distare almeno 5% dalle spalle

# Giorni di storico da scaricare
HISTORY_DAYS = 365       # 1 anno è sufficiente per pattern recenti

# ─────────────────────────────────────────────────────────────────
# LISTA TITOLI DEFAULT (MIB40)
# ─────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    'A2A.MI','AMP.MI','AZM.MI','BGN.MI','BMED.MI','BMPS.MI','BAMI.MI',
    'BPSO.MI','BPE.MI','BRE.MI','BC.MI','BZU.MI','CPR.MI','CE.MI',
    'DIA.MI','ENEL.MI','ENI.MI','ERG.MI','RACE.MI','FBK.MI','G.MI',
    'HER.MI','IP.MI','ISP.MI','INW.MI','IG.MI','IVG.MI','LDO.MI',
    'MB.MI','MONC.MI','PIRC.MI','PST.MI','PRY.MI','REC.MI','SPM.MI',
    'SRG.MI','STMMI.MI','TIT.MI','TEN.MI','TRN.MI','UCG.MI','UNI.MI'
]

# Percorsi (relativi alla root del repo)
SYMBOLS_FILE = "data/symbols.txt"
OUTPUT_FILE  = "data/signals.json"


# ═════════════════════════════════════════════════════════════════
# 1. CARICA LISTA TITOLI
# ═════════════════════════════════════════════════════════════════
def load_tickers():
    tickers = set(DEFAULT_TICKERS)
    if os.path.exists(SYMBOLS_FILE):
        with open(SYMBOLS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    tickers.add(line.upper())
        print(f"[INFO] symbols.txt trovato -> {len(tickers)} titoli totali")
    else:
        print(f"[INFO] Nessun symbols.txt -> uso default ({len(tickers)} titoli)")
    return sorted(tickers)


# ═════════════════════════════════════════════════════════════════
# 2. DOWNLOAD DATI
# ═════════════════════════════════════════════════════════════════
def download_data(ticker, days=HISTORY_DAYS):
    start = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
        if df.empty:
            return None
        # Appiattisci MultiIndex colonne se presente
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        needed = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in needed if c in df.columns]]
        df = df.dropna().reset_index()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[WARN] Download fallito per {ticker}: {e}")
        return None


# ═════════════════════════════════════════════════════════════════
# 3. SWING POINTS (uguale alla logica JS)
# ═════════════════════════════════════════════════════════════════
def find_swing_points(data, period=SWING_PERIOD, find_tops=True):
    swings = []
    prices = data['high'].values if find_tops else data['low'].values
    n = len(data)
    for i in range(period, n - period):
        current = prices[i]
        is_swing = True
        for j in range(1, period + 1):
            before = prices[i - j]
            after  = prices[i + j]
            if find_tops:
                if before >= current or after >= current:
                    is_swing = False
                    break
            else:
                if before <= current or after <= current:
                    is_swing = False
                    break
        if is_swing:
            swings.append({
                'index': i,
                'price': float(current),
                'date':  data['date'].iloc[i]
            })
    return swings


# ═════════════════════════════════════════════════════════════════
# 4. DETECT HEAD & SHOULDERS (uguale alla logica JS)
# ═════════════════════════════════════════════════════════════════
def detect_hs(data, bearish=True):
    patterns = []
    swings = find_swing_points(data, SWING_PERIOD, find_tops=bearish)
    if len(swings) < 5:
        return patterns
    closes = data['close'].values
    for i in range(len(swings) - 4):
        ls = swings[i]
        lv = swings[i + 1]
        h  = swings[i + 2]
        rv = swings[i + 3]
        rs = swings[i + 4]
        if bearish:
            valid = (h['price'] > ls['price'] * (1 + HEAD_MIN_DIFF) and
                     h['price'] > rs['price'] * (1 + HEAD_MIN_DIFF) and
                     abs(ls['price'] - rs['price']) / ls['price'] < SHOULDER_TOL)
        else:
            valid = (h['price'] < ls['price'] * (1 - HEAD_MIN_DIFF) and
                     h['price'] < rs['price'] * (1 - HEAD_MIN_DIFF) and
                     abs(ls['price'] - rs['price']) / ls['price'] < SHOULDER_TOL)
        if not valid:
            continue
        neckline = (lv['price'] + rv['price']) / 2.0
        neckline_break = None
        search_end = min(rs['index'] + NECKLINE_WINDOW, len(closes))
        for j in range(rs['index'] + 1, search_end):
            if bearish and closes[j] < neckline:
                neckline_break = j
                break
            elif (not bearish) and closes[j] > neckline:
                neckline_break = j
                break
        if neckline_break is None:
            continue
        patterns.append({
            'type':              'bearish' if bearish else 'bullish',
            'left_shoulder':     ls,
            'head':              h,
            'right_shoulder':    rs,
            'left_valley':       lv,
            'right_valley':      rv,
            'neckline':          neckline,
            'break_index':       neckline_break,
            'break_date':        data['date'].iloc[neckline_break],
            'break_price':       float(closes[neckline_break]),
            'pattern_height':    abs(h['price'] - neckline)
        })
    return patterns


# ═════════════════════════════════════════════════════════════════
# 5. SIMULA ENTRY (giorno successivo alla rottura)
# ═════════════════════════════════════════════════════════════════
def simulate_entry(pattern, data):
    entry_idx = pattern['break_index'] + 1
    if entry_idx >= len(data):
        return None
    return {
        'entry_price': float(data['close'].iloc[entry_idx]),
        'entry_date':  data['date'].iloc[entry_idx],
        'entry_index': entry_idx
    }


# ═════════════════════════════════════════════════════════════════
# 6. GENERA SEGNALI ATTIVI
# ═════════════════════════════════════════════════════════════════
def generate_signal(ticker, pattern, entry):
    entry_price = entry['entry_price']
    is_bullish  = pattern['type'] == 'bullish'
    sl_dist = entry_price * (STOP_LOSS_PCT / 100)
    tp_dist = entry_price * (TAKE_PROFIT_PCT / 100)
    stop_loss   = entry_price - sl_dist if is_bullish else entry_price + sl_dist
    take_profit = entry_price + tp_dist if is_bullish else entry_price - tp_dist
    signal_type = 'BUY' if is_bullish else 'SELL'
    height_pct  = pattern['pattern_height'] / pattern['head']['price'] * 100
    confidence  = 'high' if height_pct > 5.0 else 'medium'
    return {
        'symbol':         ticker.replace('.MI', ''),
        'ticker':         ticker,
        'signal_type':    signal_type,
        'entry_price':    round(entry_price, 4),
        'stop_loss':      round(stop_loss, 4),
        'take_profit':    round(take_profit, 4),
        'neckline':       round(pattern['neckline'], 4),
        'pattern_height': round(pattern['pattern_height'], 4),
        'height_pct':     round(height_pct, 2),
        'confidence':     confidence,
        'break_date':     pattern['break_date'].strftime('%Y-%m-%d'),
        'entry_date':     entry['entry_date'].strftime('%Y-%m-%d'),
        'head_price':     round(pattern['head']['price'], 4),
        'ls_price':       round(pattern['left_shoulder']['price'], 4),
        'rs_price':       round(pattern['right_shoulder']['price'], 4),
        'sl_pct':         STOP_LOSS_PCT,
        'tp_pct':         TAKE_PROFIT_PCT,
        'trailing_pct':   TRAILING_PCT
    }


# ═════════════════════════════════════════════════════════════════
# 7. MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(" H&S SCANNER - Generazione segnali")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC")
    print("=" * 60)

    tickers = load_tickers()
    print(f"\nScanning {len(tickers)} titoli...\n")

    all_signals = []
    today = datetime.today().date()
    recent_cutoff = today - timedelta(days=7)  # 7 calendari ~ 5 sessioni

    for ticker in tickers:
        print(f"  [{ticker:12s}] ", end="", flush=True)

        df = download_data(ticker)
        if df is None or len(df) < 50:
            print("skip (dati insufficienti)")
            continue

        bearish_patterns = detect_hs(df, bearish=True)
        bullish_patterns = detect_hs(df, bearish=False)
        all_patterns = bearish_patterns + bullish_patterns

        if not all_patterns:
            print("nessun pattern")
            continue

        found_signal = False
        # Ordina per data rottura decrescente (il più recente prima)
        all_patterns.sort(key=lambda p: p['break_date'], reverse=True)

        for p in all_patterns:
            break_date = p['break_date']
            if hasattr(break_date, 'date'):
                break_date = break_date.date()
            if break_date < recent_cutoff:
                continue

            entry = simulate_entry(p, df)
            if entry is None:
                continue

            # Verifica che il trade non sia già stato chiuso da SL/TP
            is_bullish  = p['type'] == 'bullish'
            entry_price = entry['entry_price']
            sl = entry_price * (1 - STOP_LOSS_PCT/100) if is_bullish else entry_price * (1 + STOP_LOSS_PCT/100)
            tp = entry_price * (1 + TAKE_PROFIT_PCT/100) if is_bullish else entry_price * (1 - TAKE_PROFIT_PCT/100)

            already_closed = False
            for idx in range(entry['entry_index'] + 1, len(df)):
                close = float(df['close'].iloc[idx])
                if is_bullish:
                    if close <= sl or close >= tp:
                        already_closed = True
                        break
                else:
                    if close >= sl or close <= tp:
                        already_closed = True
                        break

            if already_closed:
                continue

            signal = generate_signal(ticker, p, entry)
            all_signals.append(signal)
            print(f"{signal['signal_type']} @ {signal['entry_price']} (conf: {signal['confidence']})")
            found_signal = True
            break  # Un solo segnale per titolo (il più recente)

        if not found_signal:
            print("nessun segnale attivo")

    # ── Scrivi signals.json ──
    output = {
        'generated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
        'parameters': {
            'stop_loss_pct':   STOP_LOSS_PCT,
            'take_profit_pct': TAKE_PROFIT_PCT,
            'trailing_pct':    TRAILING_PCT,
            'swing_period':    SWING_PERIOD
        },
        'signals': all_signals
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f" OK {len(all_signals)} segnali attivi -> {OUTPUT_FILE}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
