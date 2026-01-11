#!/usr/bin/env python3
"""
VNSTOCK ANALYSIS - Phân tích và dự báo cổ phiếu Việt Nam
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import time

warnings.filterwarnings('ignore')

# ============================================================
# CẤU HÌNH - THAY ĐỔI DANH SÁCH MÃ TẠI ĐÂY
# ============================================================

STOCK_LIST = ['VNM', 'FPT', 'VIC', 'HPG', 'MWG', 'TCB', 'VCB', 'ACB', 'VPB', 'DGW']
DAYS_BACK = 180
OUTPUT_DIR = './output'

# ============================================================
# HÀM LẤY DỮ LIỆU
# ============================================================

def get_stock_data(symbol, start_date, end_date):
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        df = stock.quote.history(start=start_date, end=end_date, interval='1D')
        return df
    except Exception as e:
        try:
            from vnstock import Quote
            quote = Quote(symbol=symbol, source='VCI')
            df = quote.history(start=start_date, end=end_date, interval='1D')
            return df
        except Exception as e2:
            print(f"  Lỗi {symbol}: {e2}")
            return None

def get_index_data(index_code, start_date, end_date):
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=index_code, source='VCI')
        df = stock.quote.history(start=start_date, end=end_date, interval='1D')
        return df
    except:
        return None

# ============================================================
# TÍNH CHỈ BÁO KỸ THUẬT
# ============================================================

def calculate_indicators(df):
    results = {}
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float)
    n = len(df)
    
    # SMA
    for p in [5, 10, 20, 50]:
        if n >= p:
            results[f'SMA_{p}'] = c.rolling(p).mean().round(2)
    
    # EMA
    for p in [12, 26]:
        if n >= p:
            results[f'EMA_{p}'] = c.ewm(span=p, adjust=False).mean().round(2)
    
    # MACD
    if n >= 26:
        e12 = c.ewm(span=12, adjust=False).mean()
        e26 = c.ewm(span=26, adjust=False).mean()
        macd = e12 - e26
        sig = macd.ewm(span=9, adjust=False).mean()
        hist = macd - sig
        results['MACD'] = macd.round(3)
        results['MACD_Signal'] = sig.round(3)
        results['MACD_Hist'] = hist.round(3)
    
    # RSI
    if n >= 14:
        d = c.diff()
        g = d.where(d > 0, 0).rolling(14).mean()
        ls = (-d.where(d < 0, 0)).rolling(14).mean()
        results['RSI'] = (100 - 100/(1 + g/ls)).round(2)
    
    # Stochastic
    if n >= 14:
        lo = l.rolling(14).min()
        hi = h.rolling(14).max()
        results['Stoch_K'] = (100 * (c - lo) / (hi - lo)).round(2)
        results['Stoch_D'] = results['Stoch_K'].rolling(3).mean().round(2)
    
    # ATR
    if n >= 14:
        pc = c.shift(1)
        tr = pd.concat([h-l, abs(h-pc), abs(l-pc)], axis=1).max(axis=1)
        results['ATR'] = tr.ewm(span=14, adjust=False).mean().round(2)
    
    # Bollinger Bands
    if n >= 20:
        sma = c.rolling(20).mean()
        std = c.rolling(20).std()
        results['BB_Upper'] = (sma + 2*std).round(2)
        results['BB_Middle'] = sma.round(2)
        results['BB_Lower'] = (sma - 2*std).round(2)
    
    # MFI
    if n >= 14:
        tp = (h + l + c) / 3
        mf = tp * v
        pmf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        nmf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        results['MFI'] = (100 - 100/(1 + pmf/nmf.replace(0, np.nan))).round(2)
    
    return results

# ============================================================
# XÁC ĐỊNH PHA THỊ TRƯỜNG
# ============================================================

def detect_market_phase(df):
    if len(df) < 10:
        return 'KHÔNG_RÕ', 0
    
    recent = df.tail(10)
    close = recent['close'].values
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and pd.notna(df['RSI'].iloc[-1]) else 50
    stoch_k = df['Stoch_K'].iloc[-1] if 'Stoch_K' in df.columns and pd.notna(df['Stoch_K'].iloc[-1]) else 50
    
    min_low = np.min(recent['low'].values)
    max_high = np.max(recent['high'].values)
    price_range = max_high - min_low
    position = (close[-1] - min_low) / price_range * 100 if price_range > 0 else 50
    
    trend_3d = (close[-1] - close[-3]) / close[-3] * 100 if len(close) >= 3 else 0
    
    if position < 25 and rsi < 35:
        return 'ĐÁY', min(90, 50 + (35 - rsi))
    elif position > 75 and rsi > 65:
        return 'ĐỈNH', min(90, 50 + (rsi - 65))
    elif trend_3d > 1.5 and rsi > 45 and rsi < 70:
        return 'TĂNG', min(80, 50 + trend_3d * 3)
    elif trend_3d < -1.5 and rsi < 55:
        return 'GIẢM', min(80, 50 + abs(trend_3d) * 3)
    else:
        return 'TÍCH_LŨY', 50

# ============================================================
# DỰ BÁO
# ============================================================

def forecast(df, symbol):
    if len(df) < 30:
        return {'symbol': symbol, 'error': 'Thiếu dữ liệu'}
    
    df = df.sort_values('time').reset_index(drop=True)
    current = df.iloc[-1]
    close = float(current['close'])
    
    rsi = current.get('RSI', 50)
    rsi = rsi if pd.notna(rsi) else 50
    stoch_k = current.get('Stoch_K', 50)
    stoch_k = stoch_k if pd.notna(stoch_k) else 50
    
    phase, conf = detect_market_phase(df)
    
    closes = df['close'].values
    n = len(closes)
    
    mom_1d = (closes[-1] - closes[-2]) / closes[-2] * 100 if n > 1 else 0
    mom_3d = (closes[-1] - closes[-4]) / closes[-4] * 100 if n > 3 else 0
    mom_5d = (closes[-1] - closes[-6]) / closes[-6] * 100 if n > 5 else 0
    
    # Tính điểm xu hướng
    score = 0
    score += 2 if mom_1d > 0.5 else (-2 if mom_1d < -0.5 else 0)
    score += 2 if mom_3d > 1 else (-2 if mom_3d < -1 else 0)
    score += 1 if rsi < 40 else (-1 if rsi > 60 else 0)
    score += 1 if stoch_k < 30 else (-1 if stoch_k > 70 else 0)
    
    # Dự báo
    base = score * 0.15
    if phase == 'ĐÁY' and rsi < 35:
        base = max(base, 0.5)
    elif phase == 'ĐỈNH' and rsi > 70:
        base = min(base, -0.5)
    
    t1 = round(base, 2)
    t2 = round(base * 0.85, 2)
    t3 = round(base * 0.7, 2)
    t5 = round(base * 0.5, 2)
    total_t5 = round(t1 + t2 + t3 + t5 * 2, 2)
    
    def calc_price(pct):
        return round(close * (1 + pct/100), 2)
    
    # Hành động
    if phase == 'ĐÁY' and rsi < 35:
        action = '🟢 MUA MẠNH'
    elif phase == 'ĐÁY' or (phase == 'GIẢM' and rsi < 35):
        action = '🟢 MUA'
    elif phase == 'ĐỈNH' and rsi > 70:
        action = '🔴 BÁN MẠNH'
    elif phase == 'ĐỈNH' or (phase == 'TĂNG' and rsi > 65):
        action = '🔴 BÁN'
    elif phase == 'TĂNG':
        action = '🟢 GIỮ'
    elif score > 2:
        action = '🟢 CÂN NHẮC MUA'
    elif score < -2:
        action = '🔴 CÂN NHẮC BÁN'
    else:
        action = '⚪ CHỜ'
    
    # Hỗ trợ / Kháng cự
    bb_lower = current.get('BB_Lower', close * 0.95)
    bb_upper = current.get('BB_Upper', close * 1.05)
    support = round(min(bb_lower if pd.notna(bb_lower) else close * 0.95, df['low'].tail(20).min()), 2)
    resist = round(max(bb_upper if pd.notna(bb_upper) else close * 1.05, df['high'].tail(20).max()), 2)
    
    return {
        'symbol': symbol,
        'gia': close,
        'T1': calc_price(t1),
        'T2': calc_price(t1 + t2),
        'T3': calc_price(t1 + t2 + t3),
        'T5': calc_price(total_t5),
        'pct_T5': total_t5,
        'pha': phase,
        'action': action,
        'rsi': round(rsi, 1),
        'stoch_k': round(stoch_k, 1),
        'support': support,
        'resist': resist,
        'cat_lo': round(support * 0.97, 2),
    }

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🚀 VNSTOCK ANALYSIS - Phân tích & Dự báo Cổ phiếu Việt Nam")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=DAYS_BACK)).strftime('%Y-%m-%d')
    
    print(f"\n📅 Thời gian: {start_date} → {end_date}")
    print(f"📋 Số mã: {len(STOCK_LIST)}")
    
    # Chỉ số thị trường
    print("\n📈 Chỉ số thị trường:")
    for idx in ['VNINDEX', 'VN30']:
        df = get_index_data(idx, start_date, end_date)
        if df is not None and len(df) > 0:
            df = df.sort_values('time', ascending=False)
            cur = df.iloc[0]['close']
            prev = df.iloc[1]['close'] if len(df) > 1 else cur
            chg = (cur - prev) / prev * 100
            print(f"  {idx}: {cur:,.2f} ({chg:+.2f}%)")
    
    # Quét cổ phiếu
    print("\n" + "-" * 70)
    print("🔄 Đang quét cổ phiếu...")
    print("-" * 70)
    
    results = []
    failed = []
    
    for i, sym in enumerate(STOCK_LIST):
        print(f"[{i+1}/{len(STOCK_LIST)}] {sym}...", end=" ")
        
        try:
            df = get_stock_data(sym, start_date, end_date)
            
            if df is None or len(df) < 30:
                print(f"⚠️ Thiếu dữ liệu")
                failed.append(sym)
                continue
            
            for col in ['close', 'high', 'low', 'open', 'volume']:
                df[col] = df[col].astype(float)
            df['time'] = pd.to_datetime(df['time'])
            
            indicators = calculate_indicators(df)
            for col, vals in indicators.items():
                df[col] = vals
            
            df = df.sort_values('time', ascending=False).reset_index(drop=True)
            
            fc = forecast(df.copy(), sym)
            if 'error' not in fc:
                results.append(fc)
                print(f"✅ {fc['action']} | RSI: {fc['rsi']}")
            else:
                print(f"⚠️ {fc['error']}")
                failed.append(sym)
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ {str(e)[:40]}")
            failed.append(sym)
    
    # Kết quả
    print("\n" + "=" * 70)
    print("📊 KẾT QUẢ")
    print("=" * 70)
    
    if results:
        mua = [r for r in results if 'MUA' in r['action']]
        ban = [r for r in results if 'BÁN' in r['action']]
        giu = [r for r in results if r['action'] == '🟢 GIỮ']
        cho = [r for r in results if r['action'] == '⚪ CHỜ']
        
        if mua:
            print("\n🟢 CƠ HỘI MUA:")
            for r in mua:
                print(f"  {r['symbol']:6} | Giá: {r['gia']:>10,.0f} | {r['pha']:8} | {r['action']}")
                print(f"         | T5: {r['pct_T5']:+.1f}% → {r['T5']:,.0f} | Cắt lỗ: {r['cat_lo']:,.0f}")
        
        if ban:
            print("\n🔴 NÊN BÁN:")
            for r in ban:
                print(f"  {r['symbol']:6} | Giá: {r['gia']:>10,.0f} | {r['pha']:8} | {r['action']}")
        
        if giu:
            print("\n🟢 GIỮ:")
            for r in giu:
                print(f"  {r['symbol']:6} | Giá: {r['gia']:>10,.0f} | {r['pha']:8}")
        
        if cho:
            print("\n⚪ CHỜ:")
            for r in cho:
                print(f"  {r['symbol']:6} | Giá: {r['gia']:>10,.0f} | {r['pha']:8}")
        
        # Bảng dự báo
        print("\n" + "=" * 70)
        print("📈 BẢNG DỰ BÁO GIÁ (5 NGÀY):")
        print("=" * 70)
        print(f"{'Mã':>6} {'Giá':>10} {'T1':>10} {'T2':>10} {'T3':>10} {'T5':>10} {'%T5':>7}")
        print("-" * 70)
        for r in results:
            print(f"{r['symbol']:>6} {r['gia']:>10,.0f} {r['T1']:>10,.0f} {r['T2']:>10,.0f} {r['T3']:>10,.0f} {r['T5']:>10,.0f} {r['pct_T5']:>+6.1f}%")
        
        # Xuất Excel
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_file = f"{OUTPUT_DIR}/BaoCao_{ts}.xlsx"
        
        try:
            df_result = pd.DataFrame(results)
            df_result.to_excel(excel_file, index=False)
            print(f"\n✅ Đã xuất: {excel_file}")
        except Exception as e:
            print(f"\n❌ Lỗi xuất Excel: {e}")
    
    if failed:
        print(f"\n⚠️ Mã lỗi: {', '.join(failed)}")
    
    print("\n" + "=" * 70)
    print("🎉 HOÀN THÀNH!")
    print("=" * 70)

if __name__ == "__main__":
    main()
