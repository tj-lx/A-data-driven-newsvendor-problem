import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def parse_price(s):
    if pd.isna(s):
        return np.nan
    s = str(s)
    s = s.replace('€', '').replace('EUR', '').replace(' ', '')
    if ',' in s and '.' not in s:
        s = s.replace(',', '.')
    try:
        return float(s)
    except Exception:
        return np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv_path')
    ap.add_argument('--out', default=os.path.join('data', 'prepared', 'bakery_daily.csv'))
    args = ap.parse_args()

    encodings = ['utf-8', 'latin1', 'cp1252']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(args.csv_path, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        raise RuntimeError('Failed to read CSV with tried encodings')

    for c in list(df.columns):
        if str(c).lower().startswith('unnamed') or str(c).strip()=='' or str(c)=='Unnamed: 0':
            df = df.drop(columns=[c])

    if 'date' not in df.columns or 'article' not in df.columns:
        raise RuntimeError('CSV must contain columns: date, article')

    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    df = df.dropna(subset=['date'])

    if 'Quantity' in df.columns:
        qty = pd.to_numeric(df['Quantity'], errors='coerce')
    elif 'quantity' in df.columns:
        qty = pd.to_numeric(df['quantity'], errors='coerce')
    else:
        raise RuntimeError('CSV must contain Quantity column')
    df['Quantity'] = qty

    if 'unit_price' in df.columns:
        df['unit_price_clean'] = df['unit_price'].apply(parse_price)
    else:
        df['unit_price_clean'] = np.nan

    df = df[df['Quantity'].notna()]
    df = df[df['Quantity'] > 0]

    df['article'] = df['article'].astype(str)

    df['revenue'] = df['Quantity'] * df['unit_price_clean']

    agg = df.groupby(['date', 'article'], as_index=False).agg(
        sales=('Quantity', 'sum'),
        revenue=('revenue', 'sum')
    )
    agg['avg_price'] = np.where(agg['sales']>0, agg['revenue']/agg['sales'], np.nan)

    min_date = agg['date'].min()
    max_date = agg['date'].max()
    date_range = pd.date_range(min_date, max_date, freq='D').date
    articles = sorted(agg['article'].unique())
    idx = pd.MultiIndex.from_product([date_range, articles], names=['date', 'article'])
    agg = agg.set_index(['date','article']).reindex(idx)
    agg[['sales','revenue','avg_price']] = agg[['sales','revenue','avg_price']].fillna({'sales':0,'revenue':0,'avg_price':np.nan})
    agg = agg.reset_index()

    agg['date'] = pd.to_datetime(agg['date']).dt.date
    agg['weekday'] = pd.to_datetime(agg['date']).dt.dayofweek
    agg['month'] = pd.to_datetime(agg['date']).dt.month

    try:
        import holidays
        years = list(range(pd.to_datetime(min_date).year, pd.to_datetime(max_date).year+1))
        fr_holidays = holidays.France(years=years)
        agg['is_public_holiday'] = pd.to_datetime(agg['date']).dt.date.map(lambda d: d in fr_holidays)
    except Exception:
        agg['is_public_holiday'] = False

    agg = agg.sort_values(['article','date'])
    agg['lag_1'] = agg.groupby('article')['sales'].shift(1)
    agg['lag_7'] = agg.groupby('article')['sales'].shift(7)
    agg[['lag_1','lag_7']] = agg[['lag_1','lag_7']].fillna(0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    agg.to_csv(args.out, index=False)
    print(f'Wrote {args.out} with shape {agg.shape}')

if __name__ == '__main__':
    main()

