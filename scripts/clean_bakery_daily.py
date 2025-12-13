import argparse
import os
import pandas as pd
import numpy as np

def normalize_article(s):
    if pd.isna(s):
        return ''
    s = str(s).strip()
    s = ' '.join(s.split())
    return s.upper()

def winsorize(series, lower=0.01, upper=0.99):
    x = series.astype(float)
    ql = x.quantile(lower)
    qu = x.quantile(upper)
    return x.clip(lower=ql, upper=qu)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_csv')
    ap.add_argument('--out', default=os.path.join('data','bakery_daily.csv'))
    ap.add_argument('--min_nonzero_days', type=int, default=14)
    ap.add_argument('--top_n_articles', type=int, default=20)
    ap.add_argument('--max_zero_ratio', type=float, default=0.7)
    ap.add_argument('--winsorize_sales', action='store_true')
    ap.add_argument('--winsorize_price', action='store_true')
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    df['article'] = df['article'].apply(normalize_article)
    invalid = {'.','', 'NA','N/A','-','--'}
    df = df[~df['article'].isin(invalid)]

    df['date'] = pd.to_datetime(df['date'])
    # ensure numeric types before stats
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
    df['avg_price'] = pd.to_numeric(df['avg_price'], errors='coerce')
    df = df.sort_values(['article','date'])

    stats = df.groupby('article', as_index=False).agg(
        total_days=('sales', 'size'),
        zero_days=('sales', lambda x: int((x==0).sum())),
        nonzero_days=('sales', lambda x: int((x>0).sum())),
        total_sales=('sales','sum')
    )
    stats['zero_ratio'] = stats['zero_days'] / stats['total_days'].replace(0, np.nan)
    keep_by_days = set(stats.loc[stats['nonzero_days']>=args.min_nonzero_days, 'article'])
    keep_by_zero = set(stats.loc[stats['zero_ratio']<=args.max_zero_ratio, 'article'])
    top_n = set(stats.sort_values('total_sales', ascending=False)['article'].head(args.top_n_articles))
    keep_articles = (keep_by_days & keep_by_zero) | top_n
    df = df[df['article'].isin(keep_articles)]

    if args.winsorize_sales:
        df['sales'] = df.groupby('article')['sales'].transform(lambda s: winsorize(s, 0.01, 0.99))
    if args.winsorize_price:
        df['avg_price'] = df.groupby('article')['avg_price'].transform(lambda s: winsorize(s.dropna(), 0.01, 0.99)).combine_first(df['avg_price'])

    # strict: drop rows where sales is missing
    df = df.dropna(subset=['sales'])
    # retain zero-sales days; for sales>0, require valid positive avg_price
    nz = df['sales'] > 0
    df = df[ (~nz) | ((df['avg_price'].notna()) & (df['avg_price'] > 0)) ]

    df['revenue'] = df['sales'] * df['avg_price'].fillna(0)

    # keep existing lag_1 and lag_7 from prepared data

    # minimal time features retained in existing columns

    cols = [
        'date','article','sales','revenue','avg_price','weekday','month','is_public_holiday','lag_1','lag_7'
    ]
    df = df[cols]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'Wrote {args.out} with shape {df.shape}')

if __name__ == '__main__':
    main()
