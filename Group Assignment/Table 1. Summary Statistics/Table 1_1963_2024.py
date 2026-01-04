# ============================================================
# Cooper, Gulen, Schill (2008) Table 1 Replication Script
# Extended sample: 1963–2024
# + Add ACCRUALS, ISSUANCE to Summary Statistics
# ============================================================

import os
import wrds
import pandas as pd
import numpy as np
from scipy import stats

# 저장 경로
SAVE_PATH = r"C:\Users\Admin\OneDrive\바탕 화면\data"
os.makedirs(SAVE_PATH, exist_ok=True)

# WRDS 연결
db = wrds.Connection()
pd.set_option("display.max_columns", 100)

# =========================
# Sample period (DATA PULL)
# =========================
PORTYEAR_MIN = 1968
PORTYEAR_MAX = 2024

ISSUANCE_LAG = 5

START_FYEAR = PORTYEAR_MIN - ISSUANCE_LAG   # ✅ 1963
END_YEAR    = 2024

START_DATE = f"{START_FYEAR}-01-01"
END_DATE   = f"{END_YEAR}-12-31"

# ============================================================
# 1. Compustat Annual (funda) 불러오기
#   ✅ ACCRUALS/ISSUANCE 계산용 컬럼 추가(act, che, lct, txp, dp, csho)
# ============================================================
sql_comp = f"""
    SELECT
        gvkey,
        datadate,
        fyear,
        indfmt,
        datafmt,
        popsrc,
        consol,
        at,          -- Total Assets
        sale,        -- Sales
        seq,         -- Stockholders' Equity
        ceq,         -- Common Equity
        txditc,      -- Deferred Taxes (for BE)
        pstkrv,
        pstkl,
        pstk,
        ni,          -- Net Income
        dltt,        -- Long-term Debt
        dlc,         -- Debt in Current Liabilities
        ib,          -- Income before Extraordinary Items
        oiadp,       -- Operating Income After Depreciation

        -- ✅ ACCRUALS 재료 (Sloan-style)
        act,         -- Current Assets
        che,         -- Cash and Short-Term Investments
        lct,         -- Current Liabilities
        txp,         -- Income Taxes Payable
        dp,          -- Depreciation and Amortization

        -- ✅ ISSUANCE 재료 (5y change in shares outstanding)
        csho         -- Common Shares Outstanding
    FROM comp.funda
    WHERE indfmt = 'INDL'
      AND datafmt = 'STD'
      AND popsrc = 'D'
      AND consol = 'C'
      AND fyear BETWEEN {START_FYEAR} AND {END_YEAR}
"""

print(">>> Running Compustat query")
comp = db.raw_sql(sql_comp, date_cols=['datadate'])
print(">>> Compustat loaded:", comp.shape)

comp.to_csv(
    os.path.join(SAVE_PATH, "Compustat_funda_1963_2024.csv"),
    index=False
)
print("Saved RAW: Compustat_funda_1963_2024.csv")

# 기본 필터
comp = comp[comp['at'] > 0].copy()

# ============================================================
# 2. PERMNO–GVKEY 링크 (CUSIP 기반)
# ============================================================
sql_link = """
    SELECT
        CAST(a.permno AS INT) AS permno,
        b.gvkey,
        a.namedt,
        a.nameendt
    FROM crsp.msenames a
    LEFT JOIN (
        SELECT DISTINCT gvkey, cusip, iid
        FROM comp.security
        WHERE cusip IS NOT NULL
          AND RIGHT(iid, 1) <> 'C'
    ) b
      ON a.ncusip = SUBSTR(b.cusip, 1, 8)
    WHERE a.ncusip IS NOT NULL
    ORDER BY permno, namedt
"""
link_raw = db.raw_sql(sql_link, date_cols=['namedt', 'nameendt'])

# =========================
# RAW DATA SAVE (Link table)
# =========================
link_raw.to_csv(
    os.path.join(SAVE_PATH, "Crsp_compustat_link_1963_2024.csv"),
    index=False
)
print("Saved RAW: Crsp_compustat_link_1963_2024.csv")

link_raw = link_raw[link_raw['gvkey'].notna()].copy()

link = (
    link_raw
    .groupby(['permno', 'gvkey'])
    .agg(linkdt=('namedt', 'min'), linkenddt=('nameendt', 'max'))
    .reset_index()
)

# Compustat + link merge
comp_ccm = pd.merge(comp, link, on='gvkey', how='inner')
comp_ccm = comp_ccm[
    (comp_ccm['datadate'] >= comp_ccm['linkdt']) &
    (comp_ccm['datadate'] <= comp_ccm['linkenddt'])
].copy()

# ============================================================
# 3. CRSP Monthly (msf + msenames) 불러오기
# ============================================================
sql_crsp_m = f"""
    SELECT
        a.permno,
        a.date,
        a.ret,
        a.retx,
        a.prc,
        a.shrout,
        b.shrcd,
        b.exchcd
    FROM crsp.msf a
    LEFT JOIN crsp.msenames b
      ON a.permno = b.permno
     AND a.date BETWEEN b.namedt AND b.nameendt
    WHERE a.date BETWEEN '{START_DATE}' AND '{END_DATE}'
"""

print(">>> Running CRSP query")
crsp_m = db.raw_sql(sql_crsp_m, date_cols=['date'])
print(">>> CRSP loaded:", crsp_m.shape)

# =========================
# RAW DATA SAVE (CRSP Monthly)
# =========================
crsp_m.to_csv(
    os.path.join(SAVE_PATH, "Crsp_monthly_1963_2024.csv"),
    index=False
)
print("Saved RAW: Crsp_monthly_1963_2024.csv")
crsp_m = crsp_m[
    crsp_m['shrcd'].isin([10, 11]) &
    crsp_m['exchcd'].isin([1, 2, 3])
].copy()

crsp_m['year'] = crsp_m['date'].dt.year
crsp_m['month'] = crsp_m['date'].dt.month
crsp_m['me'] = crsp_m['prc'].abs() * crsp_m['shrout'] / 1000.0

# ============================================================
# 4. BHRET6 / BHRET36 계산
# ============================================================
crsp_m = crsp_m.sort_values(['permno', 'date'])
crsp_m['ret'] = pd.to_numeric(crsp_m['ret'], errors='coerce')

def rolling_bh_transform(x, window):
    return (1.0 + x).rolling(window=window, min_periods=window).apply(np.prod, raw=True) - 1.0

crsp_m['bhret_6'] = crsp_m.groupby('permno')['ret'].transform(lambda x: rolling_bh_transform(x, 6))
crsp_m['bhret_36'] = crsp_m.groupby('permno')['ret'].transform(lambda x: rolling_bh_transform(x, 36))

crsp_june = crsp_m[crsp_m['month'] == 6].copy()
crsp_june = crsp_june[['permno', 'year', 'me', 'bhret_6', 'bhret_36']].rename(columns={
    'year': 'portyear',
    'me': 'me_june',
    'bhret_6': 'bhret6',
    'bhret_36': 'bhret36'
})

crsp_dec = crsp_m[crsp_m['month'] == 12].copy()
crsp_dec = crsp_dec[['permno', 'year', 'me']].rename(columns={'year': 'year_dec', 'me': 'me_dec'})

# ============================================================
# 5. 회계 변수 생성 (ASSETG, L2ASSETG, BE, BM, EP, ROA, LEV)
#   ✅ + ACCRUALS, ISSUANCE 추가
# ============================================================
comp_ccm = comp_ccm.sort_values(['gvkey', 'datadate']).copy()
comp_ccm['portyear'] = (comp_ccm['datadate'] + pd.DateOffset(months=6)).dt.year

# ASSETG / L2ASSETG
comp_ccm['at_lag1'] = comp_ccm.groupby('gvkey')['at'].shift(1)
comp_ccm['at_lag2'] = comp_ccm.groupby('gvkey')['at'].shift(2)

comp_ccm['assetg'] = (comp_ccm['at'] - comp_ccm['at_lag1']) / comp_ccm['at_lag1']
comp_ccm['l2assetg'] = (comp_ccm['at_lag1'] - comp_ccm['at_lag2']) / comp_ccm['at_lag2']

# ---- Book Equity (BE) ----
comp_ccm['pstk_com'] = comp_ccm['pstkrv'].fillna(comp_ccm['pstkl']).fillna(comp_ccm['pstk'])

comp_ccm['be'] = comp_ccm['seq']
mask_seq_na = comp_ccm['be'].isna()
comp_ccm.loc[mask_seq_na, 'be'] = (
    comp_ccm.loc[mask_seq_na, 'ceq'] +
    comp_ccm.loc[mask_seq_na, 'txditc'].fillna(0) -
    comp_ccm.loc[mask_seq_na, 'pstk_com'].fillna(0)
)

comp_ccm = comp_ccm[comp_ccm['be'] > 0].copy()

# LEV / ROA / ASSETS
comp_ccm['debt'] = comp_ccm[['dltt', 'dlc']].fillna(0).sum(axis=1)
comp_ccm['lev'] = comp_ccm['debt'] / comp_ccm['at']
comp_ccm['roa'] = comp_ccm['oiadp'] / comp_ccm['at']
comp_ccm['assets'] = comp_ccm['at']

# Table 1 formation year restriction
comp_ccm = comp_ccm[
    (comp_ccm['portyear'] >= PORTYEAR_MIN) &
    (comp_ccm['portyear'] <= PORTYEAR_MAX)
].copy()

# Optional: drop missing growth variables
comp_ccm = comp_ccm.dropna(subset=['assetg', 'l2assetg'])

# ✅ ACCRUALS (Sloan-style)
# Accruals = [(ΔCA - ΔCash) - (ΔCL - ΔSTD - ΔTP) - Dep] / AvgTA
# 여기서는 STD≈DLC(단기부채), TP≈TXP(세금지급), Dep≈DP로 구현
comp_ccm['act_lag1'] = comp_ccm.groupby('gvkey')['act'].shift(1)
comp_ccm['che_lag1'] = comp_ccm.groupby('gvkey')['che'].shift(1)
comp_ccm['lct_lag1'] = comp_ccm.groupby('gvkey')['lct'].shift(1)
comp_ccm['dlc_lag1'] = comp_ccm.groupby('gvkey')['dlc'].shift(1)
comp_ccm['txp_lag1'] = comp_ccm.groupby('gvkey')['txp'].shift(1)

d_act = comp_ccm['act'] - comp_ccm['act_lag1']
d_che = comp_ccm['che'] - comp_ccm['che_lag1']
d_lct = comp_ccm['lct'] - comp_ccm['lct_lag1']
d_dlc = comp_ccm['dlc'] - comp_ccm['dlc_lag1']
d_txp = comp_ccm['txp'] - comp_ccm['txp_lag1']

avg_ta = (comp_ccm['at'] + comp_ccm['at_lag1']) / 2.0

comp_ccm['accruals'] = (
    ((d_act.fillna(0) - d_che.fillna(0)) -
     (d_lct.fillna(0) - d_dlc.fillna(0) - d_txp.fillna(0)) -
     comp_ccm['dp'].fillna(0))
    / avg_ta
)

# ✅ ISSUANCE (5-year change in shares outstanding)
# "five year change in the number of equity shares outstanding"
comp_ccm['csho_lag5'] = comp_ccm.groupby('gvkey')['csho'].shift(5)
comp_ccm['issuance'] = (comp_ccm['csho'] - comp_ccm['csho_lag5']) / comp_ccm['csho_lag5']


# ============================================================
# 6. Compustat 패널과 CRSP (June/Dec) 합치기
# ============================================================
panel = pd.merge(comp_ccm, crsp_june, on=['permno', 'portyear'], how='inner')
panel = pd.merge(panel, crsp_dec, left_on=['permno', 'portyear'], right_on=['permno', 'year_dec'], how='left')

panel['bm'] = panel['be'] / panel['me_dec']
panel['ep'] = panel['ni'] / panel['me_dec']

panel = panel.dropna(subset=['assetg', 'me_june', 'bm']).copy()

# ============================================================
# 7. 연도별 ASSETG decile + cross-sectional median
#   ✅ ACCRUALS, ISSUANCE도 같이 요약
# ============================================================
def assign_deciles_and_summarize(df_year):
    df = df_year.copy().dropna(subset=['assetg'])
    n = df.shape[0]
    if n == 0:
        return pd.DataFrame()

    df['rank_assetg'] = df['assetg'].rank(method='first')
    df['pct_assetg'] = df['rank_assetg'] / (n + 1)

    df['decile'] = np.floor(df['pct_assetg'] * 10).astype(int) + 1
    df['decile'] = df['decile'].clip(1, 10)

    grouped = df.groupby('decile')

    summary = grouped.agg({
        'assetg': 'median',
        'l2assetg': 'median',
        'assets': 'median',
        'me_june': 'median',
        'bm': 'median',
        'ep': 'median',
        'lev': 'median',
        'roa': 'median',
        'bhret6': 'median',
        'bhret36': 'median',

        # ✅ 추가
        'accruals': 'median',
        'issuance': 'median'
    }).rename(columns={
        'assetg': 'ASSETG',
        'l2assetg': 'L2ASSETG',
        'assets': 'ASSETS',
        'me_june': 'MV',
        'bm': 'BM',
        'ep': 'EP',
        'lev': 'LEV',
        'roa': 'ROA',
        'bhret6': 'BHRET6',
        'bhret36': 'BHRET36',

        # ✅ 추가
        'accruals': 'ACCRUALS',
        'issuance': 'ISSUANCE'
    })

    mv_avg = grouped['me_june'].mean().rename('MV_AVG')
    summary = summary.join(mv_avg)
    summary['year'] = df['portyear'].iloc[0]

    return summary.reset_index()

yearly_deciles = (
    panel
    .groupby('portyear', group_keys=False)
    .apply(assign_deciles_and_summarize)
)
yearly_deciles = yearly_deciles.dropna(subset=['decile']).copy()

# ============================================================
# 8. 최종 Table 1: decile별 time-series 평균
#   ✅ ACCRUALS, ISSUANCE 포함
# ============================================================
table1 = (
    yearly_deciles
    .groupby('decile')
    .agg({
        'ASSETG': 'mean',
        'L2ASSETG': 'mean',
        'ASSETS': 'mean',
        'MV': 'mean',
        'MV_AVG': 'mean',
        'BM': 'mean',
        'EP': 'mean',
        'LEV': 'mean',
        'ROA': 'mean',
        'BHRET6': 'mean',
        'BHRET36': 'mean',

        # ✅ 추가
        'ACCRUALS': 'mean',
        'ISSUANCE': 'mean'
    })
    .sort_index()
)

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 50)
print(table1.round(4))

# ============================================================
# 9. Spread (10–1) 과 t-stat 계산
#   ✅ ACCRUALS, ISSUANCE 포함
# ============================================================
spread_results = {}

cols = [
    'ASSETG', 'L2ASSETG', 'ASSETS', 'MV', 'MV_AVG',
    'BM', 'EP', 'LEV', 'ROA', 'BHRET6', 'BHRET36',
    'ACCRUALS', 'ISSUANCE'   # ✅ 추가
]

spread_ts = {}

pivoted = yearly_deciles.pivot(index='year', columns='decile')  # 반복 pivot 비용 줄이기

for c in cols:
    diff_series = pivoted[c][10] - pivoted[c][1]
    spread_ts[c] = diff_series

    spread_mean = diff_series.mean()
    spread_t = diff_series.mean() / (diff_series.std() / np.sqrt(len(diff_series)))

    spread_results[c] = [spread_mean, spread_t]

spread_df = pd.DataFrame(spread_results, index=['Spread(10-1)', 't-stat'])

# ============================================================
# 10. Final Table 1 + Spread rows
# ============================================================
final_table1 = pd.concat([table1, spread_df])
print(final_table1.round(4))

# ============================================================
# 11. 결과 테이블 CSV로 저장
# ============================================================

output_filename = (
    f"Table1_Summary_with_ACCRUALS_ISSUANCE_{PORTYEAR_MIN}_{PORTYEAR_MAX}.csv"
)

final_table1.to_csv(output_filename)
print(f">>> Final Table saved as: {output_filename}")

