# data_project_table3_mark3.py
# ---------------------------------------------------------
# Cooper, Gulen, Schill (2008, JF) - Table III (Panel A & B) 복제용 풀 버전
#  - WRDS (Compustat + CRSP) 사용
#  - CCM(ccmxpf_lnkhist) 권한 없이 comp.security + crsp.msenames CUSIP 링크
#  - 11개 설명변수 생성:
#    ASSETG, L2ASSETG, BM, MV, BHRET6, BHRET36,
#    SYSALSEG, CI, NOA/A, ACCRUALS, SYASSETG
#  - Fama–MacBeth 모형 1~7 (All / Small) 추정
#  - 논문 스타일 Panel A/B 테이블 CSV + (가능하면) LaTeX 출력
#  - 기간별:
#      · 1963–2003 (원 논문)
#      · 2004–2024 (확장)
#      · 1963–2024 (전체 통합)
# ---------------------------------------------------------

import wrds
import pandas as pd
import numpy as np
import statsmodels.api as sm

# =========================================================
# 0. 세팅
# =========================================================
START_FYEAR = 1958          # AG 계산용 최소 연도
END_FYEAR   = 2023          # fyear 최대 (portyear=fyear+1 → 2024까지)

START_CRSP  = '1960-01-01'
END_CRSP    = '2024-12-31'

# 전체 통합 분석까지 고려해서 시작 포트폴리오 연도 1963으로 설정
BASE_START_PORTYEAR = 1963   # 전체 구간 시작 (포트폴리오 연도)
BASE_END_PORTYEAR   = 2003   # 논문 구간 끝
EXT_END_PORTYEAR    = 2024   # 확장 분석용 끝

# =========================================================
# 1. WRDS 접속
# =========================================================
print("[INFO] Connecting to WRDS...")
db = wrds.Connection()

# =========================================================
# 2. Compustat funda: 재무 변수 다운로드 (sic 제외) + company에서 sic merge
# =========================================================
print("[INFO] Downloading Compustat funda (annual)...")

sql_funda = f"""
    select gvkey, datadate, fyear, 
           at, sale, capx, ppent,
           act, che, lct, dlc, dltt, 
           seq, ceq, txditc,
           pstkrv, pstkl, pstk, 
           lt, dp
    from comp.funda
    where indfmt = 'INDL'
      and datafmt = 'STD'
      and popsrc = 'D'
      and consol = 'C'
      and fyear between {START_FYEAR} and {END_FYEAR}
      and at > 0
"""

funda = db.raw_sql(sql_funda, date_cols=['datadate'])
funda = funda.sort_values(['gvkey', 'fyear'])
print("[INFO] funda loaded:", funda.shape)

# ---------- SIC 코드: comp.company에서 가져와 merge ----------
print("[INFO] Loading SIC from comp.company...")

sql_sic = """
    select gvkey, sic
    from comp.company
"""

sic_df = db.raw_sql(sql_sic)
print("[INFO] SIC table loaded:", sic_df.shape)

# gvkey 기준 merge
funda = funda.merge(sic_df, on='gvkey', how='left')
print("[INFO] funda after SIC merge:", funda.shape)

# ---------- (1) 자산 성장률 ASSETG & L2ASSETG ----------
funda['lag_at'] = funda.groupby('gvkey')['at'].shift(1)
funda['ag'] = funda['at'] / funda['lag_at'] - 1
funda['ag_lag1'] = funda.groupby('gvkey')['ag'].shift(1)

# 포트폴리오 형성연도
funda['portyear'] = funda['fyear'] + 1

# 기간 필터 (포트폴리오 연도 기준, 전체 구간 1963–2024까지 포함)
funda = funda[(funda['portyear'] >= BASE_START_PORTYEAR - 1) &
              (funda['portyear'] <= EXT_END_PORTYEAR + 1)]

# ---------- (2) Book Equity & BM용 BE ----------
funda['pstk_fill'] = (
    funda['pstkrv']
    .fillna(funda['pstkl'])
    .fillna(funda['pstk'])
)
funda['txditc'] = funda['txditc'].fillna(0)
funda['be'] = funda['seq'].fillna(
    funda['ceq'].fillna(0) + funda['txditc'] - funda['pstk_fill'].fillna(0)
)

# ---------- (3) Sales Growth ----------
funda['lag_sale'] = funda.groupby('gvkey')['sale'].shift(1)
funda['salesg'] = funda['sale'] / funda['lag_sale'] - 1

# ---------- (4) Investment (CI) = CAPX / lag(AT) ----------
funda['ci'] = funda['capx'] / funda['lag_at']

# ---------- (5) Accruals (Sloan 1996 formula / AT 평균 기준) ----------
# ΔCA − ΔCL − ΔCash + ΔSTD − Dep  / Average(AT)
funda['lag_act'] = funda.groupby('gvkey')['act'].shift(1)
funda['lag_lct'] = funda.groupby('gvkey')['lct'].shift(1)
funda['lag_che'] = funda.groupby('gvkey')['che'].shift(1)
funda['lag_dlc'] = funda.groupby('gvkey')['dlc'].shift(1)
funda['lag_dp']  = funda.groupby('gvkey')['dp'].shift(1)

delta_ca  = funda['act'] - funda['lag_act']
delta_cl  = funda['lct'] - funda['lag_lct']
delta_cash = funda['che'] - funda['lag_che']
delta_std  = funda['dlc'] - funda['lag_dlc']
dep        = funda['dp']  # 감가상각

avg_at = (funda['at'] + funda['lag_at']) / 2
funda['accruals'] = (delta_ca - delta_cl - delta_cash + delta_std - dep) / avg_at

# ---------- (6) NOA/A ----------
# NOA = (Operating Assets - Operating Liabilities)
# OA ~ AT - CHE
# OL ~ LT - DLC - DLTT
funda['oa'] = funda['at'] - funda['che']
funda['ol'] = funda['lt'] - funda['dlc'].fillna(0) - funda['dltt'].fillna(0)
funda['noa'] = funda['oa'] - funda['ol']
funda['noa_a'] = funda['noa'] / funda['at']

# ---------- (7) Industry code (2-digit SIC) ----------
funda['sic'] = pd.to_numeric(funda['sic'], errors='coerce')
funda['sic2'] = (funda['sic'] // 100).astype("Int64")

print("[INFO] funda with main vars:", funda.shape)

# =========================================================
# 3. Systematic components: industry-year means (SYSALSEG, SYASSETG)
# =========================================================
print("[INFO] Computing systematic components (industry-year means)...")

ind_year = (
    funda
    .dropna(subset=['sic2'])
    .groupby(['sic2', 'fyear'])[['ag', 'salesg']]
    .mean()
    .reset_index()
    .rename(columns={'ag': 'syassetg', 'salesg': 'sysalesg'})
)

funda = funda.merge(ind_year, on=['sic2', 'fyear'], how='left')

# =========================================================
# 4. CUSIP 기반 Compustat–CRSP 링크 (CCM 없이)
# =========================================================
print("[INFO] Building gvkey–permno link via CUSIP (no CCM)...")

sql_link = """
    select distinct
           s.gvkey,
           n.permno,
           n.namedt as linkdt,
           coalesce(n.nameendt, date '9999-12-31') as linkenddt
    from comp.security as s
    join crsp.msenames as n
      on substr(s.cusip,1,8) = substr(n.ncusip,1,8)
"""

ccm = db.raw_sql(sql_link, date_cols=['linkdt', 'linkenddt'])
print(f"[INFO] CUSIP link table (approx CCM): {ccm.shape}")

# =========================================================
# 5. CRSP 월별 데이터 + BHRET6 / BHRET36
# =========================================================
print("[INFO] Downloading CRSP monthly (msf + msenames)...")

sql_crsp = f"""
    select m.permno, m.date, m.ret, m.prc, m.shrout,
           s.shrcd, s.exchcd
    from crsp.msf as m
    join crsp.msenames as s
      on m.permno = s.permno
     and m.date >= s.namedt
     and m.date <= coalesce(s.nameendt, date '9999-12-31')
    where m.date between '{START_CRSP}' and '{END_CRSP}'
      and s.shrcd in (10,11)
      and s.exchcd in (1,2,3)
"""

crsp = db.raw_sql(sql_crsp, date_cols=['date'])

crsp['me'] = crsp['prc'].abs() * crsp['shrout']
crsp['year'] = crsp['date'].dt.year
crsp['month'] = crsp['date'].dt.month
crsp['jdate'] = np.where(crsp['month'] >= 7, crsp['year'], crsp['year'] - 1)

crsp = crsp.sort_values(['permno', 'date'])
crsp['me_lag'] = crsp.groupby('permno')['me'].shift(1)

# 지난 6/36개월 buy-and-hold 수익률
crsp['ret_filled'] = 1 + crsp['ret'].fillna(0.0)

crsp['bhret6'] = (
    crsp.groupby('permno')['ret_filled']
        .rolling(6)
        .apply(lambda x: np.prod(x) - 1, raw=True)
        .reset_index(level=0, drop=True)
)
crsp['bhret36'] = (
    crsp.groupby('permno')['ret_filled']
        .rolling(36)
        .apply(lambda x: np.prod(x) - 1, raw=True)
        .reset_index(level=0, drop=True)
)

crsp = crsp.dropna(subset=['me_lag'])
print(f"[INFO] CRSP panel with BHRET6/36: {crsp.shape}")

# =========================================================
# 6. Compustat–CRSP 매칭 (gvkey → permno) 및 특성치 결합
# =========================================================
print("[INFO] Merging Compustat with CRSP via CUSIP link...")

ag_ccm = funda.merge(ccm, on='gvkey', how='inner')
ag_ccm = ag_ccm[(ag_ccm['datadate'] >= ag_ccm['linkdt']) &
                (ag_ccm['datadate'] <= ag_ccm['linkenddt'])]

ag_ccm = ag_ccm[['permno', 'portyear', 'fyear',
                 'ag', 'ag_lag1', 'be',
                 'salesg', 'sysalesg', 'syassetg',
                 'ci', 'noa_a', 'accruals']].drop_duplicates()

print(f"[INFO] Compustat–CRSP matched: {ag_ccm.shape}")

# =========================================================
# 7. 6월 기준 특성치 (AG, BM, MV, BHRET6/36, 기타 재무변수)
# =========================================================
print("[INFO] Building June characteristics...")

june = crsp[crsp['month'] == 6].copy()
june['portyear'] = june['jdate'] + 1
june['me_june'] = june['me']

june = june.merge(ag_ccm, on=['permno', 'portyear'], how='inner')

june = june.dropna(subset=['be', 'me_june'])
june['bm'] = june['be'] / june['me_june']
june['ln_me'] = np.log(june['me_june'])

# 1–99% winsorization
for col in ['ag', 'ag_lag1', 'bm', 'ln_me',
            'bhret6', 'bhret36', 'salesg', 'sysalesg',
            'ci', 'noa_a', 'accruals', 'syassetg']:
    if col in june.columns:
        lo = june[col].quantile(0.01)
        hi = june[col].quantile(0.99)
        june[col] = june[col].clip(lower=lo, upper=hi)

chars = june[['permno', 'portyear',
              'ag', 'ag_lag1', 'bm', 'ln_me',
              'bhret6', 'bhret36',
              'sysalesg', 'ci', 'noa_a', 'accruals', 'syassetg']].drop_duplicates()

print(f"[INFO] Cross-sectional chars (June): {chars.shape}")

# =========================================================
# 8. 1년 후 수익률 (BHRET1) 생성
# =========================================================
print("[INFO] Constructing 1-year BH returns (BHRET1)...")

crsp_ret = crsp.copy()
crsp_ret['ret_plus1'] = 1 + crsp_ret['ret'].fillna(0.0)

bh = (
    crsp_ret
    .groupby(['permno', 'jdate'])['ret_plus1']
    .prod()
    .reset_index()
)
bh['bhret1'] = bh['ret_plus1'] - 1
bh['portyear'] = bh['jdate']

bh = bh[['permno', 'portyear', 'bhret1']]
print(f"[INFO] BHRET1 panel: {bh.shape}")

cs = chars.merge(bh, on=['permno', 'portyear'], how='inner')

# 최종 패널 기간: 1963–2024
cs = cs[(cs['portyear'] >= BASE_START_PORTYEAR) &
        (cs['portyear'] <= EXT_END_PORTYEAR)]

print(f"[INFO] Final firm-year panel for regressions: {cs.shape}")

# 전체 패널에서 inf/-inf를 NaN으로 정리
cs = cs.replace([np.inf, -np.inf], np.nan)

# =========================================================
# 9. Size 그룹 (Small vs All)
# =========================================================
def flag_small(df, q=0.3):
    cutoff = df['ln_me'].quantile(q)
    df['small'] = df['ln_me'] <= cutoff
    return df

cs = cs.groupby('portyear', group_keys=False).apply(flag_small)

# =========================================================
# 10. Fama–MacBeth 회귀 함수
# =========================================================
def run_fmb(firm_panel: pd.DataFrame,
            start_year: int,
            end_year: int,
            xvars,
            yvar='bhret1'):
    yearly_betas = []

    for year in range(start_year, end_year + 1):
        tmp = firm_panel[firm_panel['portyear'] == year].copy()
        cols = [yvar] + xvars

        # 1) 숫자로 변환
        for c in cols:
            tmp[c] = pd.to_numeric(tmp[c], errors='coerce')

        # 2) inf/-inf → NaN
        tmp[cols] = tmp[cols].replace([np.inf, -np.inf], np.nan)

        # 3) NaN 있는 행 제거
        tmp = tmp.dropna(subset=cols)
        if tmp.shape[0] <= len(xvars) + 1:
            continue

        y = tmp[yvar].astype(float)
        X = tmp[xvars].astype(float)

        # 4) 상수항 추가
        X = sm.add_constant(X)

        # 5) 마지막 방어선: y/X 모두 finite인 행만 남기기
        mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        y = y[mask]
        X = X[mask]

        if X.shape[0] <= len(xvars) + 1:
            continue

        res = sm.OLS(y, X).fit()

        row = {'year': year}
        row['const'] = res.params.get('const', np.nan)
        for v in xvars:
            row[v] = res.params.get(v, np.nan)
        yearly_betas.append(row)

    betas = pd.DataFrame(yearly_betas)
    if betas.empty:
        return None, None

    coef_names = ['const'] + xvars
    rows = []
    for c in coef_names:
        series = betas[c].dropna()
        n = series.shape[0]
        mean = series.mean()
        std = series.std(ddof=1)
        t = mean / (std / np.sqrt(n)) if (n > 1 and std > 0) else np.nan
        rows.append({'var': c, 'coef': mean, 'tstat': t, 'n_years': n})

    summary = pd.DataFrame(rows)
    return summary, betas

# =========================================================
# 11. 모형별 xvars 설정 (Model 1~7) - 논문 스펙 + 공통통제
#   base_ctrl 는 모든 모형에 공통으로 들어가는 BM, MV, BHRET6, BHRET36
#   M1: AG
#   M2: AG + L2ASSETG
#   M3: AG + SYSALSEG
#   M4: AG + CI
#   M5: AG + NOA/A
#   M6: AG + ACCRUALS
#   M7: AG + SYASSETG
# =========================================================

base_ctrl = ['bm', 'ln_me', 'bhret6', 'bhret36']

x_m1 = ['ag'] + base_ctrl
x_m2 = ['ag', 'ag_lag1'] + base_ctrl
x_m3 = ['ag', 'sysalesg'] + base_ctrl
x_m4 = ['ag', 'ci'] + base_ctrl
x_m5 = ['ag', 'noa_a'] + base_ctrl
x_m6 = ['ag', 'accruals'] + base_ctrl
x_m7 = ['ag', 'syassetg'] + base_ctrl

models = {
    1: x_m1,
    2: x_m2,
    3: x_m3,
    4: x_m4,
    5: x_m5,
    6: x_m6,
    7: x_m7,
}

# =========================================================
# 12. 논문 스타일 Panel A / Panel B 테이블 생성 함수
#    - 각 Model에서 실제로 사용된 변수만 숫자 표기
#    - 사용되지 않은 변수는 완전 빈칸("") (논문과 동일한 형태)
# =========================================================

VAR_ORDER = [
    ('const',   'Constant'),
    ('ag',      'ASSETG'),
    ('ag_lag1', 'L2ASSETG'),
    ('bm',      'BM'),
    ('ln_me',   'MV'),
    ('bhret6',  'BHRET6'),
    ('bhret36', 'BHRET36'),
    ('sysalesg','SYSALSEG'),
    ('ci',      'CI'),
    ('noa_a',   'NOA/A'),
    ('accruals','ACCRUALS'),
    ('syassetg','SYASSETG'),
]

def build_panel(model_summaries: dict,
                panel_name: str,
                models_dict: dict) -> pd.DataFrame:
    """
    - model_summaries[m] : run_fmb 결과 (var, coef, tstat)
    - models_dict[m]     : 해당 모형에 포함된 xvars 리스트
    → 포함된 변수만 숫자/괄호 출력, 나머지는 완전 빈칸("")으로 둠.
    """
    rows = []
    for m in sorted(model_summaries.keys()):
        if model_summaries[m] is None:
            continue

        df = model_summaries[m].set_index('var')
        included = set(['const'] + models_dict[m])

        row_b = {'Panel': panel_name, 'Model': m,  'Type': 'Beta'}
        row_t = {'Panel': panel_name, 'Model': '', 'Type': 't-stat'}

        for code, label in VAR_ORDER:
            if code not in included:
                # 이 모형에서는 아예 사용하지 않은 변수 → 완전 빈칸
                row_b[label] = ""
                row_t[label] = ""
            else:
                if code in df.index:
                    b = df.loc[code, 'coef']
                    t = df.loc[code, 'tstat']
                    row_b[label] = f"{b:.4f}"
                    row_t[label] = f"({t:.2f})"
                else:
                    # 이론상 없어야 하지만, 혹시 빠져있으면 빈칸
                    row_b[label] = ""
                    row_t[label] = ""

        rows.append(row_b)
        rows.append(row_t)

    return pd.DataFrame(rows)

# =========================================================
# 13. 기간별 FMB 실행 + 테이블 생성
#    - (1) 1963–2003: 논문 원구간
#    - (2) 2004–2024: 최신 추가 구간
#    - (3) 1963–2024: 전체 통합 구간
#    * LaTeX(.tex)은 jinja2 없으면 자동으로 건너뜀
# =========================================================

def run_fmb_for_period(cs_panel: pd.DataFrame,
                       start_year: int,
                       end_year: int,
                       period_label: str):
    """
    period_label 예: '1963_2003', '2004_2024', '1963_2024'
    결과 파일 이름에 suffix로 붙음.
    """
    print(f"[INFO] Running FMB regressions ({start_year}–{end_year})...")

    cs_sub = cs_panel[(cs_panel['portyear'] >= start_year) &
                      (cs_panel['portyear'] <= end_year)].copy()
    cs_small_sub = cs_sub[cs_sub['small']].copy()

    panelA_results = {}
    panelB_results = {}

    for m, xvars in models.items():
        print(f"  - Model {m} (All firms)...")
        panelA_results[m], _ = run_fmb(cs_sub, start_year, end_year, xvars)

        print(f"  - Model {m} (Small firms)...")
        panelB_results[m], _ = run_fmb(cs_small_sub, start_year, end_year, xvars)

        # 모형별 raw summary도 기간별로 따로 저장 (원하면 나중에 확인용)
        if panelA_results[m] is not None:
            panelA_results[m].to_csv(
                f"table3_panelA_model{m}_all_{period_label}.csv",
                index=False
            )
        if panelB_results[m] is not None:
            panelB_results[m].to_csv(
                f"table3_panelB_model{m}_small_{period_label}.csv",
                index=False
            )

    print(f"[INFO] Saved model-wise FMB summaries for {start_year}–{end_year}.")

    # ---- 논문 스타일 Panel A/B 테이블 만들기 ----
    panelA = build_panel(panelA_results, "Panel A. All Firms", models)
    panelB = build_panel(panelB_results, "Panel B. Small Size Firms", models)

    panelA.to_csv(f"table3_panelA_like_paper_{period_label}.csv", index=False)
    panelB.to_csv(f"table3_panelB_like_paper_{period_label}.csv", index=False)

    col_format = 'lll' + 'r' * len(VAR_ORDER)

    # --- LaTeX 출력: jinja2 없으면 그냥 건너뛰기 ---
    try:
        with open(f"table3_panelA_{period_label}.tex", "w") as f:
            f.write(panelA.to_latex(index=False,
                                    na_rep='',
                                    column_format=col_format))
        with open(f"table3_panelB_{period_label}.tex", "w") as f:
            f.write(panelB.to_latex(index=False,
                                    na_rep='',
                                    column_format=col_format))
        print(f"[DONE] Table III Panel A/B LaTeX created for {start_year}–{end_year}.")
    except ImportError:
        print("[WARN] Jinja2가 설치되어 있지 않아 LaTeX(.tex) 파일 생성을 건너뜁니다.")
        print("       (.venv)에서 'pip install Jinja2' 하면 tex 파일도 생성할 수 있습니다.")
    except Exception as e:
        print("[WARN] LaTeX(.tex) 생성 중 다른 오류 발생, tex 생략하고 계속 진행합니다:", e)

    print(f"[DONE] Table III Panel A/B created for {start_year}–{end_year}.")

# ---------- 실제 실행: 세 기간 모두 ----------
# (1) 논문 원 구간: 1963–2003
run_fmb_for_period(
    cs_panel=cs,
    start_year=1963,
    end_year=2003,
    period_label="1963_2003"
)

# (2) 최신 추가 구간: 2004–2024
run_fmb_for_period(
    cs_panel=cs,
    start_year=2004,
    end_year=2024,
    period_label="2004_2024"
)

# (3) 전체 통합 구간: 1963–2024
run_fmb_for_period(
    cs_panel=cs,
    start_year=BASE_START_PORTYEAR,   # 1963
    end_year=EXT_END_PORTYEAR,        # 2024
    period_label=f"{BASE_START_PORTYEAR}_{EXT_END_PORTYEAR}"
)

print("[ALL DONE] data_project_table3_mark3.py finished (three periods: 1963–2003, 2004–2024, 1963–2024).")
