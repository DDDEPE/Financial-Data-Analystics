# hw1_p1_data_collection.py
# HW1 문제1 총 수정본(월말 데이터만 추출)
# - .env 기반 WRDS 자동 로그인(프롬프트 없음)
# - 1순위: 월별 테이블에서 월말 데이터 로드
# - 2순위: 일별 테이블에서 SQL 윈도우로 "해당 월 마지막 거래일"만 추출
# - mret(월수익률) 복원(가격 기반) + mktcap(가격×주식수) 계산
# - 10개국 필터, ISO3 정규화, 통합 저장 + 국가별 분할 저장

import os, sys, re
from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# 설정
# =========================
START_DATE = "2020-03-01"
END_DATE   = "2024-12-31"
DATA_DIR = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)

COUNTRY_NAME_TO_ISO3 = {
    "Australia": "AUS", "France": "FRA", "Germany": "DEU",
    "Japan": "JPN", "United Kingdom": "GBR",
    "Brazil": "BRA", "China": "CHN", "India": "IND",
    "South Africa": "ZAF", "Turkey": "TUR",
}
TARGET_ISO3 = {"AUS","FRA","DEU","JPN","GBR","BRA","CHN","IND","ZAF","TUR"}
DEV_ISO3    = {"AUS","FRA","DEU","JPN","GBR"}

ISO2_TO_ISO3 = {
    "AU":"AUS","FR":"FRA","DE":"DEU","JP":"JPN","GB":"GBR",
    "BR":"BRA","CN":"CHN","IN":"IND","ZA":"ZAF","TR":"TUR"
}
ISO3_TO_ISO2 = {v:k for k,v in ISO2_TO_ISO3.items()}

# =========================
# 유틸
# =========================
def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

def eom_floor(dt_series: pd.Series) -> pd.Series:
    return pd.to_datetime(dt_series).dt.to_period("M").dt.to_timestamp("M")

def normalize_country_to_iso3(row, fic_col):
    val = str(row.get(fic_col, "")).strip()
    if not val:
        return None
    v = val.upper()
    if re.fullmatch(r"[A-Z]{2}", v):
        return ISO2_TO_ISO3.get(v)
    if re.fullmatch(r"[A-Z]{3}", v):
        return v
    return ISO2_TO_ISO3.get(v[:2])

def save_dual(df: pd.DataFrame, csv_path: Path, pq_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("Saved CSV:", csv_path.resolve())
    engine = None
    try:
        import pyarrow; engine = "pyarrow"
    except Exception:
        try:
            import fastparquet; engine = "fastparquet"
        except Exception:
            engine = None
    if engine:
        df.to_parquet(pq_path, index=False, engine=engine)
        print(f"Saved Parquet ({engine}):", pq_path.resolve())
    else:
        print("Parquet 엔진(pyarrow/fastparquet) 미설치 → Parquet 저장 생략")

# =========================
# WRDS 연결(.env)
# =========================
def connect_wrds():
    try:
        import wrds
    except ImportError:
        print("pip install wrds pandas numpy python-dotenv"); sys.exit(1)

    # .env 로드
    try:
        from dotenv import load_dotenv
        for p in [Path.cwd()/".env", Path(__file__).with_name(".env")]:
            if p.exists(): load_dotenv(p)
        load_dotenv()
    except Exception:
        pass

    user = os.getenv("WRDS_USERNAME") or os.getenv("WRDS_USER")
    pwd  = os.getenv("WRDS_PASSWORD") or os.getenv("PGPASSWORD")
    if not user or not pwd:
        raise RuntimeError("WRDS_USERNAME / WRDS_PASSWORD를 .env 또는 환경변수로 설정하세요.")

    os.environ["PGUSER"] = user
    os.environ["PGPASSWORD"] = pwd

    conn = wrds.Connection(wrds_username=user, wrds_password=pwd)
    # 긴 쿼리 안전장치
    try:
        conn.raw_sql("SET statement_timeout TO '15min';")
        conn.raw_sql("SET work_mem TO '512MB';")
    except Exception:
        pass
    return conn

# =========================
# information_schema
# =========================
def list_tables(db, schema_like="comp_global%"):
    q = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_schema ILIKE %s
      AND table_type='BASE TABLE'
    """
    return db.raw_sql(q, params=(schema_like,))

def get_cols(db, schema, table) -> set:
    q = """SELECT column_name FROM information_schema.columns
           WHERE table_schema=%s AND table_name=%s"""
    cols = db.raw_sql(q, params=(schema, table))["column_name"].str.lower().tolist()
    return set(cols)

# =========================
# 월별/일별 테이블 탐색
# =========================
def find_monthly_table(db):
    cand = list_tables(db, "comp_global%")
    if cand.empty:
        return None
    cand["score"] = cand["table_name"].str.lower().apply(
        lambda s: (
            (("sec" in s) or ("security" in s)) * 3 +
            (("secm" in s) * 5) +
            (("m" in s or "mon" in s or "monthly" in s) * 2)
        )
    )
    cand = cand.sort_values("score", ascending=False)

    need_any_id = [{"gvkey","gvkeyx"}, {"iid","iid2","issueid","sid"}]
    need_date   = {"datadate","date","month_end","eom"}
    price_cands = {"prccm","prc","price","pmon","pindex","px"}
    ret_cands   = {"trt1m","ret","return","rtn","mret"}
    fic_cands   = {"fic","country","ctry","iso","iso2","iso3"}
    shr_cands   = {"cshocm","csho","shrout"}

    for _, r in cand.iterrows():
        schema, table = r["table_schema"], r["table_name"]
        cols = get_cols(db, schema, table)
        if not cols: continue
        has_id   = any(len(cols & s) > 0 for s in need_any_id)
        has_date = len(cols & need_date) > 0
        has_px   = len(cols & price_cands) > 0
        has_ret  = len(cols & ret_cands) > 0
        has_fic  = len(cols & fic_cands) > 0
        has_shr  = len(cols & shr_cands) > 0
        # 월말/월수익률/주식수 중 핵심 최소 요건: ID+DATE + (PRICE or RETURN)
        if has_id and has_date and (has_ret or has_px):
            return schema, table, cols
    return None

def find_daily_table(db):
    cand = list_tables(db, "comp_global_daily%")
    if cand.empty:
        return None
    cand["score"] = cand["table_name"].str.lower().apply(
        lambda s: (("sec" in s) * 2 + ("secd" in s) * 5 + ("d" in s) * 1)
    )
    cand = cand.sort_values("score", ascending=False)

    need_any_id = [{"gvkey","gvkeyx"}, {"iid","iid2","issueid","sid"}]
    need_date   = {"date","datadate"}
    price_cands = {"prccd","prc","price","px"}
    shr_cands   = {"cshoc","shrout","csho"}
    fic_cands   = {"fic","loc","cnty","country","iso","iso2","iso3"}

    for _, r in cand.iterrows():
        schema, table = r["table_schema"], r["table_name"]
        cols = get_cols(db, schema, table)
        if not cols: continue
        has_id   = any(len(cols & s) > 0 for s in need_any_id)
        has_date = len(cols & need_date) > 0
        has_px   = len(cols & price_cands) > 0
        has_shr  = len(cols & shr_cands) > 0
        if has_id and has_date and (has_px or has_shr):
            return schema, table, cols
    return None

# =========================
# 월말 데이터 로드
# =========================
def load_monthly_eom(db):
    """월별 테이블이 있으면 그대로 월말 데이터 사용(이미 월말 기준)."""
    spec = find_monthly_table(db)
    if spec is None:
        return None
    schema, table, cols = spec
    print(f"[monthly] using {schema}.{table}")

    id_gvkey = "gvkeyx" if "gvkeyx" in cols else ("gvkey" if "gvkey" in cols else None)
    id_iid   = next((c for c in ["iid","iid2","issueid","sid"] if c in cols), None)
    date_col = next((c for c in ["datadate","date","month_end","eom"] if c in cols), None)
    fic_col  = next((c for c in ["fic","country","ctry","iso","iso2","iso3"] if c in cols), None)

    px_col   = next((c for c in ["prccm","prc","price","pmon","pindex","px"] if c in cols), None)
    ret_col  = next((c for c in ["trt1m","ret","return","rtn","mret"] if c in cols), None)
    shr_col  = next((c for c in ["cshocm","csho","shrout"] if c in cols), None)
    tic_col  = "tic" if "tic" in cols else None
    isin_col = "isin" if "isin" in cols else None
    sedol_col= "sedol" if "sedol" in cols else None
    cur_col  = next((c for c in ["curcd","curcdd","currency","cur"] if c in cols), None)
    exch_col = next((c for c in ["exchg","excntry","exchcd","exch"] if c in cols), None)
    name_col = next((c for c in ["conm","name","security_name","issue_name"] if c in cols), None)

    if id_gvkey is None or date_col is None:
        return None

    sel = [f"{id_gvkey} AS gvkey", f"{date_col}::date AS eom"]
    if id_iid:   sel.append(f"{id_iid} AS iid")
    if fic_col:  sel.append(f"{fic_col} AS fic_raw")
    if tic_col:  sel.append(f"{tic_col} AS tic")
    if isin_col: sel.append(f"{isin_col} AS isin")
    if sedol_col:sel.append(f"{sedol_col} AS sedol")
    if cur_col:  sel.append(f"{cur_col} AS curcd")
    if exch_col: sel.append(f"{exch_col} AS exchg")
    if name_col: sel.append(f"{name_col} AS conm")
    if px_col:   sel.append(f"{px_col} AS px")
    if ret_col:  sel.append(f"{ret_col} AS ret_raw")
    if shr_col:  sel.append(f"{shr_col} AS csho")

    sql = f"""
        SELECT {", ".join(sel)}
        FROM {schema}.{table}
        WHERE {date_col} BETWEEN %s AND %s
    """
    df = db.raw_sql(sql, params=(START_DATE, END_DATE))
    if df.empty: return None

    df["eom"] = pd.to_datetime(df["eom"])
    # ISO3
    if "fic_raw" in df.columns:
        df["iso3"] = df.apply(lambda r: normalize_country_to_iso3(r, "fic_raw"), axis=1)
    else:
        df["iso3"] = None
    df = df[df["iso3"].isin(TARGET_ISO3)].copy()

    # mret: 제공되면 사용, 없으면 가격으로 복원
    if "ret_raw" in df.columns:
        df["mret"] = pd.to_numeric(df["ret_raw"], errors="coerce")
    elif "px" in df.columns:
        sort_keys = ["gvkey","iid","eom"] if "iid" in df.columns else ["gvkey","eom"]
        df = df.sort_values(sort_keys)
        grp = ["gvkey","iid"] if "iid" in df.columns else ["gvkey"]
        lag = df.groupby(grp)["px"].shift(1)
        df["mret"] = np.where(
            (lag.notna()) & (pd.to_numeric(lag, errors="coerce") != 0),
            pd.to_numeric(df["px"], errors="coerce")/pd.to_numeric(lag, errors="coerce") - 1,
            np.nan
        )
    else:
        df["mret"] = np.nan

    # mktcap: px × csho (가능시)
    if "px" in df.columns and "csho" in df.columns:
        df["mktcap"] = pd.to_numeric(df["px"], errors="coerce") * pd.to_numeric(df["csho"], errors="coerce")
    else:
        df["mktcap"] = np.nan

    return df

def load_daily_eom_sql(db):
    """일별 테이블에서 월말(해당 월 마지막 거래일)만 SQL 윈도우로 추출."""
    spec = find_daily_table(db)
    if spec is None:
        return None
    schema, table, cols = spec
    print(f"[daily→EOM] using {schema}.{table}")

    id_gvkey = "gvkeyx" if "gvkeyx" in cols else ("gvkey" if "gvkey" in cols else None)
    id_iid   = next((c for c in ["iid","iid2","issueid","sid"] if c in cols), None)
    date_col = "date" if "date" in cols else ("datadate" if "datadate" in cols else None)
    fic_col  = next((c for c in ["fic","loc","cnty","country","iso","iso2","iso3"] if c in cols), None)
    px_col   = next((c for c in ["prccd","prc","price","px"] if c in cols), None)
    shr_col  = next((c for c in ["cshoc","shrout","csho"] if c in cols), None)

    if (id_gvkey is None) or (id_iid is None) or (date_col is None) or (px_col is None) or (shr_col is None):
        return None

    sql = f"""
    WITH d AS (
      SELECT
        d.{id_gvkey} AS gvkey,
        d.{id_iid}   AS iid,
        {date_col}::date AS ddate,
        {px_col}::numeric AS px,
        {shr_col}::numeric AS csho,
        {"d."+fic_col if fic_col else "NULL"} AS fic_raw,
        DATE_TRUNC('month', {date_col})::date AS mkey,
        ROW_NUMBER() OVER (
          PARTITION BY d.{id_gvkey}, d.{id_iid}, DATE_TRUNC('month', {date_col})
          ORDER BY {date_col} DESC
        ) AS rn
      FROM {schema}.{table} d
      WHERE {date_col} BETWEEN %s AND %s
    )
    SELECT
      gvkey, iid,
      (mkey + INTERVAL '1 month' - INTERVAL '1 day')::date AS eom,
      px, csho, fic_raw
    FROM d
    WHERE rn = 1 AND px IS NOT NULL;
    """
    d = db.raw_sql(sql, params=(START_DATE, END_DATE))
    if d.empty: return None

    d["eom"] = pd.to_datetime(d["eom"])
    if "fic_raw" in d.columns:
        d["iso3"] = d.apply(lambda r: normalize_country_to_iso3(r, "fic_raw"), axis=1)
    else:
        d["iso3"] = None

    d = d[d["iso3"].isin(TARGET_ISO3)].copy()

    # mret(월말 가격 기준 연속비)
    sort_keys = ["gvkey","iid","eom"]
    d = d.sort_values(sort_keys)
    lag = d.groupby(["gvkey","iid"])["px"].shift(1)
    d["mret"] = np.where(
        (lag.notna()) & (pd.to_numeric(lag, errors="coerce") != 0),
        pd.to_numeric(d["px"], errors="coerce")/pd.to_numeric(lag, errors="coerce") - 1,
        np.nan
    )

    # mktcap
    if "csho" in d.columns:
        d["mktcap"] = pd.to_numeric(d["px"], errors="coerce") * pd.to_numeric(d["csho"], errors="coerce")
    else:
        d["mktcap"] = np.nan

    return d

# =========================
# 메인 파이프라인(월말만)
# =========================
def build_firm_monthly_eom(db):
    # 1) 월별(이미 월말) 우선 사용
    df = load_monthly_eom(db)

    # 2) 월별이 없거나 커버리지 낮으면 일별→월말로 대체
    if df is None or df.empty:
        print("[info] monthly table unavailable/empty → fallback to daily EOM extraction")
        df = load_daily_eom_sql(db)
        if df is None or df.empty:
            raise RuntimeError("월별/일별 어느 경로에서도 월말 데이터 추출에 실패했습니다.")
    else:
        # 월별에 주식수/가격 결측이 많아 mktcap 커버리지 낮다면,
        # 일별→월말을 보조로 써서 mktcap만 보강(동일 키 기준)
        cov = df["mktcap"].notna().mean() if "mktcap" in df.columns else 0.0
        if cov < 0.5:  # 필요 시 기준 조정 가능
            print(f"[info] monthly mktcap coverage={cov:.1%} → try daily EOM to backfill mktcap")
            deom = load_daily_eom_sql(db)
            if deom is not None and not deom.empty:
                keys = ["gvkey","iid","eom"] if "iid" in df.columns else ["gvkey","eom"]
                cols_to_merge = ["gvkey","iid","eom","mktcap"] if "iid" in deom.columns else ["gvkey","eom","mktcap"]
                df = df.merge(deom[cols_to_merge], on=keys, how="left", suffixes=("","_d"))
                # 우선 monthly mktcap, 없으면 daily mktcap
                if "mktcap" in df.columns and "mktcap_d" in df.columns:
                    df["mktcap"] = np.where(df["mktcap"].notna(), df["mktcap"], df["mktcap_d"])
                    df.drop(columns=["mktcap_d"], inplace=True)

    # 공통 라벨/정리
    iso3_to_name = {v:k for k,v in COUNTRY_NAME_TO_ISO3.items()}
    if "iso3" not in df.columns:
        df["iso3"] = None
    df["country"]    = df["iso3"].map(iso3_to_name)
    df["msci_group"] = df["iso3"].apply(lambda x: "Developed" if x in DEV_ISO3 else "Emerging")
    df["year"]  = df["eom"].dt.year
    df["month"] = df["eom"].dt.month

    keep = ["country","iso3","msci_group","gvkey","iid","eom","year","month",
            "mret","mktcap","tic","isin","sedol","curcd","exchg","conm"]
    keep = [c for c in keep if c in df.columns]
    sort_keys = ["country","gvkey","iid","eom"] if "iid" in df.columns else ["country","gvkey","eom"]
    out = df[keep].sort_values(sort_keys).reset_index(drop=True)

    # 간단 리포트
    firms = out[["gvkey","iid"]].drop_duplicates().shape[0] if "iid" in out.columns else out[["gvkey"]].drop_duplicates().shape[0]
    cov = out["mktcap"].notna().mean() if "mktcap" in out.columns else 0.0
    print(f"[build] rows={len(out):,} | firms={firms} | mktcap notnull={cov:.1%}")
    if "mret" in out.columns:
        print(f"[build] mret notnull={out['mret'].notna().mean():.1%}")

    return out

# =========================
# 메인
# =========================
def main():
    assert TARGET_ISO3 == {"AUS","FRA","DEU","JPN","GBR","BRA","CHN","IND","ZAF","TUR"}, \
        f"TARGET_ISO3이 지정한 10개와 다릅니다: {TARGET_ISO3}"

    print("0) WRDS 연결…")
    db = connect_wrds()

    print("1) 월말 데이터 파이프라인 시작… (Monthly 우선, 필요 시 Daily→EOM)")
    firm = build_firm_monthly_eom(db)

    if "mret" not in firm.columns or firm["mret"].notna().sum() == 0:
        raise RuntimeError("mret 전부 결측 — 가격 열 확인 필요")

    print("2) 통합 저장…")
    save_dual(firm, DATA_DIR / "hw1_firm_monthly_base.csv", DATA_DIR / "hw1_firm_monthly_base.parquet")

    print("3) 국가별 분할 저장…")
    split_dir = DATA_DIR / "countries"
    split_dir.mkdir(parents=True, exist_ok=True)
    if "country" not in firm.columns:
        firm["country"] = firm["iso3"]

    for iso3, sub in firm.groupby("iso3", dropna=True):
        cname = sub["country"].iloc[0] if len(sub) else iso3
        base  = f"{iso3}_{_slug(cname)}_firm_monthly"  # 월말만 들어있음
        csv_p = split_dir / f"{base}.csv"
        pq_p  = split_dir / f"{base}.parquet"
        # 정렬/저장
        sort_keys = ["gvkey","iid","eom"] if "iid" in sub.columns else ["gvkey","eom"]
        sub = sub.sort_values(sort_keys).reset_index(drop=True)
        save_dual(sub, csv_p, pq_p)

    print(f"[INFO] 국가별 파일은 {split_dir.resolve()} 폴더에 저장되었습니다.")

    print("\n=== 요약(국가별 관측치/기업수/커버리지) ===")
    g = (firm.groupby("country")
              .agg(n_obs=("mret","count"),
                   n_firms=("gvkey", pd.Series.nunique),
                   mktcap_coverage=("mktcap", lambda s: s.notna().mean()))
              .reset_index()
              .sort_values("country"))
    print(g.to_string(index=False))

if __name__ == "__main__":
    main()
