# hw1_pipeline/loaders.py
import os
import numpy as np
import pandas as pd

from .config import START_DATE, END_DATE, TARGET_ISO3, ISO2_TO_ISO3
from .wrds_utils import get_cols, list_tables
from .progress import iter_progress
from .discovery import (
    find_price_monthly_table,
    find_share_table_candidates,
    normalize_country_to_iso3,
    find_div_table_candidates,  # 배당 테이블 탐색
)

# -------------------------------
# 0) 내부 설정 (튜닝 가능)
# -------------------------------
ATTRS_CHUNK = int(os.getenv("HW1_ATTRS_CHUNK", "5000"))  # attrs 청크 크기


# -------------------------------
# 1) 가격/수익률 베이스 로딩
# -------------------------------
def load_price_base(db):
    ps, pt, pcols = find_price_monthly_table(db)
    print(f"[price] using {ps}.{pt}")

    id_gvkey = "gvkeyx" if "gvkeyx" in pcols else ("gvkey" if "gvkey" in pcols else None)
    id_iid   = next((c for c in ["iid","iid2","issueid","sid"] if c in pcols), None)
    date_col = next((c for c in ["datadate","date","month_end","eom"] if c in pcols), None)
    fic_col  = next((c for c in ["fic","country","ctry","iso","iso2","iso3"] if c in pcols), None)
    px_col   = next((c for c in ["prccm","prc","price","pmon","pindex","px"] if c in pcols), None)
    ret_cols = [c for c in ["trt1m","ret","return","rtn","mret"] if c in pcols]

    sel = [f"{id_gvkey} AS gvkey", f"{date_col}::date AS eom"]
    if id_iid:  sel.append(f"{id_iid} AS iid")
    if fic_col: sel.append(f"{fic_col} AS fic_raw")
    if px_col:  sel.append(f"{px_col} AS px")
    for c in ret_cols: sel.append(c)
    for c in ["tic","isin","sedol","curcd","exchg","conm"]:
        if c in pcols: sel.append(c)

    sql = f"""
        SELECT {", ".join(sel)}
        FROM {ps}.{pt}
        WHERE {date_col} BETWEEN %s AND %s
    """
    df = db.raw_sql(sql, params=(START_DATE, END_DATE))
    df["eom"] = pd.to_datetime(df["eom"])

    # ISO3 normalize & filter 10 countries
    df["iso3"] = df["fic_raw"].map(lambda v: normalize_country_to_iso3(v, ISO2_TO_ISO3)) if "fic_raw" in df.columns else None
    df = df[df["iso3"].isin(TARGET_ISO3)].copy()

    # mret from table or reconstruct
    got_ret = False
    for c in ["trt1m","ret","return","rtn","mret"]:
        if c in df.columns:
            df["mret"] = pd.to_numeric(df[c], errors="coerce"); got_ret = True; break
    if not got_ret:
        if "px" not in df.columns:
            raise RuntimeError("가격/수익률 열이 없어 mret 계산 불가")
        key = ["gvkey"] + (["iid"] if "iid" in df.columns else [])
        df = df.sort_values(key + ["eom"])
        lag_px = df.groupby(key)["px"].shift(1)
        df["mret"] = np.where(
            (lag_px.notna()) & (pd.to_numeric(lag_px, errors="coerce") != 0),
            pd.to_numeric(df["px"], errors="coerce")/pd.to_numeric(lag_px, errors="coerce") - 1,
            np.nan
        )
    return df, (ps, pt, pcols)


# -------------------------------
# 2) 주식수 as-of + 기본 시가총액
# -------------------------------
def attach_shares_and_mktcap(db, price_df, price_meta):
    cand = find_share_table_candidates(db)
    if cand.empty:
        print("[진단] 주식수 테이블 후보를 찾지 못했습니다.")
        price_df["shr"] = np.nan; price_df["mktcap"] = np.nan
        return price_df, None

    best = None

    # 후보 테이블 루프 진행바
    for _, row in iter_progress(cand.iterrows(), desc="[shares] candidates", total=len(cand)):
        ss, st = row["schema"], row["table"]
        scols = get_cols(db, ss, st)

        s_id_gvkey = "gvkeyx" if "gvkeyx" in scols else ("gvkey" if "gvkey" in scols else None)
        s_id_iid   = next((c for c in ["iid","iid2","issueid","sid"] if c in scols), None)
        s_date_col = next((c for c in ["datadate","date","month_end","eom"] if c in scols), None)
        s_shr_col  = next((c for c in ["cshocm","csho","shrout"] if c in scols), None)
        if not (s_id_gvkey and s_date_col and s_shr_col): continue

        sel = [f"{s_id_gvkey} AS gvkey", f"{s_date_col}::date AS sdate", f"{s_shr_col} AS shr_raw"]
        if s_id_iid: sel.append(f"{s_id_iid} AS iid")
        sql = f"SELECT {', '.join(sel)} FROM {ss}.{st} WHERE {s_date_col} BETWEEN %s AND %s"
        sh = db.raw_sql(sql, params=("2019-01-01", END_DATE))  # as-of 넉넉
        sh["sdate"] = pd.to_datetime(sh["sdate"])

        scale = 1e6 if s_shr_col in {"csho","cshocm"} else 1.0
        sh["shr"] = pd.to_numeric(sh["shr_raw"], errors="coerce") * scale

        # as-of merge by gvkey(+iid)
        out = []
        grp_keys = ["gvkey"] + (["iid"] if ("iid" in price_df.columns and "iid" in sh.columns) else [])
        n_keys = price_df[grp_keys].drop_duplicates().shape[0]
        for k, g in iter_progress(price_df.groupby(grp_keys, sort=False),
                                  desc=f"[asof] {ss}.{st} by key", total=n_keys):
            g2 = g.copy().sort_values("eom")
            kk = k if isinstance(k, tuple) else (k,)

            mask = np.ones(len(sh), dtype=bool)
            for i, col in enumerate(grp_keys):
                mask &= (sh[col] == kk[i])
            sh_k = sh[mask].sort_values("sdate")

            if sh_k.empty:
                g2["shr"] = np.nan
            else:
                joined = pd.merge_asof(
                    g2[["eom"]].sort_values("eom"),
                    sh_k[["sdate","shr"]].rename(columns={"sdate":"keydate"}),
                    left_on="eom", right_on="keydate",
                    direction="backward", tolerance=pd.Timedelta(days=400)
                )
                g2["shr"] = joined["shr"].values
            out.append(g2)
        merged = pd.concat(out, ignore_index=True)

        px  = pd.to_numeric(merged.get("px"), errors="coerce")
        shr = pd.to_numeric(merged.get("shr"), errors="coerce")
        merged["mktcap"] = px * shr

        mcap = pd.to_numeric(merged["mktcap"], errors="coerce")
        valid = mcap.notna(); denom = int(valid.sum())
        pos_ratio = ((mcap > 0) & valid).sum() / denom if denom > 0 else 0.0
        print(f"[asof 시도] {ss}.{st} | shares_col={s_shr_col} | mktcap>0 비율={pos_ratio:.2%}")

        if (best is None) or (pos_ratio > best[0]):
            best = (pos_ratio, merged, (ss, st, s_shr_col))
        if pos_ratio >= 0.30:
            return merged, (ss, st, s_shr_col)

    if best is not None:
        print("[알림] 최고 비율 후보를 채택합니다.")
        return best[1], best[2]

    print("[경고] shares as-of도 실패. mktcap 결측일 수 있습니다.")
    price_df["mktcap"] = np.nan
    return price_df, None


# -------------------------------
# 3) 회사 속성 로딩 (키 기반 부분조회)
# -------------------------------
def _pick_best_attr_table(db):
    cand = list_tables(db, "comp_global%")
    pref = []
    for _, r in cand.iterrows():
        s, t = r["table_schema"], r["table_name"]
        low = t.lower()
        if any(k in low for k in ["secd","security","company","master"]):
            cols = get_cols(db, s, t)
            if not cols:
                continue
            if ("gvkey" in cols) or ("gvkeyx" in cols):
                pref.append((s, t, cols))
    def score(cols):
        want = {"conm","tic","isin","sedol","sic","gsector","ggroup","gind","subind",
                "curcd","exchg","loc","fic","cout","iid"}
        base = 100 if ("iid" in cols or "secd" in cols) else 0
        return base + len(cols & want)
    if not pref:
        return None
    pref.sort(key=lambda x: score(x[2]), reverse=True)
    return pref[0]  # (schema, table, cols)

def _build_attr_select(schema, table, cols, with_iid: bool):
    id_gvkey = "gvkeyx" if "gvkeyx" in cols else "gvkey"
    sel = [f"{id_gvkey} AS gvkey"]
    table_has_iid = ("iid" in cols)
    if with_iid and table_has_iid:
        sel.append("iid")
    for c in ["conm","tic","isin","sedol","sic","gsector","ggroup","gind","subind",
              "curcd","exchg","loc","fic","cout"]:
        if c in cols: sel.append(c)
    return f"SELECT {', '.join(sel)} FROM {schema}.{table}", table_has_iid

def _fetch_attrs_by_keys(db, schema, table, cols, keys_df, with_iid_in_out):
    out = []
    total = len(keys_df)
    select_clause, table_has_iid = _build_attr_select(schema, table, cols, with_iid_in_out)
    with_iid = (with_iid_in_out and table_has_iid)

    for i in iter_progress(range(0, total, ATTRS_CHUNK), desc="[attrs] fetch chunks",
                           total=(total + ATTRS_CHUNK - 1)//ATTRS_CHUNK):
        sl = keys_df.iloc[i:i+ATTRS_CHUNK]
        if with_iid:
            vals = ",".join([f"('{str(a)}','{str(b)}')" for a,b in sl[["gvkey","iid"]].itertuples(index=False, name=None)])
            key_cols = "(gvkey,iid)"
            join_on = "USING (gvkey,iid)"
        else:
            vals = ",".join([f"('{str(a)}')" for a in sl["gvkey"]])
            key_cols = "(gvkey)"
            join_on = "USING (gvkey)"
        cte = f"WITH k{key_cols} AS (VALUES {vals}) "
        sql = cte + select_clause + f" JOIN k {join_on}"
        part = db.raw_sql(sql)
        out.append(part)

    if not out:
        return pd.DataFrame(columns=["gvkey"] + (["iid"] if with_iid else []))
    df = pd.concat(out, ignore_index=True)
    on = ["gvkey"] + (["iid"] if ("iid" in df.columns) else [])
    return df.sort_values(on).drop_duplicates(on, keep="first").reset_index(drop=True)

def load_company_attrs(db, price_df):
    """
    1) price_df에 속성열이 충분하면 재활용(빠름)
    2) 부족하면 price_df의 키로만 서버 부분조회(청크)
    """
    try:
        have = [c for c in ["conm","tic","isin","sedol","curcd","exchg"] if c in price_df.columns]
        if len(have) >= 3:
            print(f"[attrs] price base already has columns: {have} → 별도 로딩 생략")
            cols = ["gvkey"] + (["iid"] if "iid" in price_df.columns else []) + have
            return price_df[cols].drop_duplicates(cols, keep="first")

        picked = _pick_best_attr_table(db)
        if not picked:
            print("[attrs] 후보 테이블 없음 → 스킵")
            return pd.DataFrame(columns=["gvkey"])

        s, t, cols = picked
        print(f"[attrs] using {s}.{t} (filtered by keys)")
        key_cols = ["gvkey"] + (["iid"] if "iid" in price_df.columns else [])
        keys_df = price_df[key_cols].drop_duplicates().reset_index(drop=True)

        return _fetch_attrs_by_keys(db, s, t, cols, keys_df, with_iid_in_out=("iid" in key_cols))

    except Exception as e:
        print(f"[attrs][경고] 속성 로딩 실패 → 스킵합니다. 이유: {e}")
        return pd.DataFrame(columns=["gvkey"])


# -------------------------------
# 4) 펀더멘털 as-of 로딩
# -------------------------------
def load_funda_asof(db):
    cand = list_tables(db, "comp_global%")
    pick = []
    for _, r in cand.iterrows():
        s, t = r["table_schema"], r["table_name"]
        low = t.lower()
        if any(k in low for k in ["funda", "fundq", "fund", "fundamental", "fin"]):
            cols = get_cols(db, s, t)
            if ("datadate" in cols) and (("gvkey" in cols) or ("gvkeyx" in cols)):
                pick.append((s, t, cols))
    if not pick:
        print("[funda] fundamentals 후보 없음")
        return pd.DataFrame(columns=["gvkey","datadate"])

    pick.sort(key=lambda x: len(x[2]), reverse=True)
    s, t, cols = pick[0]
    print(f"[funda] using {s}.{t}")
    id_gvkey = "gvkeyx" if "gvkeyx" in cols else "gvkey"
    want = [
        "sale","sales","revt","at","ceq","lt","ebit","oibdp","ni","capx",
        "prccq","prcc_f","prcc","cshoq","csho","mkvaltq","mkvalt"
    ]
    got  = [c for c in want if c in cols]
    sel = [f"{id_gvkey} AS gvkey", "datadate::date AS datadate"] + got
    sql = f"""
        SELECT {", ".join(sel)}
        FROM {s}.{t}
        WHERE datadate BETWEEN %s AND %s
    """
    f = db.raw_sql(sql, params=("2019-01-01", END_DATE))
    f["datadate"] = pd.to_datetime(f["datadate"])
    return f.sort_values(["gvkey","datadate"]).drop_duplicates(["gvkey","datadate"], keep="last")


# -------------------------------
# 5) 배당 월집계 + MRET 재구성
# -------------------------------
def _choose_div_columns(cols):
    """
    배당(주당 현금배당) 컬럼 우선순위.
    dvpsxm > dvpsx_f > dvpsx > dvps > dvpspy > dvpspq > (fallback: div/dvd/dvc 등 접두)
    """
    pref = ["dvpsxm", "dvpsx_f", "dvpsx", "dvps", "dvpspy", "dvpspq"]
    fallback_prefix = ["div", "dvd", "cashdiv", "dvc"]

    for p in pref:
        if p in cols:
            return p
    for p in pref:
        cand = [c for c in cols if c.startswith(p)]
        if cand:
            return cand[0]
    for fx in fallback_prefix:
        cand = [c for c in cols if c.startswith(fx)]
        if cand:
            return cand[0]
    return None

def load_dividends_monthly(db, price_df):
    """
    가격 베이스(price_df: gvkey,(iid),eom,px)를 기준으로 배당을 월말 단위로 합산하여 붙이고,
    mret_recon = (px_t + divps_m_t - px_{t-1}) / px_{t-1} 를 계산해 반환.
    """
    cand = find_div_table_candidates(db)
    if cand.empty:
        print("[div] 배당 테이블 후보 없음 → 배당 없이 진행")
        out = price_df.copy()
        out["divps_m"] = np.nan
        key = ["gvkey"] + (["iid"] if "iid" in out.columns else [])
        out = out.sort_values(key + ["eom"])
        lag_px = out.groupby(key)["px"].shift(1)
        out["mret_recon"] = np.where(
            (lag_px.notna()) & (pd.to_numeric(lag_px, errors="coerce") != 0),
            (pd.to_numeric(out["px"], errors="coerce") - pd.to_numeric(lag_px, errors="coerce")) / pd.to_numeric(lag_px, errors="coerce"),
            np.nan
        )
        return out, None

    ss, st = cand.iloc[0]["schema"], cand.iloc[0]["table"]
    dcols = set(get_cols(db, ss, st))
    date_col = next((c for c in ["datadate","date","month_end","eom","exdt","exdate"] if c in dcols), None)
    id_gvkey = "gvkeyx" if "gvkeyx" in dcols else ("gvkey" if "gvkey" in dcols else None)
    id_iid   = next((c for c in ["iid","iid2","issueid","sid"] if c in dcols), None)
    div_col  = _choose_div_columns(dcols)
    if not (date_col and id_gvkey and div_col):
        print(f"[div] {ss}.{st} 에서 필수 컬럼 부족 → 배당 없이 진행")
        out = price_df.copy()
        out["divps_m"] = np.nan
        key = ["gvkey"] + (["iid"] if "iid" in out.columns else [])
        out = out.sort_values(key + ["eom"])
        lag_px = out.groupby(key)["px"].shift(1)
        out["mret_recon"] = np.where(
            (lag_px.notna()) & (pd.to_numeric(lag_px, errors="coerce") != 0),
            (pd.to_numeric(out["px"], errors="coerce") - pd.to_numeric(lag_px, errors="coerce")) / pd.to_numeric(lag_px, errors="coerce"),
            np.nan
        )
        return out, None

    print(f"[div] using {ss}.{st} | date={date_col} | div={div_col} | iid={bool(id_iid)}")

    sel = [f"{id_gvkey} AS gvkey", f"{date_col}::date AS ddate", f"{div_col} AS div_raw"]
    if id_iid:
        sel.append(f"{id_iid} AS iid")
    sql = f"""
        SELECT {", ".join(sel)}
        FROM {ss}.{st}
        WHERE {date_col} BETWEEN %s AND %s
    """
    dv = db.raw_sql(sql, params=(START_DATE, END_DATE))
    dv["ddate"] = pd.to_datetime(dv["ddate"])
    dv["divps"] = pd.to_numeric(dv["div_raw"], errors="coerce")

    # 월말 버킷으로 합산
    dv["eom"] = dv["ddate"] + pd.offsets.MonthEnd(0)
    group_keys = ["gvkey","eom"] + (["iid"] if ("iid" in dv.columns and "iid" in price_df.columns) else [])
    dv_m = dv.groupby(group_keys, as_index=False)["divps"].sum().rename(columns={"divps":"divps_m"})

    # 가격 DF와 병합
    out = price_df.merge(dv_m, on=group_keys, how="left")
    key = ["gvkey"] + (["iid"] if "iid" in out.columns else [])
    out = out.sort_values(key + ["eom"])
    lag_px = out.groupby(key)["px"].shift(1)
    divps = pd.to_numeric(out.get("divps_m"), errors="coerce").fillna(0.0)
    px    = pd.to_numeric(out.get("px"), errors="coerce")

    out["mret_recon"] = np.where(
        (lag_px.notna()) & (pd.to_numeric(lag_px, errors="coerce") != 0),
        (px + divps - pd.to_numeric(lag_px, errors="coerce")) / pd.to_numeric(lag_px, errors="coerce"),
        np.nan
    )
    return out, (ss, st, div_col)
