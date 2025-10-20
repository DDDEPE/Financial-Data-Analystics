import re
import pandas as pd
from .wrds_utils import list_tables, get_cols

def normalize_country_to_iso3(val, ISO2_TO_ISO3):
    if val is None: return None
    s = str(val).strip().upper()
    if re.fullmatch(r"[A-Z]{3}", s): return s
    if re.fullmatch(r"[A-Z]{2}", s): return ISO2_TO_ISO3.get(s)
    return ISO2_TO_ISO3.get(s[:2])

def find_price_monthly_table(db):
    cand = list_tables(db, "comp_global%")
    cand["score"] = cand["table_name"].str.lower().apply(
        lambda s: (
            (("sec" in s) or ("security" in s)) * 3 +
            (("secm" in s)) * 5 +
            (("m" in s or "mon" in s or "monthly" in s) * 2)
        )
    )
    cand = cand.sort_values("score", ascending=False)

    need_any_id = [{"gvkey","gvkeyx"}, {"iid","iid2","issueid","sid"}]
    need_date   = {"datadate","date","month_end","eom"}
    price_cands = {"prccm","prc","price","pmon","pindex","px"}
    ret_cands   = {"trt1m","ret","return","rtn","mret"}

    for _, r in cand.iterrows():
        schema, table = r["table_schema"], r["table_name"]
        cols = get_cols(db, schema, table)
        if not cols: continue
        has_id   = any(len(cols & s) > 0 for s in need_any_id)
        has_date = len(cols & need_date) > 0
        has_px   = len(cols & price_cands) > 0
        has_ret  = len(cols & ret_cands) > 0
        if has_id and has_date and (has_px or has_ret):
            return schema, table, cols
    raise RuntimeError("월별 가격/수익률 테이블을 찾지 못했습니다. (comp_global%.*sec*m*)")

def find_share_table_candidates(db):
    cand = list_tables(db, "comp_global%")
    rows = []
    for _, r in cand.iterrows():
        schema, table = r["table_schema"], r["table_name"]
        cols = get_cols(db, schema, table)
        hit = len(cols & {"cshocm","csho","shrout"})
        has_date = len(cols & {"datadate","date","month_end","eom"}) > 0
        if hit > 0 and has_date:
            rows.append((schema, table, hit, ",".join(sorted(cols & {"cshocm","csho","shrout"}))))
    if not rows:
        return pd.DataFrame(columns=["schema","table","hits","share_cols"])
    return pd.DataFrame(rows, columns=["schema","table","hits","share_cols"]).sort_values(
        ["hits","schema","table"], ascending=[False,True,True]
    )
# === (ADD) Dividend table discovery ===
def find_div_table_candidates(db):
    """
    Compustat Global 내 배당(주당 현금배당)을 담은 테이블 후보 탐색.
    - 컬럼 후보: dvps*, div*, cashdiv*, dvd*
    - 날짜: datadate/date/month_end/eom 중 하나
    """
    cand = list_tables(db, "comp_global%")
    rows = []
    div_name_keys = {"dvps", "dvpsx", "dvpspy", "dvpsxm", "div", "dvd", "cashdiv", "dvc"}
    date_keys = {"datadate", "date", "month_end", "eom", "exdt", "exdate"}
    id_keys = [{"gvkey","gvkeyx"}, {"iid","iid2","issueid","sid"}]

    for _, r in cand.iterrows():
        schema, table = r["table_schema"], r["table_name"]
        cols = get_cols(db, schema, table)
        if not cols:
            continue
        has_id = any(len(cols & k) > 0 for k in id_keys)
        has_date = len(cols & date_keys) > 0
        has_div = any(any(c.startswith(x) for c in cols) for x in div_name_keys) or (len(cols & div_name_keys) > 0)
        if has_id and has_date and has_div:
            # 어떤 배당 컬럼들이 있는지 기록
            div_cols = [c for c in cols if any(c.startswith(x) for x in div_name_keys)]
            rows.append((schema, table, len(div_cols), ",".join(sorted(div_cols))))
    if not rows:
        return pd.DataFrame(columns=["schema","table","n_divcols","div_cols"])
    return pd.DataFrame(rows, columns=["schema","table","n_divcols","div_cols"])\
             .sort_values(["n_divcols","schema","table"], ascending=[False, True, True])
