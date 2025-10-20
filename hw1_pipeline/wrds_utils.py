import os, sys
from pathlib import Path

def connect_wrds():
    try:
        import wrds
    except ImportError:
        print("pip install wrds pandas numpy python-dotenv"); sys.exit(1)

    # .env 로드
    try:
        from dotenv import load_dotenv
        for p in [Path.cwd()/".env", Path(__file__).with_name(".env")]:
            if p.exists():
                load_dotenv(p)
        load_dotenv()
    except Exception:
        pass

    user = os.getenv("WRDS_USERNAME") or os.getenv("WRDS_USER")
    pwd  = os.getenv("WRDS_PASSWORD") or os.getenv("PGPASSWORD")
    if not user or not pwd:
        raise RuntimeError("WRDS_USERNAME/WRDS_PASSWORD를 .env 또는 환경변수로 설정하세요.")
    os.environ["PGUSER"] = user
    os.environ["PGPASSWORD"] = pwd
    return wrds.Connection(wrds_username=user, wrds_password=pwd)

def list_tables(db, schema_like="comp_global%"):
    q = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_schema ILIKE %s AND table_type='BASE TABLE'
    """
    return db.raw_sql(q, params=(schema_like,))

def get_cols(db, schema, table):
    q = """SELECT column_name FROM information_schema.columns
           WHERE table_schema=%s AND table_name=%s"""
    cols = db.raw_sql(q, params=(schema, table))["column_name"].str.lower().tolist()
    return set(cols)
