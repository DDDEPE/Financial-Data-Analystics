import os
from contextlib import contextmanager

def _load_tqdm():
    try:
        from tqdm.auto import tqdm as _tqdm
        return _tqdm
    except Exception:
        return None

_TQDM = _load_tqdm()

def iter_progress(iterable, desc=None, total=None, disable=None):
    """tqdm 있으면 진행바, 없으면 원래 iterable 그대로."""
    if _TQDM is None:
        return iterable
    if disable is None:
        disable = (os.getenv("HW1_PROGRESS", "1") == "0")
    return _TQDM(iterable, desc=desc, total=total, leave=False, dynamic_ncols=True, disable=disable)

@contextmanager
def step(desc):
    """큰 단계 시작/끝 로그 출력."""
    print(f"[..] {desc}")
    try:
        yield
        print(f"[OK] {desc}")
    except Exception:
        print(f"[!!] {desc} 실패")
        raise
