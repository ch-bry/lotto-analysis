"""로또 6/45 회차 데이터 수집.

동행복권 공식 API(common.do?method=getLottoNumber)는 비브라우저 클라이언트를
302로 차단하므로, 매주 토요일 추첨 직후 자동 갱신되는 smok95/lotto 미러를 쓴다.
회차별 JSON을 data/raw/ 에 캐시하고 data/draws.csv 로 정규화한다.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/smok95/lotto/master/results/{n}.json"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CSV = ROOT / "data" / "draws.csv"
TIMEOUT = 20
UA = "lotto-analysis/1.0 (+https://github.com/ch-bry)"


def _get(n: int) -> dict | None:
    """회차 n의 원본 JSON. 없으면 None."""
    req = urllib.request.Request(BASE.format(n=n), headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def fetch_round(n: int, *, refetch: bool = False) -> dict | None:
    """캐시 우선. 캐시에 없으면 내려받아 저장."""
    path = RAW / f"{n}.json"
    if path.exists() and not refetch:
        return json.loads(path.read_text(encoding="utf-8"))
    data = _get(n)
    if data is None:
        return None
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def _exists(n: int) -> bool:
    return (RAW / f"{n}.json").exists() or _get(n) is not None


def find_latest(start: int) -> int:
    """실재하는 마지막 회차. 지수 탐색으로 상한을 잡고 이분 탐색한다.

    한 칸씩 올라가면 최초 실행 시 1200회 이상을 순차 요청하게 되므로 쓰지 않는다.
    """
    lo = max(start, 1)
    if not _exists(lo):
        lo = 1
    step = 1
    hi = lo + step
    while _exists(hi):
        lo, step = hi, step * 2
        hi = lo + step
    # lo는 존재, hi는 부재 -> 경계를 이분 탐색
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _exists(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _division(divisions: list, i: int) -> tuple[int, int]:
    """i등(0-indexed)의 (당첨금, 당첨자수). 1회차 1등처럼 빈 객체면 (0, 0)."""
    if i >= len(divisions):
        return 0, 0
    d = divisions[i] or {}
    return int(d.get("prize", 0)), int(d.get("winners", 0))


def normalize(raw: dict) -> dict:
    nums = sorted(raw["numbers"])
    if len(nums) != 6 or len(set(nums)) != 6 or not all(1 <= v <= 45 for v in nums):
        raise ValueError(f"{raw['draw_no']}회 번호 이상: {raw['numbers']}")
    row = {
        "draw_no": int(raw["draw_no"]),
        "date": raw["date"][:10],
        **{f"n{i + 1}": v for i, v in enumerate(nums)},
        "bonus": int(raw["bonus_no"]),
        "sales": int(raw.get("total_sales_amount") or 0),
    }
    for i in range(5):
        prize, winners = _division(raw.get("divisions") or [], i)
        row[f"w{i + 1}_prize"] = prize
        row[f"w{i + 1}_winners"] = winners
    wc = raw.get("winners_combination") or {}
    # 자동/수동 집계는 830회차 무렵부터만 공개된다. 없으면 결측으로 남긴다.
    has_wc = bool(wc)
    row["auto"] = int(wc.get("auto", 0)) if has_wc else None
    row["semi_auto"] = int(wc.get("semi_auto", 0)) if has_wc else None
    row["manual"] = int(wc.get("manual", 0)) if has_wc else None
    return row


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    cached = sorted(int(p.stem) for p in RAW.glob("*.json") if p.stem.isdigit())
    known = max(cached) if cached else 0

    if known:
        print(f"캐시 {len(cached)}건 (최신 {known}회). 신규 회차 확인 중...")
    latest = find_latest(known or 1)

    missing = [n for n in range(1, latest + 1) if not (RAW / f"{n}.json").exists()]
    if missing:
        print(f"{len(missing)}개 회차 수집 중 (1~{latest})...")
        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(fetch_round, missing))

    rows = []
    for n in range(1, latest + 1):
        raw = json.loads((RAW / f"{n}.json").read_text(encoding="utf-8"))
        rows.append(normalize(raw))

    df = pd.DataFrame(rows).sort_values("draw_no").reset_index(drop=True)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False)
    print(f"저장: {CSV} ({len(df)}회차, 1~{latest}회, {df.date.iloc[0]}~{df.date.iloc[-1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
