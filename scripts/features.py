"""조합 특징 추출 — Group A(가설 검정), B(인기도), C(포트폴리오)가 공유한다.

모든 함수는 (N, 6) 정렬된 int 배열을 받아 (N,) 또는 (N, k) 배열을 돌려준다.
C(45,6)=8,145,060 전체 공간을 스코어링해야 하므로 전부 벡터화한다.
"""
from __future__ import annotations

import numpy as np

N_MAX = 45
PICK = 6
TOTAL_COMBOS = 8_145_060  # C(45, 6)

# 동행복권 용지 배치: 7열 × 7행 (43~45는 마지막 줄에 3개만)
SLIP_COLS = 7
PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43])
FIBS = np.array([1, 2, 3, 5, 8, 13, 21, 34])
SQUARES = np.array([1, 4, 9, 16, 25, 36])

_PRIME_MASK = np.zeros(N_MAX + 1, dtype=bool); _PRIME_MASK[PRIMES] = True
_FIB_MASK = np.zeros(N_MAX + 1, dtype=bool); _FIB_MASK[FIBS] = True
_SQUARE_MASK = np.zeros(N_MAX + 1, dtype=bool); _SQUARE_MASK[SQUARES] = True


def as_array(combos) -> np.ndarray:
    """(N, 6) int16 정렬 배열로 정규화."""
    a = np.atleast_2d(np.asarray(combos, dtype=np.int16))
    return np.sort(a, axis=1)


# ---------------------------------------------------------------- 기본 통계

def total(a): return a.sum(axis=1)
def spread(a): return a[:, -1] - a[:, 0]
def stdev(a): return a.std(axis=1)
def odd_count(a): return (a % 2 == 1).sum(axis=1)
def low_count(a): return (a <= 22).sum(axis=1)          # 고저 분할
def under31_count(a): return (a <= 31).sum(axis=1)      # 생일 편향
def under12_count(a): return (a <= 12).sum(axis=1)      # 월(月) 편향


def gaps(a):
    """정렬된 번호 사이의 간격 5개."""
    return np.diff(a, axis=1)


def min_gap(a): return gaps(a).min(axis=1)


def consecutive_pairs(a):
    """인접 차이가 1인 쌍의 개수 (0~5)."""
    return (gaps(a) == 1).sum(axis=1)


def has_consecutive(a): return consecutive_pairs(a) > 0


def max_run(a):
    """가장 긴 연속 번호 사슬의 길이."""
    g = (gaps(a) == 1)
    best = np.ones(len(a), dtype=np.int16)
    cur = np.ones(len(a), dtype=np.int16)
    for j in range(g.shape[1]):
        cur = np.where(g[:, j], cur + 1, 1)
        best = np.maximum(best, cur)
    return best


def zone_counts(a):
    """1-9 / 10-19 / 20-29 / 30-39 / 40-45 다섯 구간의 개수 (N, 5)."""
    edges = np.array([0, 9, 19, 29, 39, 45])
    return np.stack([((a > edges[i]) & (a <= edges[i + 1])).sum(axis=1)
                     for i in range(5)], axis=1)


def zone_spread(a):
    """구간 분포의 균등함. 값이 클수록 한쪽으로 몰린 조합."""
    return zone_counts(a).std(axis=1)


def tail_sum(a): return (a % 10).sum(axis=1)


def _n_distinct(v):
    """행별 서로 다른 값의 개수. 정렬 후 이웃 비교 (루프 없음)."""
    s = np.sort(v, axis=1)
    return 1 + (np.diff(s, axis=1) != 0).sum(axis=1)


def tail_unique(a):
    """서로 다른 끝수의 개수 (1~6). 낮으면 끝수가 겹치는 조합."""
    return _n_distinct(a % 10).astype(np.int16)


_IU, _JU = np.triu_indices(PICK, k=1)


def ac_value(a):
    """산술복잡도: 모든 쌍의 차이 중 서로 다른 값의 수 - 5. 범위 0~10."""
    diffs = np.abs(a[:, _IU].astype(np.int16) - a[:, _JU].astype(np.int16))
    return (_n_distinct(diffs) - (PICK - 1)).astype(np.int16)


def prime_count(a): return _PRIME_MASK[a].sum(axis=1)
def fib_count(a): return _FIB_MASK[a].sum(axis=1)
def square_count(a): return _SQUARE_MASK[a].sum(axis=1)
def mult3_count(a): return (a % 3 == 0).sum(axis=1)


# ------------------------------------------------------- 용지 기하 (마킹 패턴)

def slip_rc(a):
    """용지상 (행, 열). 7열 배치."""
    z = a - 1
    return z // SLIP_COLS, z % SLIP_COLS


def _max_category_count(v, n_cat, offset=0):
    """행별로 같은 카테고리에 몰린 최대 개수. 원핫 합산으로 벡터화."""
    cats = np.arange(n_cat, dtype=np.int16) - offset
    return (v[:, :, None] == cats[None, None, :]).sum(axis=1).max(axis=1).astype(np.int16)


def max_same_col(a):
    """같은 열에 몰린 최대 개수 — 세로줄 마킹 선호를 잡는다."""
    _, col = slip_rc(a)
    return _max_category_count(col, SLIP_COLS)


def max_same_row(a):
    """같은 행에 몰린 최대 개수 — 가로줄 마킹."""
    row, _ = slip_rc(a)
    return _max_category_count(row, 7)


def max_diagonal(a):
    """대각선(row-col, row+col) 정렬 최대 개수 — 대각 마킹."""
    row, col = slip_rc(a)
    # row-col 은 -6..6, row+col 은 0..12
    return np.maximum(_max_category_count(row - col, 13, offset=6),
                      _max_category_count(row + col, 13))


# ------------------------------------------------------------------ 특징 행렬

#: 인기도 회귀와 웹 표시에 함께 쓰는 조합 수준 특징
FEATURE_FNS = {
    "sum": total,
    "spread": spread,
    "stdev": stdev,
    "odd_count": odd_count,
    "low_count": low_count,
    "under31_count": under31_count,
    "under12_count": under12_count,
    "min_gap": min_gap,
    "consecutive_pairs": consecutive_pairs,
    "max_run": max_run,
    "zone_spread": zone_spread,
    "tail_sum": tail_sum,
    "tail_unique": tail_unique,
    "ac_value": ac_value,
    "prime_count": prime_count,
    "fib_count": fib_count,
    "square_count": square_count,
    "mult3_count": mult3_count,
    "max_same_col": max_same_col,
    "max_same_row": max_same_row,
    "max_diagonal": max_diagonal,
}

FEATURE_NAMES = list(FEATURE_FNS)


def feature_matrix(combos, names=None) -> np.ndarray:
    """(N, len(names)) float64 특징 행렬."""
    a = as_array(combos)
    names = names or FEATURE_NAMES
    return np.stack([np.asarray(FEATURE_FNS[k](a), dtype=np.float64) for k in names], axis=1)


def all_combinations(dtype=np.int8) -> np.ndarray:
    """C(45,6) 전체 조합 (8,145,060 × 6). 약 47MB."""
    from itertools import combinations
    return np.fromiter(
        (v for c in combinations(range(1, N_MAX + 1), PICK) for v in c),
        dtype=dtype, count=TOTAL_COMBOS * PICK,
    ).reshape(TOTAL_COMBOS, PICK)
