"""Group A — 번호 예측 통념 가설 검정.

핫넘버, 콜드넘버, 궁합수, "직전 회차 번호 제외" 같은 통념을 전부 같은
프로토콜로 검정한다.

프로토콜
    1. 각 가설의 검정통계량을 실제 1238회 데이터에서 계산한다.
    2. 균일 6/45 추첨을 1238회씩 n_sim 번 시뮬레이션해 귀무분포를 만든다.
       (해석적 근사 대신 몬테카를로를 쓰는 이유: 워크포워드 통계량처럼 분포를
        닫힌 형태로 쓸 수 없는 것이 많고, 회차 순서 구조까지 보존해야 한다.)
    3. 경험적 p값을 낸다.
    4. Benjamini-Hochberg 로 FDR 을 보정한다. 25개를 alpha=0.05 로 검정하면
       우연히 1.25개가 "유의"하게 나오므로 이 단계가 없으면 자기기만이 된다.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

import features as F

ROOT = Path(__file__).resolve().parent.parent
N_MAX, PICK = F.N_MAX, F.PICK
WARMUP = 100          # 워크포워드 통계량이 참조할 최소 과거 회차 수


# --------------------------------------------------------------- 검정통계량

def _counts(W):
    """번호별 출현 횟수 (45,)."""
    return np.bincount(W.ravel() - 1, minlength=N_MAX)


def _chisq_uniform(observed, expected):
    return float((((observed - expected) ** 2) / expected).sum())


def _rolling_topk(W, window, k=PICK, mode="hot"):
    """회차 t 직전 window 회의 출현빈도 상위/하위 k개를 t 에서 맞힌 개수 평균.

    미래 정보를 쓰지 않는 워크포워드 방식이다.
    """
    T = len(W)
    onehot = np.zeros((T, N_MAX), dtype=np.int16)
    np.put_along_axis(onehot, W - 1, 1, axis=1)
    cum = np.vstack([np.zeros(N_MAX, dtype=np.int32), np.cumsum(onehot, axis=0)])
    hits = np.empty(T - WARMUP)
    for i, t in enumerate(range(WARMUP, T)):
        lo = max(0, t - window)
        freq = cum[t] - cum[lo]
        # 동점은 번호 순서로 결정론적으로 끊는다(시뮬레이션에서도 동일 규칙).
        pick = np.argsort(freq if mode == "cold" else -freq, kind="stable")[:k]
        hits[i] = onehot[t, pick].sum()
    return float(hits.mean())


def _last_gap_topk(W, k=PICK):
    """가장 오래 안 나온 k개를 다음 회차에서 맞힌 개수 평균 (콜드넘버)."""
    T = len(W)
    onehot = np.zeros((T, N_MAX), dtype=np.int16)
    np.put_along_axis(onehot, W - 1, 1, axis=1)
    last = np.full(N_MAX, -1)
    for t in range(WARMUP):
        last[W[t] - 1] = t
    hits = np.empty(T - WARMUP)
    for i, t in enumerate(range(WARMUP, T)):
        pick = np.argsort(last, kind="stable")[:k]
        hits[i] = onehot[t, pick].sum()
        last[W[t] - 1] = t
    return float(hits.mean())


def _overlap_series(W, back=1):
    """각 회차가 back 회 전과 겹치는 번호 개수 시계열."""
    T = len(W)
    onehot = np.zeros((T, N_MAX), dtype=np.int16)
    np.put_along_axis(onehot, W - 1, 1, axis=1)
    return (onehot[back:] * onehot[:-back]).sum(axis=1)


def _union_overlap(W, n_back):
    """직전 n_back 회차 번호 합집합과 겹치는 개수 평균."""
    T = len(W)
    onehot = np.zeros((T, N_MAX), dtype=bool)
    np.put_along_axis(onehot, W - 1, True, axis=1)
    out = np.empty(T - n_back)
    for t in range(n_back, T):
        union = onehot[t - n_back:t].any(axis=0)
        out[t - n_back] = (onehot[t] & union).sum()
    return float(out.mean())


def _neighbor_hits(W):
    """직전 회차 번호 ±1 이 다음 회차에 나온 개수 평균."""
    T = len(W)
    onehot = np.zeros((T, N_MAX + 2), dtype=bool)
    np.put_along_axis(onehot, W, True, axis=1)      # 1..45 -> 인덱스 1..45
    hits = np.empty(T - 1)
    for t in range(1, T):
        nb = np.zeros(N_MAX + 2, dtype=bool)
        for d in (-1, 1):
            idx = np.clip(W[t - 1] + d, 0, N_MAX + 1)
            nb[idx] = True
        nb[0] = nb[N_MAX + 1] = False
        hits[t - 1] = (onehot[t] & nb).sum()
    return float(hits.mean())


def _markov_chisq(W):
    """직전 회차 번호 i -> 이번 회차 번호 j 전이표의 균일성 카이제곱."""
    T = len(W)
    idx = (W[:-1, :, None] - 1) * N_MAX + (W[1:, None, :] - 1)
    tab = np.bincount(idx.ravel(), minlength=N_MAX * N_MAX).astype(float)
    return _chisq_uniform(tab, tab.sum() / (N_MAX * N_MAX))


def _pair_chisq(W):
    """990개 번호쌍 동시출현 빈도의 균일성 카이제곱 (궁합수)."""
    iu, ju = np.triu_indices(PICK, k=1)
    a, b = W[:, iu], W[:, ju]
    lo, hi = np.minimum(a, b) - 1, np.maximum(a, b) - 1
    idx = lo * N_MAX + hi
    tab = np.bincount(idx.ravel(), minlength=N_MAX * N_MAX)
    mask = np.triu(np.ones((N_MAX, N_MAX), dtype=bool), k=1).ravel()
    tab = tab[mask].astype(float)
    return _chisq_uniform(tab, tab.sum() / len(tab))


_TRI = np.array(list(combinations(range(PICK), 3)))


def _triple_max(W):
    """모든 3수 조합 중 최다 동시출현 횟수 (극단값 검정)."""
    tri = np.sort(W[:, _TRI], axis=2) - 1
    idx = tri[:, :, 0] * N_MAX * N_MAX + tri[:, :, 1] * N_MAX + tri[:, :, 2]
    return float(np.bincount(idx.ravel()).max())


def _ks_vs_reference(values, ref_sorted):
    """경험분포 대 기준분포의 KS 통계량."""
    v = np.sort(values)
    cdf_ref = np.searchsorted(ref_sorted, v, side="right") / len(ref_sorted)
    n = len(v)
    upper = np.arange(1, n + 1) / n - cdf_ref
    lower = cdf_ref - np.arange(0, n) / n
    return float(max(upper.max(), lower.max()))


def _bucket_chisq(values, n_bucket):
    tab = np.bincount(np.asarray(values, dtype=int), minlength=n_bucket).astype(float)
    return _chisq_uniform(tab, tab.sum() / n_bucket)


def _halves_chisq(W):
    """전반부 vs 후반부 번호 빈도 차이 (시간 추세)."""
    h = len(W) // 2
    a, b = _counts(W[:h]).astype(float), _counts(W[h:]).astype(float)
    tot = a + b
    ea = tot * a.sum() / tot.sum()
    eb = tot * b.sum() / tot.sum()
    return float((((a - ea) ** 2) / ea + ((b - eb) ** 2) / eb).sum())


def _month_chisq(W, months):
    """월별 번호 빈도의 독립성 (계절 효과)."""
    tab = np.zeros((12, N_MAX))
    for m in range(12):
        sel = months == (m + 1)
        if sel.any():
            tab[m] = _counts(W[sel])
    row, col = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True)
    exp = row * col / tab.sum()
    exp = np.maximum(exp, 1e-9)
    return float((((tab - exp) ** 2) / exp).sum())


def _cusum_max(W):
    """번호 빈도의 최대 누적합 이탈 (볼세트 교체 같은 변화점 탐지)."""
    T = len(W)
    onehot = np.zeros((T, N_MAX))
    np.put_along_axis(onehot, W - 1, 1.0, axis=1)
    dev = np.cumsum(onehot - PICK / N_MAX, axis=0)
    return float(np.abs(dev).max() / np.sqrt(T))


def _ljung_box(x, lags=10):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    denom = (x ** 2).sum()
    q = 0.0
    for k in range(1, lags + 1):
        r = (x[k:] * x[:-k]).sum() / denom
        q += r ** 2 / (n - k)
    return float(n * (n + 2) * q)


# ------------------------------------------------------------------ 가설 정의

@dataclass
class Hypothesis:
    id: str
    title: str
    claim: str
    verdict_if_null: str
    fn: object
    alt: str = "greater"      # greater | two-sided


def build_hypotheses(ref):
    """ref: 균일 시뮬레이션에서 미리 만든 참조 분포 (KS 검정용)."""
    H = []
    add = H.append

    for n in (5, 10, 20, 50):
        add(Hypothesis(
            f"A1-{n}", f"핫넘버 (최근 {n}회)",
            f"최근 {n}회에 자주 나온 번호 6개가 다음 회차에 더 잘 나온다",
            "기대 적중 0.800개와 차이 없음",
            lambda W, c, n=n: _rolling_topk(W, n, mode="hot")))

    add(Hypothesis(
        "A2", "콜드넘버 (최장 미출현)",
        "가장 오래 안 나온 번호 6개는 '나올 때가 됐다'",
        "기대 적중 0.800개와 차이 없음", lambda W, c: _last_gap_topk(W)))

    add(Hypothesis(
        "A3", "직전 회차 번호 재출현",
        "직전 회차 당첨번호는 다음 회차에 잘 안 나오므로 제외해야 한다",
        "기대 재출현 0.800개와 차이 없음",
        lambda W, c: float(_overlap_series(W, 1).mean()), "two-sided"))

    for n in (2, 3, 5):
        add(Hypothesis(
            f"A4-{n}", f"직전 {n}회 누적 번호 제외",
            f"최근 {n}회에 나온 번호는 당분간 안 나온다",
            "누적 기대치와 차이 없음",
            lambda W, c, n=n: _union_overlap(W, n), "two-sided"))

    add(Hypothesis(
        "A5", "이웃수", "직전 회차 번호 ±1 이 다음 회차에 잘 나온다",
        "기대치와 차이 없음", lambda W, c: _neighbor_hits(W)))

    add(Hypothesis("A6", "마르코프 전이", "직전 번호가 다음 번호를 예측한다",
                   "전이표가 균일", lambda W, c: _markov_chisq(W)))
    add(Hypothesis("A7", "궁합수 (번호쌍)", "특정 두 번호는 함께 잘 나온다",
                   "990개 쌍의 빈도가 균일", lambda W, c: _pair_chisq(W)))
    add(Hypothesis("A8", "삼중 궁합", "특정 세 번호 조합이 유난히 자주 나온다",
                   "최다 트리플이 우연 수준", lambda W, c: _triple_max(W)))

    add(Hypothesis("A9", "합계 구간", "당첨번호 6개 합은 특정 구간에 몰린다",
                   "이론 분포와 일치",
                   lambda W, c: _ks_vs_reference(F.total(W), ref["sum"]), "two-sided"))
    add(Hypothesis("A10", "홀짝 비율", "홀짝 3:3 이 유난히 자주 나온다",
                   "이론 분포와 일치",
                   lambda W, c: _bucket_chisq(F.odd_count(W), 7)))
    add(Hypothesis("A11", "고저 비율", "1~22 와 23~45 가 3:3 으로 갈린다",
                   "이론 분포와 일치",
                   lambda W, c: _bucket_chisq(F.low_count(W), 7)))
    add(Hypothesis("A12", "끝수 합", "일의 자리 합이 특정 값에 몰린다",
                   "이론 분포와 일치",
                   lambda W, c: _ks_vs_reference(F.tail_sum(W), ref["tail_sum"]), "two-sided"))
    add(Hypothesis("A13", "구간 분포", "5개 구간에 고르게 흩어진다",
                   "이론 분포와 일치",
                   lambda W, c: _ks_vs_reference(F.zone_spread(W), ref["zone_spread"]), "two-sided"))
    add(Hypothesis("A14", "연속번호 출현율", "연속번호는 잘 안 나온다",
                   "이론 출현율(약 49.5%)과 일치",
                   lambda W, c: float(F.has_consecutive(W).mean()), "two-sided"))
    add(Hypothesis("A15", "AC값 (산술복잡도)", "AC값이 낮은 조합은 잘 안 나온다",
                   "이론 분포와 일치",
                   lambda W, c: _ks_vs_reference(F.ac_value(W), ref["ac_value"]), "two-sided"))
    add(Hypothesis("A16", "번호별 출현빈도", "특정 번호가 더 자주 나온다",
                   "45개 번호 빈도가 균일",
                   lambda W, c: _chisq_uniform(_counts(W).astype(float),
                                               len(W) * PICK / N_MAX)))
    add(Hypothesis("A17", "보너스 번호", "보너스 번호에 편향이 있다",
                   "45개 균일",
                   lambda W, c: _chisq_uniform(
                       np.bincount(c["bonus"] - 1, minlength=N_MAX).astype(float),
                       len(c["bonus"]) / N_MAX)))
    add(Hypothesis("A18", "시간 추세", "번호 편향이 시기별로 달라진다",
                   "전·후반 빈도 동일", lambda W, c: _halves_chisq(W)))
    add(Hypothesis("A19", "계절 효과", "월별로 잘 나오는 번호가 다르다",
                   "월과 번호가 독립", lambda W, c: _month_chisq(W, c["month"])))
    add(Hypothesis("A20", "볼세트 변화점", "볼세트를 바꾸면 번호 편향이 생긴다",
                   "누적 이탈이 랜덤워크 수준", lambda W, c: _cusum_max(W)))
    add(Hypothesis("A21", "이월수 자기상관", "이월 개수에 주기성이 있다",
                   "자기상관 없음",
                   lambda W, c: _ljung_box(_overlap_series(W, 1))))
    add(Hypothesis("A22", "번호 간격 패턴", "번호 사이 간격에 규칙이 있다",
                   "이론 분포와 일치",
                   lambda W, c: _ks_vs_reference(F.min_gap(W), ref["min_gap"]), "two-sided"))
    add(Hypothesis("A23", "직전 3개 이상 겹침", "직전 회차와 3개 이상 겹치는 일은 거의 없다",
                   "이론 확률과 일치",
                   lambda W, c: float((_overlap_series(W, 1) >= 3).mean()), "two-sided"))
    add(Hypothesis("A24", "소수·3의 배수", "소수나 3의 배수가 편중된다",
                   "이론 분포와 일치",
                   lambda W, c: _bucket_chisq(F.prime_count(W), 7)
                                + _bucket_chisq(F.mult3_count(W), 7)))
    add(Hypothesis("A25", "특수 수열", "피보나치·제곱수가 유난히 잘 나온다",
                   "이론 기대치와 일치",
                   lambda W, c: float(F.fib_count(W).mean() + F.square_count(W).mean()),
                   "two-sided"))
    return H


# ------------------------------------------------------------------- 실행부

def simulate(T, rng):
    """균일 6/45 추첨 T회."""
    return np.sort(np.argsort(rng.random((T, N_MAX)), axis=1)[:, :PICK], axis=1) + 1


def build_reference(rng, n=200_000):
    """KS 검정용 이론(균일) 분포 표본."""
    S = simulate(n, rng)
    return {"sum": np.sort(F.total(S)), "tail_sum": np.sort(F.tail_sum(S)),
            "zone_spread": np.sort(F.zone_spread(S)), "ac_value": np.sort(F.ac_value(S)),
            "min_gap": np.sort(F.min_gap(S))}


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg. (기각여부, q값) 반환."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out_q = np.empty(m)
    out_q[order] = q
    return out_q <= alpha, out_q


#: 시뮬레이션 참조 분포의 고정 시드. 워커마다 동일한 기준분포를 써야 한다.
REF_SEED = 20240101
#: 시뮬레이션을 나눌 작업 개수. 실행 머신의 코어 수와 무관하게 고정해야
#: 어디서 돌리든 같은 p값이 나온다. 워커 수는 이와 별개로 코어 수에 맞춘다.
N_JOBS = 12


def _null_batch(job):
    """워커 하나가 담당하는 시뮬레이션 묶음. (ge, le) 카운트를 돌려준다."""
    seed, n_batch, observed, T, months = job
    rng = np.random.default_rng(seed)
    hyps = build_hypotheses(build_reference(np.random.default_rng(REF_SEED)))
    observed = np.asarray(observed)
    ge = np.zeros(len(hyps))
    le = np.zeros(len(hyps))
    allnum = np.arange(1, N_MAX + 1)
    for _ in range(n_batch):
        Ws = simulate(T, rng)
        cs = {"bonus": np.array([rng.choice(np.setdiff1d(allnum, w)) for w in Ws]),
              "month": months}
        vals = np.array([h.fn(Ws, cs) for h in hyps])
        ge += vals >= observed
        le += vals <= observed
    return ge, le


def run(df, *, n_sim=10_000, seed=1, verbose=True, workers=None):
    W = df[[f"n{i}" for i in range(1, 7)]].to_numpy(np.int64)
    months = pd.to_datetime(df.date).dt.month.to_numpy()
    ctx = {"bonus": df.bonus.to_numpy(np.int64), "month": months}
    T = len(W)
    H = build_hypotheses(build_reference(np.random.default_rng(REF_SEED)))
    observed = np.array([h.fn(W, ctx) for h in H])

    # 작업 분할은 코어 수와 무관하게 고정한다. 워커 수에 따라 쪼개면 12코어
    # 노트북과 4코어 CI 러너가 서로 다른 난수열을 써서 p값이 달라진다.
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    n_job = min(N_JOBS, max(1, n_sim))
    sizes = [n_sim // n_job + (1 if i < n_sim % n_job else 0) for i in range(n_job)]
    jobs = [(seed * 1000 + i, sz, observed, T, months)
            for i, sz in enumerate(sizes) if sz > 0]

    ge = np.zeros(len(H))
    le = np.zeros(len(H))
    if workers == 1:
        for j in jobs:
            a, b = _null_batch(j)
            ge += a
            le += b
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (a, b) in enumerate(ex.map(_null_batch, jobs)):
                ge += a
                le += b
                if verbose:
                    print(f"    워커 {i + 1}/{len(jobs)} 완료", flush=True)

    pvals = []
    for i, h in enumerate(H):
        if h.alt == "greater":
            p = (ge[i] + 1) / (n_sim + 1)
        else:
            p = 2 * min((ge[i] + 1) / (n_sim + 1), (le[i] + 1) / (n_sim + 1))
        pvals.append(min(p, 1.0))
    pvals = np.array(pvals)
    reject, qvals = bh_fdr(pvals)

    return [{"id": h.id, "title": h.title, "claim": h.claim,
             "verdict_if_null": h.verdict_if_null, "alt": h.alt,
             "statistic": float(observed[i]), "p": float(pvals[i]),
             "q": float(qvals[i]), "raw_sig": bool(pvals[i] < 0.05),
             "fdr_sig": bool(reject[i])}
            for i, h in enumerate(H)]


def main():
    n_sim = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    df = pd.read_csv(ROOT / "data" / "draws.csv")
    print(f"Group A 가설 검정 — {len(df)}회차, 몬테카를로 {n_sim:,}회\n")
    res = run(df, n_sim=n_sim)

    print(f"\n{'ID':<7}{'가설':<24}{'통계량':>12}{'p값':>9}{'q값(FDR)':>11}  판정")
    print("-" * 78)
    for r in sorted(res, key=lambda x: x["p"]):
        mark = "★ 유의" if r["fdr_sig"] else ("· raw만" if r["raw_sig"] else "효과없음")
        print(f"{r['id']:<7}{r['title']:<24}{r['statistic']:>12.4f}"
              f"{r['p']:>9.4f}{r['q']:>11.4f}  {mark}")

    raw = sum(r["raw_sig"] for r in res)
    fdr = sum(r["fdr_sig"] for r in res)
    print("-" * 78)
    print(f"총 {len(res)}개 가설 중  raw p<0.05: {raw}개 (우연 기대 {len(res)*0.05:.1f}개)"
          f"  /  FDR 보정 후 생존: {fdr}개")

    out = ROOT / "data" / "hypotheses.json"
    out.write_text(json.dumps({"n_sim": n_sim, "n_draws": len(df), "results": res},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
