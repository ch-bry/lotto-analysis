"""Group C — N장 포트폴리오 최적화.

두 축을 동시에 개선한다. 1등 당첨 확률은 어떤 조합을 골라도 1/8,145,060 으로
동일하며 이 코드도 그것을 바꾸지 않는다.

  (1) 기대 수령액 — 인기도가 낮은 조합을 고르면 당첨 시 나눌 사람이 줄어든다.
  (2) 최소 1개 당첨 확률 — 티켓끼리 번호가 겹칠수록 "당첨/미당첨"이 같이 움직여
      전멸 확률이 커진다. 서로 번호를 적게 공유하도록 짜면 P(하나도 못 맞힘)이
      실제로 줄어든다. 이건 미신이 아니라 확률의 성질이다.

목적함수
    minimize  lam_pop * mean_i log Mult(t_i)
            + lam_cov * mean_{i<j} Pcollide(|t_i ∩ t_j|)
    Pcollide(k) 는 겹치는 번호가 k개인 두 티켓이 "둘 다 3개 이상 맞을" 확률로,
    몬테카를로로 한 번 표를 만들어 둔다.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
import popularity as P

ROOT = Path(__file__).resolve().parent.parent
N_MAX, PICK = F.N_MAX, F.PICK
M = F.TOTAL_COMBOS

#: 등수별 확률과 상금(5등 5,000원 / 4등 50,000원은 정액, 1~3등은 파리뮤추얼 평균값)
P_MATCH = {k: comb(PICK, k) * comb(N_MAX - PICK, PICK - k) / M for k in range(PICK + 1)}
P_ANY_PRIZE = sum(P_MATCH[k] for k in (3, 4, 5, 6))
CACHE = ROOT / "data" / "collide.json"


# --------------------------------------------------------- 충돌 확률표

def collide_table(n_sim=400_000, seed=11, use_cache=True) -> np.ndarray:
    """Pcollide[k] = 번호를 k개 공유하는 두 티켓이 둘 다 3개 이상 맞을 확률."""
    if use_cache and CACHE.exists():
        return np.asarray(json.loads(CACHE.read_text()))
    rng = np.random.default_rng(seed)
    out = np.zeros(PICK + 1)
    for k in range(PICK + 1):
        # 공유 k개 + 각자 6-k개 (서로 겹치지 않게) 를 구성
        both = 0
        chunk = 50_000
        done = 0
        while done < n_sim:
            n = min(chunk, n_sim - done)
            perm = np.argsort(rng.random((n, N_MAX)), axis=1) + 1
            shared = perm[:, :k]
            a = np.hstack([shared, perm[:, k:PICK]])
            b = np.hstack([shared, perm[:, PICK:2 * PICK - k]])
            draw = np.argsort(rng.random((n, N_MAX)), axis=1)[:, :PICK] + 1
            dm = np.zeros((n, N_MAX + 1), dtype=bool)
            np.put_along_axis(dm, draw, True, axis=1)
            ma = np.take_along_axis(dm, a, axis=1).sum(axis=1)
            mb = np.take_along_axis(dm, b, axis=1).sum(axis=1)
            both += int(((ma >= 3) & (mb >= 3)).sum())
            done += n
        out[k] = both / n_sim
    if use_cache:
        CACHE.write_text(json.dumps(out.tolist()))
    return out


# ------------------------------------------------------------------- 최적화

def _overlap_matrix(tickets: np.ndarray) -> np.ndarray:
    """(N, N) 티켓 쌍별 공유 번호 개수."""
    oh = np.zeros((len(tickets), N_MAX + 1), dtype=np.int8)
    np.put_along_axis(oh, tickets, 1, axis=1)
    return oh @ oh.T


def _onehot(tickets):
    oh = np.zeros((len(tickets), N_MAX + 1), dtype=np.int16)
    np.put_along_axis(oh, np.asarray(tickets, np.int64), 1, axis=1)
    return oh


def optimize(model, n_tickets, prev=None, *, lam_pop=1.0, lam_cov=None,
             n_rounds=700, batch=96, seed=0, collide=None, temp0=0.05):
    """시뮬레이티드 어닐링으로 N장을 고른다.

    한 번에 batch 개의 후보 수정안을 만들어 벡터 연산으로 한꺼번에 평가한다.
    수정되는 티켓은 하나뿐이므로 인기도와 겹침 비용을 증분으로 갱신한다.
    (전체를 매번 다시 계산하면 회차당 수 초가 걸려 백테스트가 불가능해진다.)
    """
    rng = np.random.default_rng(seed)
    collide = collide_table() if collide is None else np.asarray(collide)
    if lam_cov is None:
        lam_cov = 1.0 / collide[1]
    n = n_tickets
    n_pairs = n * (n - 1) / 2

    # 초기해: 45개 번호를 최대한 고르게 쓰도록 배치
    reps = int(np.ceil(n * PICK / N_MAX))
    pool = np.concatenate([rng.permutation(N_MAX) + 1 for _ in range(reps)])
    tickets = pool[:n * PICK].reshape(n, PICK)
    for i in range(n):
        while len(np.unique(tickets[i])) < PICK:
            tickets[i] = rng.choice(N_MAX, PICK, replace=False) + 1
    tickets = np.sort(tickets, axis=1).astype(np.int64)

    pop = model.log_multiplier(tickets, prev)
    oh = _onehot(tickets)
    ov = oh @ oh.T
    np.fill_diagonal(ov, 0)
    iu = np.triu_indices(n, k=1)
    cov_sum = float(collide[ov[iu]].sum())
    cur = lam_pop * pop.mean() + lam_cov * cov_sum / n_pairs
    best, best_tk = cur, tickets.copy()

    ar = np.arange(batch)
    for r in range(n_rounds):
        temp = temp0 * (1.0 - r / n_rounds) + 1e-6
        i = rng.integers(n, size=batch)
        j = rng.integers(PICK, size=batch)
        v = rng.integers(1, N_MAX + 1, size=batch)
        cand = tickets[i].copy()
        keep = ~(cand == v[:, None]).any(axis=1)        # 티켓 내 중복 방지
        if not keep.any():
            continue
        cand[ar, j] = v
        cand = np.sort(cand, axis=1)

        new_pop = model.log_multiplier(cand, prev)
        d_pop = (new_pop - pop[i]) / n

        cand_oh = _onehot(cand)
        new_ov = cand_oh @ oh.T                          # (batch, n)
        old_ov = ov[i]
        # 자기 자신과의 겹침은 쌍 합계에 들어가지 않으므로 제외한다.
        new_c = collide[new_ov]
        old_c = collide[old_ov]
        new_c[ar, i] = 0.0
        old_c[ar, i] = 0.0
        d_cov = (new_c - old_c).sum(axis=1) / n_pairs

        d_cost = lam_pop * d_pop + lam_cov * d_cov
        d_cost[~keep] = np.inf
        b = int(np.argmin(d_cost))
        if not np.isfinite(d_cost[b]):
            continue
        if d_cost[b] < 0 or rng.random() < np.exp(-d_cost[b] / temp):
            ti = int(i[b])
            tickets[ti] = cand[b]
            pop[ti] = new_pop[b]
            oh[ti] = cand_oh[b]
            row = (oh @ oh[ti]).astype(ov.dtype)
            row[ti] = 0
            ov[ti, :] = row
            ov[:, ti] = row
            cov_sum = float(collide[ov[iu]].sum())
            cur = lam_pop * pop.mean() + lam_cov * cov_sum / n_pairs
            if cur < best:
                best, best_tk = cur, tickets.copy()
    return best_tk, {"cost": float(best), "rounds": n_rounds, "batch": batch}


def random_tickets(n_tickets, rng):
    """자동선택(QuickPick) 기준선."""
    return np.sort(np.argsort(rng.random((n_tickets, N_MAX)), axis=1)[:, :PICK], axis=1) + 1


# ------------------------------------------------------------------ 정확 계산

def exact_evaluate_many(portfolios, model=None, prev=None, *, chunk=400_000):
    """여러 포트폴리오를 한 번의 전수 열거로 함께 평가한다.

    8,145,060가지를 포트폴리오마다 따로 도는 대신 티켓을 모두 이어붙여 한 번만
    돈다. 무작위 기준선을 여러 번 뽑아 평균낼 때 필요하다 — 무작위 포트폴리오
    하나만 쓰면 그 하나의 운이 기준선이 되어 버린다.
    """
    tks = [np.asarray(t, dtype=np.int64) for t in portfolios]
    sizes = [len(t) for t in tks]
    flat = np.vstack(tks)
    bounds = np.cumsum([0] + sizes)
    allc = F.all_combinations(np.int8)
    total = len(allc)

    stats = [{"any": 0, "jack": 0, "prizes": 0, "fixed": 0} for _ in tks]
    for s0 in range(0, total, chunk):
        block = allc[s0:s0 + chunk].astype(np.int64)
        c = len(block)
        dm = np.zeros((c, N_MAX + 1), dtype=bool)
        np.put_along_axis(dm, block, True, axis=1)
        matches = dm[:, flat].sum(axis=2)                  # (c, 총 티켓 수)
        for i in range(len(tks)):
            m = matches[:, bounds[i]:bounds[i + 1]]
            prized = m >= 3
            st = stats[i]
            st["any"] += int(prized.any(axis=1).sum())
            st["jack"] += int((m == PICK).any(axis=1).sum())
            st["prizes"] += int(prized.sum())
            st["fixed"] += int((m == 3).sum()) * 5_000 + int((m == 4).sum()) * 50_000

    out = []
    for tk, st in zip(tks, stats):
        n = len(tk)
        r = {
            "exact": True, "n_tickets": n,
            "distinct_tickets": int(len({tuple(t) for t in tk.tolist()})),
            "p_any_prize": st["any"] / total,
            "p_jackpot": st["jack"] / total,
            "mean_prizes": st["prizes"] / total,
            "mean_fixed_prize": st["fixed"] / total,
            "mean_overlap": float(_overlap_matrix(tk)[np.triu_indices(n, 1)].mean()),
            "distinct_numbers": int(len(np.unique(tk))),
        }
        if model is not None:
            mult = model.multiplier(tk, prev)
            r["mean_multiplier"] = float(mult.mean())
            r["payout_index"] = float((1.0 / mult).mean())
        out.append(r)
    return out


def random_baseline(model, prev, n_tickets, *, reps=10, seed=5):
    """자동선택 기준선. 무작위 포트폴리오 reps 개를 전수 열거해 평균낸다."""
    rng = np.random.default_rng(seed)
    rs = exact_evaluate_many([random_tickets(n_tickets, rng) for _ in range(reps)],
                             model, prev)
    keys = ["p_any_prize", "p_jackpot", "mean_prizes", "mean_fixed_prize",
            "mean_overlap", "mean_multiplier", "payout_index"]
    out = {k: float(np.mean([r[k] for r in rs])) for k in keys if k in rs[0]}
    out.update(exact=True, n_tickets=n_tickets, reps=reps,
               distinct_numbers=int(np.mean([r["distinct_numbers"] for r in rs])),
               p_any_prize_sd=float(np.std([r["p_any_prize"] for r in rs])))
    return out


def exact_evaluate(tickets, model=None, prev=None, *, chunk=400_000):
    """가능한 추첨 8,145,060가지를 전부 열거해 성능을 정확히 계산한다.

    몬테카를로로는 P(1등)=약 1.2e-6 을 검증할 수 없다(100만 회를 돌려도 기대
    적중이 1회 남짓이다). 전수 열거는 표본오차가 0 이므로 "P(1등)이 랜덤과
    동일하다"는 주장을 추정이 아니라 사실로 보일 수 있다.
    """
    tk = np.asarray(tickets, dtype=np.int64)
    n = len(tk)
    allc = F.all_combinations(np.int8)
    total = len(allc)
    tier_draws = np.zeros(PICK + 1, dtype=np.int64)   # 최고 매칭이 k인 추첨 수
    n_prizes = 0
    fixed = 0
    any_prize = 0
    for s in range(0, total, chunk):
        block = allc[s:s + chunk].astype(np.int64)
        c = len(block)
        dm = np.zeros((c, N_MAX + 1), dtype=bool)
        np.put_along_axis(dm, block, True, axis=1)
        matches = dm[:, tk].sum(axis=2)               # (c, n)
        best = matches.max(axis=1)
        tier_draws += np.bincount(best, minlength=PICK + 1)
        prized = matches >= 3
        any_prize += int(prized.any(axis=1).sum())
        n_prizes += int(prized.sum())
        fixed += int((matches == 3).sum()) * 5_000 + int((matches == 4).sum()) * 50_000

    out = {
        "exact": True,
        "n_tickets": n,
        "distinct_tickets": int(len({tuple(t) for t in tk.tolist()})),
        "p_any_prize": any_prize / total,
        "p_jackpot": int(tier_draws[PICK]) / total,
        "mean_prizes": n_prizes / total,
        "mean_fixed_prize": fixed / total,
        "best_tier_counts": tier_draws.tolist(),
        "mean_overlap": float(_overlap_matrix(tk)[np.triu_indices(n, 1)].mean()),
        "distinct_numbers": int(len(np.unique(tk))),
    }
    if model is not None:
        mult = model.multiplier(tk, prev)
        out["mean_multiplier"] = float(mult.mean())
        out["payout_index"] = float((1.0 / mult).mean())
    return out


def load_model():
    """인기도 모델 로드. 정식 산출물(web/data) 우선, 없으면 진단용(data) 사용."""
    for path in (ROOT / "web" / "data" / "popularity.json",
                 ROOT / "data" / "popularity.json"):
        if path.exists():
            return P.PopularityModel.from_dict(json.loads(path.read_text())["model"])
    raise SystemExit("popularity.json 이 없습니다. 먼저 scripts/build.py 를 실행하세요.")


def main():
    df = pd.read_csv(ROOT / "data" / "draws.csv")
    model = load_model()
    nxt = int(df.draw_no.max()) + 1
    prev = P.prev_draws(df, [int(df.draw_no.max()) + 1])[0]   # 다음 회차 기준

    collide = collide_table()
    print("두 티켓이 번호를 k개 공유할 때 '둘 다 당첨'될 확률")
    for k in range(PICK + 1):
        print(f"  k={k}: {collide[k]:.6f}")
    print(f"  (티켓이 독립이라면 {P_ANY_PRIZE**2:.6f}. k=0 에서 이보다 훨씬 낮은 것은")
    print(f"   번호가 안 겹치면 당첨번호 6개가 두 티켓에 나뉘어 '둘 다 당첨'이")
    print(f"   어려워지기 때문이다. 그만큼 '적어도 하나 당첨'은 올라간다.)")

    print(f"\n{nxt}회 포트폴리오 — 가능한 추첨 8,145,060가지 전수 열거 (표본오차 0)")
    print("=" * 100)
    print(f"{'구성':<24}{'P(1등)':>14}{'P(1개이상 당첨)':>17}{'평균 당첨수':>12}"
          f"{'평균 인기배수':>14}{'평균 겹침':>10}")
    print("-" * 100)
    for n in (10, 20):
        tk, _ = optimize(model, n, prev, seed=1)
        for label, r in (("자동선택 (10회 평균)", random_baseline(model, prev, n)),
                         ("최적화 포트폴리오", exact_evaluate(tk, model, prev))):
            print(f"{f'{n}장 {label}':<24}{r['p_jackpot']:>14.9f}{r['p_any_prize']:>17.5f}"
                  f"{r['mean_prizes']:>12.5f}{r['mean_multiplier']:>14.3f}"
                  f"{r['mean_overlap']:>10.3f}")
        print(f"{'':24}{n/M:>14.9f}{'<- 이론값 n/8145060':>17}")
        print("-" * 100)


if __name__ == "__main__":
    main()
