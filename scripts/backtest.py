"""검증 3·4 — 워크포워드 백테스트와 몬테카를로.

세 가지를 정직하게 확인한다.

  1. 거짓양성 통제: 균일 난수로 만든 가짜 이력에 Group A 파이프라인을 통째로
     돌려 "효과 있음"이 튀어나오지 않는지 본다.
  2. 포트폴리오 몬테카를로: P(1등)이 랜덤과 동일한지(다르면 버그), P(1개 이상
     당첨)과 기대 수령액이 실제로 오르는지.
  3. 워크포워드: 회차 t 를 예측할 때 t-1 까지의 데이터만 쓴다. 실제 회차에
     대해 우리 방식 N장과 자동선택 N장을 비교한다. ROI 는 양쪽 다 약 -50%다.
"""
from __future__ import annotations

import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd

import hypotheses as H
import popularity as P
import portfolio as PF

ROOT = Path(__file__).resolve().parent.parent
TICKET_PRICE = 1000

#: 등수별 당첨 확률. 어떤 조합을 고르든 동일하다.
P6 = 1 / PF.M                                   # 1등: 6개
P5B = 6 / PF.M                                  # 2등: 5개 + 보너스
P5 = (6 * 39 - 6) / PF.M                        # 3등: 5개 (보너스 제외)
P4 = PF.P_MATCH[4]                              # 4등: 4개
P3 = PF.P_MATCH[3]                              # 5등: 3개


def _prize_for(row, matches, bonus_hit):
    if matches == 6:
        return float(row.w1_prize)
    if matches == 5 and bonus_hit:
        return float(row.w2_prize)
    if matches == 5:
        return float(row.w3_prize)
    if matches == 4:
        return float(row.w4_prize)
    if matches == 3:
        return float(row.w5_prize)
    return 0.0


def score_tickets(tickets, row):
    """실제 회차 결과로 티켓들을 채점. (총 상금, 당첨 티켓 수, 매칭 분포)."""
    win = set(int(row[f"n{i}"]) for i in range(1, 7))
    bonus = int(row.bonus)
    total, n_win = 0.0, 0
    dist = {k: 0 for k in range(7)}
    for t in tickets:
        s = set(int(v) for v in t)
        m = len(s & win)
        dist[m] += 1
        p = _prize_for(row, m, bonus in s)
        if p > 0:
            n_win += 1
            total += p
    return total, n_win, dist


def analytic_expected_return(df, rounds):
    """회차별 실제 상금 구조로 계산한 티켓 1장의 해석적 기대 회수액.

    등수별 당첨 확률은 어떤 조합을 고르든 동일하므로 이 값도 두 방식이 같다.
    (인기도 최적화는 1등 '1인당' 금액을 올리지만, 아래 계산은 그 회차에 실제로
    지급된 금액을 그대로 쓰므로 그 효과는 반영하지 않는다 — 보수적인 기준선이다.)
    """
    sub = df[df.draw_no.isin(rounds)]
    per = (P6 * sub.w1_prize + P5B * sub.w2_prize + P5 * sub.w3_prize
           + P4 * sub.w4_prize + P3 * sub.w5_prize)
    return float(per.mean())


def walk_forward(df, *, n_tickets=10, start=1000, refit_every=26, n_reps=5,
                 sa_rounds=500, cache=None, seed=0, verbose=True):
    """회차 t 예측에 t-1 까지만 사용하는 워크포워드 백테스트.

    우리 방식과 자동선택 모두 회차마다 n_reps 번 반복해 평균을 낸다. 반복 수를
    맞추지 않으면 (예: 자동선택만 여러 번 평균) 희귀 등수의 기댓값이 한쪽에만
    반영되어 ROI 비교가 체계적으로 왜곡된다.
    """
    cache = cache or P.DesignCache(df)
    collide = PF.collide_table()
    rng = np.random.default_rng(seed)
    rounds = df[df.draw_no > start].reset_index(drop=True)

    model = None
    last_fit = -10 ** 9
    rows = []
    for i, row in rounds.iterrows():
        t = int(row.draw_no)
        if t - last_fit >= refit_every:
            model = cache.fit(train_to=t - 1)       # 미래 정보 차단
            last_fit = t
        prev = P.prev_draws(df, [t])[0]

        ours_tot, ours_win, ours_mult = [], [], []
        rnd_tot, rnd_win = [], []
        for rep in range(n_reps):
            tk, _ = PF.optimize(model, n_tickets, prev, n_rounds=sa_rounds,
                                seed=t * 100 + rep, collide=collide)
            a, b, _ = score_tickets(tk, row)
            ours_tot.append(a)
            ours_win.append(b)
            ours_mult.append(float(model.multiplier(tk, prev).mean()))

            rt = PF.random_tickets(n_tickets, rng)
            c, d, _ = score_tickets(rt, row)
            rnd_tot.append(c)
            rnd_win.append(d)

        rows.append({
            "draw_no": t,
            "ours_prize": float(np.mean(ours_tot)),
            "ours_wins": float(np.mean(ours_win)),
            "ours_any": float(np.mean([w > 0 for w in ours_win])),
            "ours_multiplier": float(np.mean(ours_mult)),
            "rand_prize": float(np.mean(rnd_tot)),
            "rand_wins": float(np.mean(rnd_win)),
            "rand_any": float(np.mean([w > 0 for w in rnd_win])),
        })
        if verbose and (i + 1) % 25 == 0:
            print(f"    {t}회 완료 ({i + 1}/{len(rounds)})", flush=True)
    return pd.DataFrame(rows)


def summarize(bt, n_tickets, df=None, n_reps=5):
    spend = len(bt) * n_tickets * TICKET_PRICE
    analytic = (analytic_expected_return(df, bt.draw_no.tolist()) if df is not None else None)
    return {
        "n_rounds": int(len(bt)),
        "n_tickets": n_tickets,
        "n_reps": n_reps,
        "spend": spend,
        # 등수별 확률이 동일하므로 두 방식의 기대 회수액도 동일하다.
        "analytic_return_per_ticket": analytic,
        "analytic_roi": (analytic / TICKET_PRICE - 1) if analytic is not None else None,
        "ours": {
            "prize": float(bt.ours_prize.sum()),
            "roi": float(bt.ours_prize.sum() / spend - 1),
            "any_rate": float(bt.ours_any.mean()),
            "wins": float(bt.ours_wins.sum()),
            "mean_multiplier": float(bt.ours_multiplier.mean()),
        },
        "random": {
            "prize": float(bt.rand_prize.sum()),
            "roi": float(bt.rand_prize.sum() / spend - 1),
            "any_rate": float(bt.rand_any.mean()),
            "wins": float(bt.rand_wins.sum()),
            "mean_multiplier": 1.0,
        },
    }


def exact_compare(model, prev, n_tickets, *, seed=5):
    """검증 4 — 가능한 추첨 8,145,060가지 전수 열거.

    P(1등)은 티켓 집합과 무관하게 (서로 다른 티켓 수)/8,145,060 이어야 한다.
    몬테카를로로는 이 크기(1.2e-6)를 검증할 수 없어 전수 열거를 쓴다.
    """
    tk, _ = PF.optimize(model, n_tickets, prev, seed=1)
    opt = PF.exact_evaluate(tk, model, prev)
    base = PF.random_baseline(model, prev, n_tickets, seed=seed)
    theo = n_tickets / PF.M
    assert abs(opt["p_jackpot"] - theo) < 1e-15, "P(1등)이 이론값과 다르다 — 버그"
    assert abs(base["p_jackpot"] - theo) < 1e-15
    return {"n_tickets": n_tickets, "optimized": opt, "random": base,
            "theoretical_p_jackpot": theo, "tickets": tk.tolist()}


def false_positive_check(n_sim=2000, seed=99, n_draws=1238):
    """검증 3 — 가짜(균일 난수) 이력에 Group A 를 돌려 거짓양성을 확인."""
    rng = np.random.default_rng(seed)
    W = H.simulate(n_draws, rng)
    fake = pd.DataFrame({
        **{f"n{i+1}": W[:, i] for i in range(6)},
        "bonus": [rng.choice(np.setdiff1d(np.arange(1, 46), w)) for w in W],
        "date": pd.date_range("2002-12-07", periods=n_draws, freq="7D").strftime("%Y-%m-%d"),
        "draw_no": np.arange(1, n_draws + 1),
    })
    res = H.run(fake, n_sim=n_sim, seed=seed + 1, verbose=False)
    return {
        "n_hypotheses": len(res),
        "raw_significant": int(sum(r["raw_sig"] for r in res)),
        "fdr_significant": int(sum(r["fdr_sig"] for r in res)),
        "expected_raw_by_chance": len(res) * 0.05,
        "results": res,
    }


def load_model():
    """인기도 모델 로드. 정식 산출물(web/data) 우선, 없으면 진단용(data) 사용."""
    for path in (ROOT / "web" / "data" / "popularity.json",
                 ROOT / "data" / "popularity.json"):
        if path.exists():
            return P.PopularityModel.from_dict(json.loads(path.read_text())["model"])
    raise SystemExit("popularity.json 이 없습니다. 먼저 scripts/build.py 를 실행하세요.")


def main():
    quick = "--quick" in sys.argv
    df = pd.read_csv(ROOT / "data" / "draws.csv")
    model = load_model()
    prev = P.prev_draws(df, [int(df.draw_no.max()) + 1])[0]   # 다음 회차 기준

    print("=" * 92)
    print("검증 4 — 포트폴리오 전수 열거 (표본오차 0)")
    print("=" * 92)
    mc = {}
    for n in (10, 20):
        r = exact_compare(model, prev, n)
        mc[n] = r
        o, b = r["optimized"], r["random"]
        print(f"\n[{n}장, 가능한 추첨 8,145,060가지 전수 열거]")
        print(f"  {'지표':<26}{'자동선택 랜덤':>16}{'최적화':>16}{'변화':>12}")
        print(f"  {'P(1등)':<26}{b['p_jackpot']:>16.9f}{o['p_jackpot']:>16.9f}"
              f"{'  정확히 동일':>12}")
        print(f"  {'P(1개 이상 당첨)':<26}{b['p_any_prize']:>16.4f}{o['p_any_prize']:>16.4f}"
              f"{(o['p_any_prize']/b['p_any_prize']-1)*100:>+11.2f}%")
        print(f"  {'평균 당첨 티켓 수':<26}{b['mean_prizes']:>16.4f}{o['mean_prizes']:>16.4f}"
              f"{(o['mean_prizes']/b['mean_prizes']-1)*100:>+11.2f}%")
        print(f"  {'평균 인기배수(낮을수록 좋음)':<26}{b['mean_multiplier']:>16.3f}"
              f"{o['mean_multiplier']:>16.3f}"
              f"{(o['mean_multiplier']/b['mean_multiplier']-1)*100:>+11.2f}%")
        print(f"  {'티켓 간 평균 겹침':<26}{b['mean_overlap']:>16.3f}{o['mean_overlap']:>16.3f}")
        print(f"  {'사용한 서로 다른 번호':<26}{b['distinct_numbers']:>16d}"
              f"{o['distinct_numbers']:>16d}")
        print(f"  이론값 P(1등) = {n}/8,145,060 = {n/PF.M:.7f}")

    print("\n" + "=" * 92)
    print("워크포워드 백테스트 (회차 t 예측에 t-1 까지만 사용)")
    print("=" * 92)
    cache = P.DesignCache(df)
    start = 1150 if quick else 1000
    bts = {}
    for n in (10, 20):
        print(f"\n[{n}장, {start+1}~1238회]")
        bt = walk_forward(df, n_tickets=n, start=start, cache=cache,
                          sa_rounds=300 if quick else 500, n_reps=2 if quick else 5)
        s = summarize(bt, n, df)
        bts[n] = {"summary": s, "rounds": bt.to_dict("records")}
        print(f"  {'':<22}{'자동선택 랜덤':>18}{'최적화 포트폴리오':>20}")
        print(f"  {'총 구매액':<22}{s['spend']:>18,}원{'':>20}")
        print(f"  {'총 당첨금':<22}{s['random']['prize']:>17,.0f}원{s['ours']['prize']:>19,.0f}원")
        print(f"  {'ROI':<22}{s['random']['roi']*100:>17.1f}%{s['ours']['roi']*100:>19.1f}%")
        print(f"  {'회차당 1개이상 당첨률':<22}{s['random']['any_rate']*100:>17.1f}%"
              f"{s['ours']['any_rate']*100:>19.1f}%")
        print(f"  {'총 당첨 티켓 수':<22}{s['random']['wins']:>18,.0f}{s['ours']['wins']:>20,.0f}")
        print(f"  {'평균 인기배수':<22}{1.0:>18.3f}{s['ours']['mean_multiplier']:>20.3f}")

    print("\n" + "=" * 92)
    print("검증 3 — 거짓양성 통제 (균일 난수 가짜 이력에 Group A 전체 실행)")
    print("=" * 92)
    fp = false_positive_check(n_sim=300 if quick else 2000)
    print(f"  가설 {fp['n_hypotheses']}개 중  raw p<0.05: {fp['raw_significant']}개 "
          f"(우연 기대 {fp['expected_raw_by_chance']:.1f}개)  /  FDR 생존: "
          f"{fp['fdr_significant']}개  <- 0이어야 정상")

    out = {"montecarlo": {str(k): {kk: vv for kk, vv in v.items() if kk != "tickets"}
                          for k, v in mc.items()},
           "walk_forward": {str(k): v for k, v in bts.items()},
           "false_positive_check": {k: v for k, v in fp.items() if k != "results"}}
    (ROOT / "data" / "backtest.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: data/backtest.json")


if __name__ == "__main__":
    main()
