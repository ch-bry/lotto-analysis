"""전체 파이프라인 — web/data/*.json 생성.

GitHub Actions 가 매주 일요일 새벽에 실행하는 단일 진입점이다.
    fetch.py -> (인기도 모델 / 가설 검정 / 백테스트) -> 추천 -> web/data/*.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
import hypotheses as H
import popularity as P
import portfolio as PF
import backtest as BT

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web" / "data"
KST = timezone(timedelta(hours=9))

#: 조합 인기도 백분위를 매기기 위한 참조 표본 크기
PCTL_SAMPLE = 300_000


def write(name, obj):
    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / name).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")),
                            encoding="utf-8")
    kb = (WEB / name).stat().st_size / 1024
    print(f"  web/data/{name}  ({kb:,.0f} KB)")


def describe(combo, prev, model, pct, base_winners):
    """조합 하나에 대한 사람이 읽을 근거."""
    a = np.asarray(combo)
    mult = float(model.multiplier(a[None, :], prev)[0])
    notes = []
    hi = int((a > 31).sum())
    if hi >= 3:
        notes.append(f"32~45번을 {hi}개 포함 (생일로 고르기 힘든 구간)")
    elif hi == 0:
        notes.append("전부 1~31 구간 — 생일 조합과 겹칠 수 있음")
    cp = int(F.consecutive_pairs(a[None, :])[0])
    if cp:
        notes.append(f"연속번호 {cp}쌍 — 사람들은 번호를 흩뿌려 고르는 편이라 덜 선택됨")
    ov = int(np.isin(a, prev[0]).sum())
    if ov:
        notes.append(f"직전 회차 번호 {ov}개 포함 (피할 이유가 없음)")
    total = int(a.sum())
    if total < 110 or total > 170:
        notes.append(f"번호 합 {total} — 중앙(약 138)에서 벗어난 구간")
    else:
        notes.append(f"번호 합 {total}")
    return {
        "numbers": [int(v) for v in a],
        "multiplier": mult,
        "percentile": float((pct < mult).mean() * 100),
        "expected_co_winners": float(base_winners * mult),
        "notes": notes,
    }


def popularity_factors(model, prev, *, n=300_000, seed=13):
    """조합 특성별 평균 인기 배수 — 사이트에 싣는 해석용.

    회귀 계수를 그대로 보여주면 안 된다. min_gap 과 consecutive_pairs 처럼 서로
    거의 같은 것을 재는 특징들이 있어 개별 계수의 부호가 뒤집혀 보이기 때문이다.
    대신 무작위 조합을 대량으로 뽑아 특성별 평균 배수를 직접 측정한다.
    이 값은 공선성과 무관하게 "이런 조합은 실제로 얼마나 인기 있나"를 말해준다.
    """
    rng = np.random.default_rng(seed)
    S = np.argsort(rng.random((n, F.N_MAX)), axis=1)[:, :F.PICK] + 1
    S = np.sort(S, axis=1)
    mult = model.multiplier(S, np.broadcast_to(prev, (n,) + prev.shape))
    overall = float(mult.mean())

    def group(name, key, labels, desc):
        out = []
        for value, label in labels:
            m = key == value
            if m.sum() < 200:
                continue
            out.append({"label": label, "n": int(m.sum()),
                        "multiplier": float(mult[m].mean()),
                        "vs_overall": float(mult[m].mean() / overall - 1)})
        return {"name": name, "desc": desc, "groups": out}

    hi = (S > 31).sum(axis=1)
    cp = F.consecutive_pairs(S)
    ov = np.isin(S, prev[0]).sum(axis=1)
    total = F.total(S)
    sum_bin = np.digitize(total, [100, 125, 150, 175])
    mg = np.minimum(F.min_gap(S), 5)

    return {
        "overall": overall,
        "n_sample": n,
        "factors": [
            group("32~45번(생일로 쓸 수 없는 번호) 개수", hi,
                  [(k, f"{k}개") for k in range(7)],
                  "날짜(1~31)로 번호를 고르는 사람이 많아, 큰 번호가 많을수록 덜 겹칩니다."),
            group("연속번호 쌍의 개수", np.minimum(cp, 3),
                  [(0, "없음"), (1, "1쌍"), (2, "2쌍"), (3, "3쌍 이상")],
                  "실제 회차의 약 49%가 연속번호를 포함하는데도 사람들은 이를 피합니다."),
            group("번호 사이 최소 간격", mg,
                  [(1, "1 (연속)"), (2, "2"), (3, "3"), (4, "4"), (5, "5 이상")],
                  "사람들은 번호를 고르게 흩뿌려 고르는 경향이 있습니다."),
            group("번호 6개의 합", sum_bin,
                  [(0, "100 미만"), (1, "100~124"), (2, "125~149"),
                   (3, "150~174"), (4, "175 이상")],
                  "합이 중앙(약 138) 근처인 조합이 가장 흔하게 선택됩니다."),
            group("직전 회차 당첨번호와 겹치는 개수", np.minimum(ov, 3),
                  [(0, "0개"), (1, "1개"), (2, "2개"), (3, "3개 이상")],
                  "'지난 번호는 빼라'는 통념과 달리, 겹쳐도 인기가 크게 오르지 않습니다."),
        ],
    }


def build_recommendations(df, model, ns=(10, 20), seed=1):
    latest = int(df.draw_no.max())
    target = latest + 1
    # 예측 대상은 target 회차이므로 직전 회차는 latest 다. prev_draws 에 latest 를
    # 넘기면 latest-1 이 나와 한 회차씩 밀린다.
    prev = P.prev_draws(df, [target])[0]
    assert prev[0].tolist() == sorted(
        df[df.draw_no == latest][[f"n{i}" for i in range(1, 7)]].to_numpy()[0].tolist()), \
        "직전 회차가 최신 회차와 다르다"
    rng = np.random.default_rng(7)
    sample = np.argsort(rng.random((PCTL_SAMPLE, F.N_MAX)), axis=1)[:, :F.PICK] + 1
    pct = model.multiplier(sample, prev)

    recent_tickets = float((df.sales.tail(20) / 1000).mean())
    base_winners = recent_tickets * P.U[6]

    collide = PF.collide_table()
    out = {}
    for n in ns:
        # 커버리지 가중치의 안전 경계는 티켓 수에 따라 다르다. 고정값을 쓰면
        # 10장에 맞춘 값이 20장에서 자동선택보다 나빠진다. 매번 보정한다.
        cal = PF.calibrate_coverage_weight(model, prev, n, collide=collide)
        tk, _ = PF.optimize(model, n, prev, seed=seed, collide=collide,
                            lam_cov=cal["weight"] / collide[1])
        ev = PF.exact_evaluate(tk, model, prev)
        rnd = PF.random_baseline(model, prev, n, seed=5)   # backtest 와 동일 기준선
        assert ev["p_any_prize"] >= rnd["p_any_prize"], \
            f"{n}장: 최소1개 당첨 확률이 자동선택보다 낮다 — 가중치 보정 실패"
        out[str(n)] = {
            "tickets": [describe(t, prev, model, pct, base_winners) for t in tk],
            "portfolio": ev,
            "random_baseline": rnd,
            "coverage_weight": cal["weight"],
            "calibration": cal,
        }
        print(f"  {n}장: 커버리지 가중치 {cal['weight']} · 인기배수 "
              f"{ev['mean_multiplier']:.3f} · P(1개↑) {rnd['p_any_prize']:.5f} -> "
              f"{ev['p_any_prize']:.5f}")
    return {
        "target_draw": target,
        "based_on_draw": latest,
        "prev_numbers": [int(v) for v in prev[0]],
        "assumed_tickets": recent_tickets,
        "base_expected_jackpot_winners": base_winners,
        "multiplier_percentiles": {
            str(q): float(np.percentile(pct, q)) for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        },
        "by_size": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--quick", action="store_true", help="개발용 축소 실행")
    ap.add_argument("--sims", type=int, default=None)
    args = ap.parse_args()

    if not args.skip_fetch:
        print("[1/6] 데이터 수집")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch.py")], check=True)

    df = pd.read_csv(ROOT / "data" / "draws.csv")
    latest = int(df.draw_no.max())
    print(f"\n[2/6] 인기도 모델 (1~{latest}회)")
    cache = P.DesignCache(df, n_sample=1500 if args.quick else 4000)
    model = cache.fit(train_to=latest)

    # 검증 2 — 1등을 학습에 쓰지 않은 상태에서의 아웃오브샘플 성능
    validation = {}
    for split in (900, 1000, 1100):
        m = cache.fit(train_to=split)
        te = df[df.draw_no > split]
        W = te[[f"n{i}" for i in range(1, 7)]].to_numpy(np.int64)
        pv = P.prev_draws(df, te.draw_no.to_numpy())
        tickets = te.sales.to_numpy() / P.game_price(te.draw_no.to_numpy())
        base = tickets * P.U[6]
        act = te.w1_winners.to_numpy(float)
        mult = m.multiplier(W, pv)
        ll = float((act * np.log(np.maximum(base * mult, 1e-9)) - base * mult).sum())
        ll0 = float((act * np.log(base) - base).sum())
        validation[str(split)] = {
            "n_test": int(len(te)), "loglik_gain": ll - ll0,
            "corr": float(np.corrcoef(act / base, mult)[0, 1]),
            "mae_baseline": float(np.abs(act - base).mean()),
            "mae_model": float(np.abs(act - base * mult).mean()),
        }
    effect = P.effect_size_report(df, split=1000, moments=(cache.feat_mean, cache.feat_std),
                                 n_sample=1500 if args.quick else 4000)
    print(f"  아웃오브샘플 상관 r={effect['corr']:+.3f}  순열검정 p={effect['perm_p']:.5f}")
    print(f"  비인기/인기 절반 1등 수령액 비 {effect['payout']['prize_ratio']:.2f}배")

    print(f"\n[3/6] Group A 가설 검정")
    n_sim = args.sims or (300 if args.quick else 10_000)
    hyp = H.run(df, n_sim=n_sim, verbose=False)
    print(f"  {len(hyp)}개 가설  raw p<0.05: {sum(h['raw_sig'] for h in hyp)}개  "
          f"FDR 생존: {sum(h['fdr_sig'] for h in hyp)}개")

    print(f"\n[4/6] 백테스트")
    prev = P.prev_draws(df, [latest + 1])[0]      # 다음 회차 기준
    collide = PF.collide_table()
    weights = {n: PF.calibrate_coverage_weight(model, prev, n, collide=collide)["weight"]
               for n in (10, 20)}
    mc = {str(n): BT.exact_compare(model, prev, n, lam_cov=weights[n] / collide[1])
          for n in (10, 20)}
    for n, r in mc.items():
        o, b = r["optimized"], r["random"]
        print(f"  {n}장  P(1등) {b['p_jackpot']:.9f} -> {o['p_jackpot']:.9f}  |  "
              f"P(1개이상) {b['p_any_prize']:.4f} -> {o['p_any_prize']:.4f}  |  "
              f"인기배수 {b['mean_multiplier']:.3f} -> {o['mean_multiplier']:.3f}")
    wf = {}
    for n in (10, 20):
        reps = 2 if args.quick else 5
        bt = BT.walk_forward(df, n_tickets=n, start=1150 if args.quick else 1000,
                             cache=cache, sa_rounds=300 if args.quick else 500,
                             n_reps=reps, verbose=False)
        wf[str(n)] = {"summary": BT.summarize(bt, n, df, n_reps=reps),
                      "rounds": bt.to_dict("records")}
        s = wf[str(n)]["summary"]
        print(f"  {n}장 워크포워드 {s['n_rounds']}회  ROI 랜덤 {s['random']['roi']*100:.1f}% "
              f"vs 최적화 {s['ours']['roi']*100:.1f}%  |  1개이상 당첨률 "
              f"{s['random']['any_rate']*100:.1f}% vs {s['ours']['any_rate']*100:.1f}%")
    fp = BT.false_positive_check(n_sim=300 if args.quick else 2000)
    print(f"  거짓양성 통제: 가짜 이력에서 FDR 생존 {fp['fdr_significant']}개 (0이어야 정상)")

    print(f"\n[5/6] {latest + 1}회 추천 생성")
    rec = build_recommendations(df, model)
    rec["factors"] = popularity_factors(model, prev,
                                        n=80_000 if args.quick else 300_000)

    print(f"\n[6/6] web/data 출력")
    write("meta.json", {
        "collide": PF.collide_table().tolist(),   # JS 폴백이 같은 상수를 쓰도록
        "latest_draw": latest,
        "latest_date": str(df.date.iloc[-1]),
        "target_draw": latest + 1,
        "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "n_draws": int(len(df)),
        "data_source": "https://github.com/smok95/lotto (동행복권 공식 결과 미러)",
        "total_combinations": F.TOTAL_COMBOS,
    })
    write("draws.json", {
        "columns": ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus",
                    "sales", "w1_winners", "w1_prize", "w2_winners", "w3_winners",
                    "w4_winners", "w5_winners"],
        "rows": df[["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus",
                    "sales", "w1_winners", "w1_prize", "w2_winners", "w3_winners",
                    "w4_winners", "w5_winners"]].values.tolist(),
    })
    write("popularity.json", {"model": model.to_dict(), "validation": validation,
                              "effect": effect})
    write("hypotheses.json", {"n_sim": n_sim, "n_draws": int(len(df)), "results": hyp})
    write("backtest.json", {
        "montecarlo": {k: {kk: vv for kk, vv in v.items() if kk != "tickets"}
                       for k, v in mc.items()},
        "walk_forward": wf,
        "false_positive_check": {k: v for k, v in fp.items() if k != "results"},
    })
    write("recommendations.json", rec)
    print("\n완료.")


if __name__ == "__main__":
    main()
