"""Group B — 플레이어 번호 선호(인기도) 모델.

당첨 확률은 조합마다 동일하지만 1~3등은 파리뮤추얼이라 당첨금을 당첨자끼리
나눈다. 따라서 "사람들이 덜 고르는 조합"을 고르면 당첨 시 수령액이 커진다.

식별 아이디어
    5등(3개 매칭) 당첨자는 회당 약 160만 명이라 포아송 잡음이 0.08%인데 실제
    편차는 4.4%다(54배). 이 편차는 잡음이 아니라 플레이어의 번호 선호다.

모델
    조합 c 가 선택될 상대 확률(균일 대비 배수)을
        log Mult(c) = sum_{i in c} beta_i  +  f(c) . gamma
    로 둔다. 앞항은 번호별 선호, 뒷항은 조합 단위 선호(연속수 기피, 합계 선호,
    용지 마킹 패턴 등)다.

    매칭 k개인 등수의 관측 로그비는 선형 근사에서
        log r_k(W) = c_k * D_W  +  g_k(W) . gamma
        c_k = k/6 - (6-k)/39            (번호 선호에 대한 등수별 민감도)
        D_W = sum_{i in W} beta_i
        g_k(W) = mean_{c: |c ∩ W| = k} f(c)  -  E_uniform[f]
    가 된다. c_k 는 조합론으로 정해지는 상수고, g_k 는 A_k(W) 집합에서 직접
    계산한다(k=5,6은 완전열거, k=3,4는 몬테카를로).

    자동선택(균일) 비중 alpha 는 선형 영역에서 beta, gamma 에 그대로 흡수되므로
    따로 추정하지 않는다. 우리가 원하는 건 희석 후의 실효 인기도다.

    실측된 D_W 표준편차가 매칭 3->6개로 갈수록 0.104 -> 0.286 으로 커진다.
    번호별 선호만 있었다면 등수와 무관하게 같아야 하므로, 이것이 조합 단위
    선호가 실재한다는 직접 증거다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

import features as F

ROOT = Path(__file__).resolve().parent.parent
M = F.TOTAL_COMBOS
PICK, N_MAX = F.PICK, F.N_MAX

#: 매칭 k개 조합의 균일 확률
U = {k: comb(PICK, k) * comb(N_MAX - PICK, PICK - k) / M for k in range(PICK + 1)}
#: 번호 선호에 대한 등수별 민감도 (조합론 상수)
C_K = {k: k / PICK - (PICK - k) / (N_MAX - PICK) for k in range(PICK + 1)}

#: 게임 가격은 88회(2004-08-07)에 2,000원 -> 1,000원으로 인하됐다.
PRICE_CHANGE_ROUND = 88
#: 1~100회는 제도 과도기라 내재 티켓수가 ±40% 요동친다. 학습에서 제외한다.
TRAIN_FROM = 101

#: 인기도 회귀에 쓰는 조합 특징.
#: sum / low_count / under31_count / odd_count / tail_sum 처럼 번호별 값의 단순
#: 합인 특징은 45개 beta 항과 완전 공선이라 넣으면 안 된다(계수 부호가 뒤집힌다).
#: 여기에는 번호 하나로 분해되지 않는 상호작용 특징만 남긴다.
POP_FEATURES = [
    "spread", "stdev", "min_gap", "consecutive_pairs", "max_run",
    "zone_spread", "tail_unique", "ac_value",
    "max_same_col", "max_same_row", "max_diagonal",
]

#: 직전 회차 의존 특징 — "사람들이 저번 당첨번호를 그대로 찍는가"(가설 B8).
CONTEXT_FEATURES = ["prev_overlap", "prev2_overlap", "prev_neighbor"]


def context_matrix(combos, prev_draws) -> np.ndarray:
    """(N, 3) 직전 회차 의존 특징.

    prev_draws: (N, 3, 6) — 각 조합에 대응하는 직전 1·2회차 당첨번호와
    직전 회차 번호(이웃 계산용).
    """
    a = F.as_array(combos).astype(np.int64)
    n = len(a)
    out = np.zeros((n, 3))
    p1, p2 = prev_draws[:, 0, :], prev_draws[:, 1, :]
    m1 = np.zeros((n, N_MAX + 2), dtype=bool)
    np.put_along_axis(m1, p1, True, axis=1)
    m2 = np.zeros((n, N_MAX + 2), dtype=bool)
    np.put_along_axis(m2, p2, True, axis=1)
    mn = np.zeros((n, N_MAX + 3), dtype=bool)
    for d in (-1, 1):
        idx = np.clip(p1 + d, 0, N_MAX + 2)
        np.put_along_axis(mn, idx, True, axis=1)
    out[:, 0] = np.take_along_axis(m1, a, axis=1).sum(axis=1)
    out[:, 1] = np.take_along_axis(m2, a, axis=1).sum(axis=1)
    out[:, 2] = np.take_along_axis(mn, a, axis=1).sum(axis=1)
    return out


def game_price(draw_no):
    return np.where(np.asarray(draw_no) < PRICE_CHANGE_ROUND, 2000.0, 1000.0)


def winner_counts(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """매칭 개수 -> 당첨자 수. 2등(5개+보너스)과 3등(5개)은 같은 5매칭이다."""
    return {
        6: df.w1_winners.to_numpy(float),
        5: (df.w2_winners + df.w3_winners).to_numpy(float),
        4: df.w4_winners.to_numpy(float),
        3: df.w5_winners.to_numpy(float),
    }


# ------------------------------------------------------- A_k(W) 집합의 특징 평균

def _sample_match_set(W: np.ndarray, k: int, n: int, rng) -> np.ndarray:
    """당첨번호 W 와 정확히 k개 겹치는 조합을 n개 표본추출. (T, n, 6)."""
    T = len(W)
    # W 에서 k개
    pick_in = np.argsort(rng.random((T, n, PICK)), axis=2)[:, :, :k]
    ins = np.take_along_axis(np.broadcast_to(W[:, None, :], (T, n, PICK)), pick_in, axis=2)
    # W 밖에서 6-k개
    others = np.empty((T, N_MAX - PICK), dtype=np.int64)
    allnum = np.arange(1, N_MAX + 1)
    for t in range(T):
        others[t] = np.setdiff1d(allnum, W[t], assume_unique=True)
    pick_out = np.argsort(rng.random((T, n, N_MAX - PICK)), axis=2)[:, :, :PICK - k]
    outs = np.take_along_axis(np.broadcast_to(others[:, None, :], (T, n, N_MAX - PICK)),
                              pick_out, axis=2)
    return np.concatenate([ins, outs], axis=2)


def _enumerate_match_set(W: np.ndarray, k: int) -> np.ndarray:
    """A_k(W) 완전열거. k=5(234개), k=6(1개)에만 쓴다."""
    T = len(W)
    allnum = np.arange(1, N_MAX + 1)
    rows = []
    for t in range(T):
        others = np.setdiff1d(allnum, W[t], assume_unique=True)
        cur = [list(a) + list(b)
               for a in combinations(W[t], k)
               for b in combinations(others, PICK - k)]
        rows.append(cur)
    return np.asarray(rows, dtype=np.int64)


def match_set_features(W, k, names, *, n_sample=3000, rng=None, prev=None):
    """A_k(W) 에서의 특징 평균 (T, n_features [+ n_context])."""
    S = _enumerate_match_set(W, k) if k >= 5 else \
        _sample_match_set(W, k, n_sample, rng or np.random.default_rng(0))
    T, n, _ = S.shape
    flat = S.reshape(T * n, PICK)
    fm = F.feature_matrix(flat, names).reshape(T, n, -1).mean(axis=1)
    if prev is None:
        return fm
    prev_flat = np.repeat(prev, n, axis=0)
    cm = context_matrix(flat, prev_flat).reshape(T, n, -1).mean(axis=1)
    return np.hstack([fm, cm])


def uniform_feature_moments(names, *, n=400_000, seed=7, with_context=True):
    """균일 랜덤 조합에서의 특징 평균/표준편차 (표준화용)."""
    rng = np.random.default_rng(seed)
    samples = np.argsort(rng.random((n, N_MAX)), axis=1)[:, :PICK] + 1
    fm = F.feature_matrix(samples, names)
    if with_context:
        prev = np.argsort(rng.random((n, 2, N_MAX)), axis=2)[:, :, :PICK] + 1
        fm = np.hstack([fm, context_matrix(samples, prev)])
    return fm.mean(axis=0), fm.std(axis=0)


# ---------------------------------------------------------------------- 모델

@dataclass
class PopularityModel:
    beta: np.ndarray              # (45,) 번호별 로그 인기 가중치, 합 0
    gamma: np.ndarray             # (n_features,) 조합 특징 계수 (표준화 척도)
    feature_names: list
    feat_mean: np.ndarray
    feat_std: np.ndarray
    train_rounds: tuple
    diagnostics: dict = field(default_factory=dict)

    def log_multiplier(self, combos, prev=None) -> np.ndarray:
        """prev: (2, 6) 또는 (N, 2, 6) 직전 1·2회차 당첨번호."""
        a = F.as_array(combos).astype(np.int64)
        num = self.beta[a - 1].sum(axis=1)
        fm = F.feature_matrix(a, self.feature_names)
        if prev is None:                      # 맥락 미지정 -> 균일 평균으로 대체
            cm = np.broadcast_to(self.feat_mean[len(self.feature_names):], (len(a), 3))
        else:
            pv = np.asarray(prev, dtype=np.int64)
            if pv.ndim == 2:
                pv = np.broadcast_to(pv, (len(a),) + pv.shape)
            cm = context_matrix(a, pv)
        z = (np.hstack([fm, cm]) - self.feat_mean) / self.feat_std
        return num + z @ self.gamma

    def multiplier(self, combos, prev=None) -> np.ndarray:
        """조합별 1등 동시당첨자 배수. 1.0=평균, 1.5=당첨자 1.5배(수령액 2/3)."""
        return np.exp(self.log_multiplier(combos, prev))

    @property
    def all_feature_names(self):
        return list(self.feature_names) + CONTEXT_FEATURES

    def to_dict(self) -> dict:
        return {
            "beta": self.beta.tolist(),
            "gamma": self.gamma.tolist(),
            "feature_names": list(self.feature_names),
            "feat_mean": self.feat_mean.tolist(),
            "feat_std": self.feat_std.tolist(),
            "train_rounds": list(self.train_rounds),
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["beta"]), np.asarray(d["gamma"]), d["feature_names"],
                   np.asarray(d["feat_mean"]), np.asarray(d["feat_std"]),
                   tuple(d["train_rounds"]), d.get("diagnostics", {}))


# ---------------------------------------------------------------------- 학습

def prev_draws(full: pd.DataFrame, draw_nos) -> np.ndarray:
    """각 회차의 직전 1·2회차 당첨번호 (T, 2, 6). 1·2회차는 자기 자신으로 채운다."""
    cols = [f"n{i}" for i in range(1, 7)]
    by_no = {int(r.draw_no): r[cols].to_numpy(np.int64) for _, r in full.iterrows()}
    out = np.empty((len(draw_nos), 2, PICK), dtype=np.int64)
    for i, no in enumerate(draw_nos):
        for j, back in enumerate((1, 2)):
            out[i, j] = by_no.get(int(no) - back, by_no[int(no)])
    return out


def _design(df, full, names, feat_mean, feat_std, divisions, *, n_sample, rng):
    """등수별 관측을 세로로 쌓은 (X, y, poisson_var)."""
    W = df[[f"n{i}" for i in range(1, 7)]].to_numpy(np.int64)
    tickets = df.sales.to_numpy() / game_price(df.draw_no.to_numpy())
    counts = winner_counts(df)
    prev = prev_draws(full, df.draw_no.to_numpy())
    T = len(df)

    onehot = np.zeros((T, N_MAX))
    np.put_along_axis(onehot, W - 1, 1.0, axis=1)

    Xs, ys, vs = [], [], []
    for k in divisions:
        g = (match_set_features(W, k, names, n_sample=n_sample, rng=rng, prev=prev)
             - feat_mean) / feat_std
        exp = tickets * U[k]
        Xs.append(np.hstack([C_K[k] * onehot, g]))
        ys.append(np.log(np.maximum(counts[k], 0.5) / exp))
        vs.append(1.0 / exp)                      # 로그비의 포아송 분산
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(vs)


#: 홀드아웃(900/1000/1100 분할)으로 튜닝한 기본값. 릿지는 이 구간에서 평평하다.
DEFAULTS = dict(divisions=(3, 4, 5), l2_beta=1e-3, l2_gamma=1e-3,
                n_sample=4000, var_floor=1e-3)


def fit(df, *, train_from=TRAIN_FROM, train_to=None, names=None,
        divisions=(3, 4, 5), l2_beta=1e-3, l2_gamma=1e-3, n_sample=4000,
        var_floor=1e-3, seed=0, moments=None):
    """4·5매칭(그리고 3매칭)으로 beta, gamma 를 적합. 6매칭(1등)은 쓰지 않는다."""
    names = names or POP_FEATURES
    train_to = train_to or int(df.draw_no.max())
    d = df[(df.draw_no >= train_from) & (df.draw_no <= train_to)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    feat_mean, feat_std = moments or uniform_feature_moments(names)

    X, y, pvar = _design(d, df, names, feat_mean, feat_std, divisions,
                         n_sample=n_sample, rng=rng)
    # 관측 잡음(포아송)보다 모델 오차가 지배적이므로 분산 하한을 둔다.
    w = 1.0 / (pvar + var_floor)
    nb, ng = N_MAX, len(names) + len(CONTEXT_FEATURES)

    # 합 0 제약: beta 의 자유도는 44. 마지막 성분을 나머지의 음수합으로 둔다.
    Bc = np.vstack([np.eye(nb - 1), -np.ones((1, nb - 1))])
    Z = np.hstack([X[:, :nb] @ Bc, X[:, nb:]])

    pen = np.diag(np.concatenate([np.full(nb - 1, l2_beta), np.full(ng, l2_gamma)]))
    Zw = Z * np.sqrt(w)[:, None]
    A = Zw.T @ Zw + pen * len(y)
    b = Zw.T @ (y * np.sqrt(w))
    coef = np.linalg.solve(A, b)

    beta = Bc @ coef[:nb - 1]
    gamma = coef[nb - 1:]
    return PopularityModel(beta, gamma, names, feat_mean, feat_std,
                           (train_from, train_to),
                           {"divisions": list(divisions), "l2_beta": l2_beta,
                            "l2_gamma": l2_gamma, "n_sample": n_sample})


class DesignCache:
    """설계행렬을 한 번만 만들어 두고 학습 구간만 바꿔 재적합한다.

    워크포워드 백테스트는 모델을 수십 번 다시 적합해야 하는데, 비용의 대부분은
    A_k(W) 표본추출과 특징 추출이다. 그 부분은 학습 구간과 무관하므로 캐시한다.
    """

    def __init__(self, df, *, names=None, train_from=TRAIN_FROM, divisions=(3, 4, 5, 6),
                 n_sample=4000, seed=0, moments=None):
        self.names = names or POP_FEATURES
        self.feat_mean, self.feat_std = moments or uniform_feature_moments(self.names)
        self.train_from = train_from
        d = df[df.draw_no >= train_from].reset_index(drop=True)
        self.draw_no = d.draw_no.to_numpy()
        self.W = d[[f"n{i}" for i in range(1, 7)]].to_numpy(np.int64)
        self.tickets = d.sales.to_numpy() / game_price(self.draw_no)
        self.prev = prev_draws(df, self.draw_no)
        counts = winner_counts(d)
        onehot = np.zeros((len(d), N_MAX))
        np.put_along_axis(onehot, self.W - 1, 1.0, axis=1)
        rng = np.random.default_rng(seed)
        self.blocks = {}
        for k in divisions:
            g = (match_set_features(self.W, k, self.names, n_sample=n_sample,
                                    rng=rng, prev=self.prev) - self.feat_mean) / self.feat_std
            exp = self.tickets * U[k]
            self.blocks[k] = {"X": np.hstack([C_K[k] * onehot, g]),
                              "y": np.log(np.maximum(counts[k], 0.5) / exp),
                              "pvar": 1.0 / exp}
        self._Bc = np.vstack([np.eye(N_MAX - 1), -np.ones((1, N_MAX - 1))])

    def fit(self, train_to, *, divisions=(3, 4, 5), l2_beta=1e-3, l2_gamma=1e-3,
            var_floor=1e-3) -> PopularityModel:
        m = self.draw_no <= train_to
        Zs, ys, ws = [], [], []
        for k in divisions:
            B = self.blocks[k]
            Zs.append(np.hstack([B["X"][m, :N_MAX] @ self._Bc, B["X"][m, N_MAX:]]))
            ys.append(B["y"][m])
            ws.append(1.0 / (B["pvar"][m] + var_floor))
        Z, y, w = np.vstack(Zs), np.concatenate(ys), np.concatenate(ws)
        ng = Z.shape[1] - (N_MAX - 1)
        pen = np.diag(np.concatenate([np.full(N_MAX - 1, l2_beta), np.full(ng, l2_gamma)]))
        Zw = Z * np.sqrt(w)[:, None]
        coef = np.linalg.solve(Zw.T @ Zw + pen * len(y), Zw.T @ (y * np.sqrt(w)))
        return PopularityModel(self._Bc @ coef[:N_MAX - 1], coef[N_MAX - 1:], self.names,
                               self.feat_mean, self.feat_std, (self.train_from, train_to),
                               {"divisions": list(divisions), "l2_beta": l2_beta,
                                "l2_gamma": l2_gamma, "cached": True})


# ------------------------------------------------------------------ 검증 2

def holdout_report(df, *, split=1000, moments=None, **fit_kw) -> dict:
    """1~split 학습 -> split+1~끝 의 1등 당첨자 수를 아웃오브샘플 예측.

    1등(6매칭)은 학습에 전혀 쓰지 않으므로 완전히 독립적인 검증이다.
    """
    model = fit(df, train_to=split, moments=moments, **fit_kw)
    test = df[df.draw_no > split].reset_index(drop=True)
    W = test[[f"n{i}" for i in range(1, 7)]].to_numpy(np.int64)
    tickets = test.sales.to_numpy() / game_price(test.draw_no.to_numpy())
    pv = prev_draws(df, test.draw_no.to_numpy())

    actual = test.w1_winners.to_numpy(float)
    base = tickets * U[6]
    mult = model.multiplier(W, pv)
    pred = base * mult

    def stats(p):
        resid = actual - p
        return {"mae": float(np.abs(resid).mean()),
                "rmse": float(np.sqrt((resid ** 2).mean())),
                "loglik": float((actual * np.log(np.maximum(p, 1e-9)) - p).sum())}

    r_actual, r_pred = actual / base, mult
    return {"split": split, "n_test": len(test),
            "baseline": stats(base), "model": stats(pred),
            "corr": float(np.corrcoef(r_actual, r_pred)[0, 1]),
            "pred_spread": float(r_pred.std()), "model_obj": model}


def effect_size_report(df, *, split=1000, moments=None, **fit_kw) -> dict:
    """아웃오브샘플에서 "저인기 조합이 실제로 덜 갈리는가"를 사분위로 확인.

    1등은 학습에 쓰지 않았으므로 이 표가 인기도 모델의 최종 검증이다.
    """
    model = fit(df, train_to=split, moments=moments, **fit_kw)
    te = df[df.draw_no > split].reset_index(drop=True)
    W = te[[f"n{i}" for i in range(1, 7)]].to_numpy(np.int64)
    pv = prev_draws(df, te.draw_no.to_numpy())
    tickets = te.sales.to_numpy() / game_price(te.draw_no.to_numpy())
    base = tickets * U[6]
    actual = te.w1_winners.to_numpy(float)
    mult = model.multiplier(W, pv)

    q = pd.qcut(mult, 4, labels=False)
    quartiles = [{
        "quartile": int(i + 1),
        "n": int((q == i).sum()),
        "pred_multiplier": float(mult[q == i].mean()),
        "actual_winners": float(actual[q == i].sum()),
        "expected_winners": float(base[q == i].sum()),
        "actual_over_expected": float(actual[q == i].sum() / base[q == i].sum()),
    } for i in range(4)]

    # 순열검정: 예측배수와 실제비율의 상관이 우연인지
    rng = np.random.default_rng(3)
    ratio = actual / base
    obs = float(np.corrcoef(ratio, mult)[0, 1])
    null = np.array([np.corrcoef(ratio, rng.permutation(mult))[0, 1] for _ in range(20_000)])
    perm_p = float((np.sum(null >= obs) + 1) / (len(null) + 1))

    won = te[te.w1_winners > 0]
    mw = model.multiplier(won[[f"n{i}" for i in range(1, 7)]].to_numpy(np.int64),
                          prev_draws(df, won.draw_no.to_numpy()))
    half = mw <= np.median(mw)
    payout = {
        "unpopular_half": {"n": int(half.sum()),
                           "mean_winners": float(won.w1_winners.to_numpy()[half].mean()),
                           "mean_prize": float(won.w1_prize.to_numpy()[half].mean())},
        "popular_half": {"n": int((~half).sum()),
                         "mean_winners": float(won.w1_winners.to_numpy()[~half].mean()),
                         "mean_prize": float(won.w1_prize.to_numpy()[~half].mean())},
    }
    payout["prize_ratio"] = (payout["unpopular_half"]["mean_prize"]
                             / payout["popular_half"]["mean_prize"])
    return {"split": split, "n_test": len(te), "quartiles": quartiles,
            "corr": obs, "perm_p": perm_p, "payout": payout}


def main():
    df = pd.read_csv(ROOT / "data" / "draws.csv")
    moments = uniform_feature_moments(POP_FEATURES)
    print(f"데이터 {len(df)}회차 (1~{int(df.draw_no.max())})")
    print("=" * 74)
    print("검증 2 — 인기도 모델 홀드아웃 (1등=6매칭은 학습에 미사용)")
    print("=" * 74)

    for split in (900, 1000, 1100):
        r = holdout_report(df, split=split, moments=moments)
        b, m = r["baseline"], r["model"]
        print(f"\n[1~{split} 학습 -> {split+1}~1238 검정, n={r['n_test']}]")
        print(f"  1등 당첨자 MAE    균일 {b['mae']:.3f}명 -> 모델 {m['mae']:.3f}명 "
              f"({(b['mae']-m['mae'])/b['mae']*100:+.1f}%)")
        print(f"  1등 당첨자 RMSE   균일 {b['rmse']:.3f}명 -> 모델 {m['rmse']:.3f}명 "
              f"({(b['rmse']-m['rmse'])/b['rmse']*100:+.1f}%)")
        print(f"  포아송 로그가능도 균일 {b['loglik']:.1f} -> 모델 {m['loglik']:.1f} "
              f"({m['loglik']-b['loglik']:+.1f})")
        print(f"  실제/예측 배수 상관  r = {r['corr']:+.3f}   예측배수 SD = {r['pred_spread']:.3f}")

    model = fit(df, moments=moments)
    p = np.exp(model.beta)
    order = np.argsort(-model.beta)
    print("\n" + "=" * 74)
    print("전체 학습 결과 — 번호별 인기 가중치")
    print("=" * 74)
    print("  상위 12:", ", ".join(f"{i+1}:{p[i]:.3f}" for i in order[:12]))
    print("  하위 12:", ", ".join(f"{i+1}:{p[i]:.3f}" for i in order[-12:]))
    print(f"\n  1~31 평균 {p[:31].mean():.4f}  vs  32~45 평균 {p[31:].mean():.4f}"
          f"  -> 생일 편향 {p[:31].mean()/p[31:].mean():.3f}배")
    print("\n  조합 특징 계수 (표준화, 양수=인기):")
    for n, g in sorted(zip(model.all_feature_names, model.gamma), key=lambda x: -abs(x[1])):
        print(f"    {n:20s} {g:+.4f}")

    print("\n" + "=" * 74)
    print("효과 크기 — 저인기 조합이 실제로 덜 갈리는가 (1001~1238회 아웃오브샘플)")
    print("=" * 74)
    eff = effect_size_report(df, split=1000, moments=moments)
    print(f"{'인기도 사분위':<16}{'n':>5}{'예측배수':>10}{'실제/기대 당첨자':>18}")
    for qd in eff["quartiles"]:
        lab = {1: "최저(비인기)", 2: "2분위", 3: "3분위", 4: "최고(인기)"}[qd["quartile"]]
        print(f"{lab:<16}{qd['n']:>5}{qd['pred_multiplier']:>10.3f}"
              f"{qd['actual_over_expected']:>18.3f}")
    pay = eff["payout"]
    print(f"\n  상관 r={eff['corr']:+.3f}  순열검정 p={eff['perm_p']:.5f}")
    print(f"  1등 실제 수령액  비인기 절반 {pay['unpopular_half']['mean_prize']:,.0f}원"
          f"  vs  인기 절반 {pay['popular_half']['mean_prize']:,.0f}원"
          f"  ({pay['prize_ratio']:.2f}배)")

    out = {"model": model.to_dict(), "validation": eff}
    (ROOT / "data" / "popularity.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장: data/popularity.json")


if __name__ == "__main__":
    main()
