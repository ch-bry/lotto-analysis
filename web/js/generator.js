/* 인기도 스코어링과 포트폴리오 최적화의 경량 JS 포팅.
 *
 * 평소에는 build.py 가 미리 계산해 둔 추천을 그대로 보여준다. 이 파일은
 * GitHub Actions 가 아직 안 돌았는데 새 회차가 나온 경우(토요일 밤~일요일 새벽)
 * 브라우저가 스스로 최신 회차를 반영해 추천을 다시 만들기 위한 폴백이다.
 * beta / gamma / 표준화 상수는 파이썬이 적합한 값을 그대로 쓴다.
 */
const LOTTO = (() => {
  const N_MAX = 45, PICK = 6, TOTAL = 8145060;

  /* features.py 의 POP_FEATURES 와 순서까지 같아야 한다. */
  const FEATURES = ["spread", "stdev", "min_gap", "consecutive_pairs", "max_run",
                    "zone_spread", "tail_unique", "ac_value",
                    "max_same_col", "max_same_row", "max_diagonal"];
  const CONTEXT = ["prev_overlap", "prev2_overlap", "prev_neighbor"];

  const ZONES = [[1, 9], [10, 19], [20, 29], [30, 39], [40, 45]];

  function std(arr) {
    const m = arr.reduce((a, b) => a + b, 0) / arr.length;
    return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
  }
  function maxCount(vals) {
    const m = new Map();
    let best = 0;
    for (const v of vals) { const c = (m.get(v) || 0) + 1; m.set(v, c); if (c > best) best = c; }
    return best;
  }

  /** 조합 하나(정렬된 6개 번호)의 상호작용 특징. */
  function combFeatures(c) {
    const a = [...c].sort((x, y) => x - y);
    const gaps = [];
    for (let i = 1; i < a.length; i++) gaps.push(a[i] - a[i - 1]);

    let run = 1, maxRun = 1;
    for (const g of gaps) { run = g === 1 ? run + 1 : 1; if (run > maxRun) maxRun = run; }

    const zc = ZONES.map(([lo, hi]) => a.filter(v => v >= lo && v <= hi).length);
    const diffs = [];
    for (let i = 0; i < a.length; i++)
      for (let j = i + 1; j < a.length; j++) diffs.push(Math.abs(a[i] - a[j]));

    const row = a.map(v => Math.floor((v - 1) / 7));
    const col = a.map(v => (v - 1) % 7);

    return {
      spread: a[5] - a[0],
      stdev: std(a),
      min_gap: Math.min(...gaps),
      consecutive_pairs: gaps.filter(g => g === 1).length,
      max_run: maxRun,
      zone_spread: std(zc),
      tail_unique: new Set(a.map(v => v % 10)).size,
      ac_value: new Set(diffs).size - (PICK - 1),
      max_same_col: maxCount(col),
      max_same_row: maxCount(row),
      max_diagonal: Math.max(maxCount(a.map((_, i) => row[i] - col[i])),
                             maxCount(a.map((_, i) => row[i] + col[i]))),
    };
  }

  /** 직전 1·2회차 당첨번호에 대한 맥락 특징. */
  function contextFeatures(c, prev1, prev2) {
    const p1 = new Set(prev1), p2 = new Set(prev2);
    const nb = new Set();
    for (const v of prev1) { if (v - 1 >= 1) nb.add(v - 1); if (v + 1 <= N_MAX) nb.add(v + 1); }
    return {
      prev_overlap: c.filter(v => p1.has(v)).length,
      prev2_overlap: c.filter(v => p2.has(v)).length,
      prev_neighbor: c.filter(v => nb.has(v)).length,
    };
  }

  /** 인기도 모델. build.py 가 내보낸 popularity.json 의 model 을 받는다. */
  class Model {
    constructor(m) {
      this.beta = m.beta;
      this.gamma = m.gamma;
      this.mean = m.feat_mean;
      this.std = m.feat_std;
      this.names = m.feature_names || FEATURES;
    }
    logMultiplier(combo, prev1, prev2) {
      let s = 0;
      for (const v of combo) s += this.beta[v - 1];
      const f = combFeatures(combo), g = contextFeatures(combo, prev1, prev2);
      const all = [...this.names.map(n => f[n]), ...CONTEXT.map(n => g[n])];
      for (let i = 0; i < all.length; i++)
        s += this.gamma[i] * (all[i] - this.mean[i]) / this.std[i];
      return s;
    }
    multiplier(combo, prev1, prev2) {
      return Math.exp(this.logMultiplier(combo, prev1, prev2));
    }
  }

  /* 번호를 k개 공유하는 두 티켓이 '둘 다 3개 이상 맞을' 확률.
     portfolio.collide_table() 이 계산한 값. meta.json 에 있으면 그걸 쓴다. */
  const COLLIDE_DEFAULT = [0.000063, 0.000423, 0.001648, 0.003853,
                           0.007403, 0.013495, 0.023925];

  /* 커버리지 항의 가중치. portfolio.py 의 COVERAGE_WEIGHT 와 반드시 같아야
     브라우저 폴백이 파이썬과 같은 조합을 만든다. */
  const COVERAGE_WEIGHT = 0.05;

  function overlap(a, b) {
    const s = new Set(a);
    return b.reduce((n, v) => n + (s.has(v) ? 1 : 0), 0);
  }

  /** 결정론적 난수 (mulberry32). 같은 시드는 항상 같은 결과를 준다. */
  function rngFrom(seed) {
    let s = seed | 0;
    return () => {
      s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** 시뮬레이티드 어닐링. portfolio.optimize() 와 같은 목적함수를 쓴다.
   *
   * 한 번에 티켓 하나만 바뀌므로 인기도와 겹침 비용을 증분으로 갱신한다.
   * 매번 전부 다시 계산하면 20장 기준으로 수십 초가 걸려 폴백이 무용지물이 된다.
   */
  function optimize(model, n, prev1, prev2, opts = {}) {
    const collide = opts.collide || COLLIDE_DEFAULT;
    const lamPop = opts.lamPop ?? 1.0;
    const lamCov = opts.lamCov ?? (COVERAGE_WEIGHT / collide[1]);
    const iters = opts.iters ?? 150000;
    const temp0 = opts.temp0 ?? 0.05;
    const rand = rngFrom(opts.seed ?? 1);
    const randInt = k => Math.floor(rand() * k);
    const nPairs = (n * (n - 1)) / 2;

    // 초기해: 45개 번호를 최대한 고르게 쓰도록 배치
    const tickets = [];
    for (let i = 0; i < n; i++) {
      const set = new Set();
      while (set.size < PICK) set.add(randInt(N_MAX) + 1);
      tickets.push([...set].sort((a, b) => a - b));
    }

    const inTicket = tickets.map(t => {           // 번호 포함 여부 비트맵
      const m = new Uint8Array(N_MAX + 1);
      for (const v of t) m[v] = 1;
      return m;
    });
    const ovOf = (bi, bj) => {
      let c = 0;
      for (let v = 1; v <= N_MAX; v++) if (bi[v] && bj[v]) c++;
      return c;
    };

    const pop = tickets.map(t => model.logMultiplier(t, prev1, prev2));
    const ov = [];
    for (let i = 0; i < n; i++) {
      ov.push(new Int8Array(n));
      for (let j = 0; j < i; j++) {
        const c = ovOf(inTicket[i], inTicket[j]);
        ov[i][j] = c; ov[j][i] = c;
      }
    }
    let popSum = pop.reduce((a, b) => a + b, 0);
    let covSum = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) covSum += collide[ov[i][j]];

    const costOf = () => lamPop * (popSum / n) + lamCov * (covSum / nPairs);
    let cur = costOf();
    let best = cur, bestTk = tickets.map(t => [...t]);

    const cand = new Uint8Array(N_MAX + 1);
    for (let it = 0; it < iters; it++) {
      const temp = temp0 * (1 - it / iters) + 1e-6;
      const i = randInt(n), j = randInt(PICK), v = randInt(N_MAX) + 1;
      const old = tickets[i][j];
      if (inTicket[i][v]) continue;               // 티켓 내 중복 방지

      const next = tickets[i].slice();
      next[j] = v;
      next.sort((a, b) => a - b);
      const newPop = model.logMultiplier(next, prev1, prev2);
      const dPop = (newPop - pop[i]) / n;

      cand.set(inTicket[i]);
      cand[old] = 0; cand[v] = 1;
      let dCov = 0;
      const newOv = new Int8Array(n);
      for (let k = 0; k < n; k++) {
        if (k === i) continue;
        const c = ovOf(cand, inTicket[k]);
        newOv[k] = c;
        dCov += collide[c] - collide[ov[i][k]];
      }
      dCov /= nPairs;

      const dCost = lamPop * dPop + lamCov * dCov;
      if (dCost < 0 || rand() < Math.exp(-dCost / temp)) {
        tickets[i] = next;
        popSum += newPop - pop[i];
        pop[i] = newPop;
        inTicket[i].set(cand);
        for (let k = 0; k < n; k++) {
          if (k === i) continue;
          ov[i][k] = newOv[k]; ov[k][i] = newOv[k];
        }
        covSum += dCov * nPairs;
        cur = costOf();
        if (cur < best) { best = cur; bestTk = tickets.map(t => [...t]); }
      }
    }
    return bestTk;
  }

  /** 인기도 백분위 계산용 참조 표본. */
  function percentileSampler(model, prev1, prev2, n = 20000, seed = 99) {
    const rand = rngFrom(seed);
    const vals = [];
    for (let i = 0; i < n; i++) {
      const set = new Set();
      while (set.size < PICK) set.add(Math.floor(rand() * N_MAX) + 1);
      vals.push(model.multiplier([...set].sort((a, b) => a - b), prev1, prev2));
    }
    vals.sort((a, b) => a - b);
    return m => {
      let lo = 0, hi = vals.length;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (vals[mid] < m) lo = mid + 1; else hi = mid; }
      return (lo / vals.length) * 100;
    };
  }

  return { N_MAX, PICK, TOTAL, FEATURES, CONTEXT, Model, combFeatures,
           contextFeatures, optimize, overlap, percentileSampler,
           COLLIDE_DEFAULT, COVERAGE_WEIGHT };
})();
