/* 공통 유틸 — 데이터 로딩, 헤더/고지 렌더링, 최신 회차 폴백. */
const APP = (() => {
  const DATA = "data/";
  const MIRROR = "https://raw.githubusercontent.com/smok95/lotto/master/results/";

  const PAGES = [
    ["index.html", "추천 조합"],
    ["hypotheses.html", "가설 검증"],
    ["backtest.html", "백테스트"],
    ["data.html", "원시 데이터"],
  ];

  const fmt = {
    int: n => Math.round(n).toLocaleString("ko-KR"),
    won: n => Math.round(n).toLocaleString("ko-KR") + "원",
    eok: n => (n / 1e8).toFixed(1) + "억원",
    pct: (n, d = 2) => (n * 100).toFixed(d) + "%",
    sign: (n, d = 1) => (n >= 0 ? "+" : "") + (n * 100).toFixed(d) + "%",
    fix: (n, d = 3) => Number(n).toFixed(d),
    date: s => s,
  };

  async function load(name) {
    const r = await fetch(DATA + name + "?v=" + Date.now());
    if (!r.ok) throw new Error(`${name} 을(를) 불러오지 못했습니다 (HTTP ${r.status})`);
    return r.json();
  }

  function ballClass(n) {
    if (n <= 10) return "b1";
    if (n <= 20) return "b2";
    if (n <= 30) return "b3";
    if (n <= 40) return "b4";
    return "b5";
  }

  function balls(nums, { small = false, bonus = null } = {}) {
    const cls = small ? "ball sm " : "ball ";
    let h = `<div class="balls">` +
      nums.map(n => `<span class="${cls}${ballClass(n)}">${n}</span>`).join("");
    if (bonus != null) {
      h += `<span class="muted tiny" style="margin:0 2px">+</span>` +
           `<span class="${cls}${ballClass(bonus)} bonus">${bonus}</span>`;
    }
    return h + `</div>`;
  }

  function chrome(active) {
    const nav = PAGES.map(([href, label]) =>
      `<a href="${href}"${href === active ? ' aria-current="page"' : ""}>${label}</a>`).join("");
    document.body.insertAdjacentHTML("afterbegin", `
      <div class="topbar"><div class="topbar-inner">
        <div class="brand">로또 6/45 조합 분석 <span>· 통계 기반</span></div>
        <nav>${nav}</nav>
      </div></div>`);
  }

  const DISCLAIMER = `
    <div class="notice">
      <b>먼저 알아두실 것</b> — 로또 추첨은 매회 독립이며,
      <b>어떤 번호를 고르든 1등 당첨 확률은 8,145,060분의 1로 완전히 같습니다.</b>
      이 사이트는 그 확률을 바꾸지 못하며, 바꾼다고 주장하지도 않습니다.
      <ul>
        <li>이 사이트가 실제로 개선하는 것은 두 가지뿐입니다:
            <b>(1)</b> 1~3등은 당첨금을 당첨자끼리 나누므로, 남들이 덜 고르는 조합을 택하면
            <b>당첨됐을 때 받는 금액</b>이 커집니다.
            <b>(2)</b> 여러 장을 살 때 번호가 겹치지 않게 짜면
            <b>최소 1개 이상 당첨될 확률</b>이 올라갑니다(기대 당첨 개수는 그대로입니다).</li>
        <li>기대수익률은 어떤 방법을 써도 <b>약 -50%</b>입니다. 로또는 판매액의 절반만
            당첨금으로 돌려주는 구조이며, 이 사이트도 그것을 바꾸지 못합니다.</li>
        <li>복권은 <b>19세 미만 구매·양수가 불가</b>합니다.
            여윳돈 안에서만 이용하세요.</li>
      </ul>
      <div class="notice-foot">
        도박 문제로 어려움을 겪고 있다면 한국도박문제예방치유원 <b>1336</b> (국번없이, 24시간 무료).
      </div>
    </div>`;

  /** 미리 계산된 데이터보다 새로운 회차가 나왔는지 확인한다. */
  async function checkFresh(latest) {
    try {
      const r = await fetch(MIRROR + (latest + 1) + ".json", { cache: "no-store" });
      if (!r.ok) return null;
      const j = await r.json();
      if (!j || !Array.isArray(j.numbers)) return null;
      return { draw_no: j.draw_no, numbers: j.numbers.slice().sort((a, b) => a - b),
               bonus: j.bonus_no, date: (j.date || "").slice(0, 10) };
    } catch { return null; }        // 오프라인이거나 미러가 막혀도 조용히 넘어간다
  }

  function el(id) { return document.getElementById(id); }
  function set(id, html) { const e = el(id); if (e) e.innerHTML = html; }

  return { load, chrome, balls, ballClass, fmt, DISCLAIMER, checkFresh, el, set, PAGES };
})();
