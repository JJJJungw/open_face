"""웹 UI (단일 HTML 문자열).

빌드 도구도 CDN 의존도 없이 파일 하나로 끝낸다. 서버가 파일을 못 찾는 사고를
없애려고 패키지 데이터가 아니라 파이썬 모듈 상수로 들고 있다.

화면은 대시보드 구성이다 — 좌측 사이드바(브랜드 · 내비 · 상태 박스), 상단
KPI 타일, 본문에 S3 브라우저와 작업 목록.
"""

INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>face-anonymizer</title>
<style>
  :root{
    /* 흰색 표면 + 파란색 강조. 상태색은 고정 팔레트를 쓰되, 밝은 표면에서는
       대비가 낮은 값이 있어 **글자는 항상 잉크 색**으로 두고 색은 점/테두리로만
       뜻을 거든다. */
    --bg:#f4f7fb; --surface:#ffffff; --raise:#eef4fc; --line:#e2e8f0;
    --fg:#0f172a; --dim:#526075; --faint:#94a3b8;
    --accent:#2a78d6;                     /* 흰 바탕 대비 4.3:1 */
    --accent-ink:#1c5aa8;                 /* 파란 글자용 (더 진한 단계) */
    --accent-dim:#e4eefb;                 /* 같은 램프의 밝은 단계 = 미터 트랙 */
    --accent-sel:#eff5fd;
    --good:#0ca30c; --warning:#fab219; --critical:#d03b3b;
    --critical-ink:#a82a2a; --critical-bg:#fdf1f1;
    --shadow:0 1px 2px rgba(15,23,42,.05), 0 1px 3px rgba(15,23,42,.04);
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:13.5px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
            "Helvetica Neue","Apple SD Gothic Neo","Noto Sans KR",sans-serif}

  /* ── 뼈대 ───────────────────────────────────────────────────────────── */
  .app{display:grid;grid-template-columns:216px 1fr;height:100%}
  .side{background:var(--surface);border-right:1px solid var(--line);
        display:flex;flex-direction:column;min-height:0}
  .brand{padding:18px 18px 14px;font-size:14px;font-weight:600;
         letter-spacing:-.01em;display:flex;align-items:center;gap:9px}
  .brand .mk{width:20px;height:20px;border-radius:6px;flex:none;
             background:linear-gradient(135deg,#2a78d6,#5b9dff)}
  nav{padding:4px 10px;flex:1;min-height:0;overflow:auto}
  nav a{display:flex;align-items:center;gap:9px;padding:8px 10px;border-radius:8px;
        color:var(--dim);text-decoration:none;font-size:13px;cursor:pointer}
  nav a:hover{background:var(--raise);color:var(--fg)}
  nav a.on{background:var(--accent-dim);color:var(--accent-ink);font-weight:600}
  nav .grp{font-size:10.5px;color:var(--faint);padding:14px 10px 5px;
           letter-spacing:.04em;text-transform:uppercase}

  main{min-width:0;overflow:auto;display:flex;flex-direction:column}
  .top{display:flex;align-items:center;gap:14px;padding:15px 24px;
       border-bottom:1px solid var(--line);position:sticky;top:0;
       background:rgba(255,255,255,.9);backdrop-filter:blur(8px);z-index:10}
  .top h1{font-size:16px;margin:0;font-weight:600;letter-spacing:-.01em;flex:1}
  .badge{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;
         color:var(--dim);background:var(--surface);border:1px solid var(--line);
         border-radius:20px;padding:3px 11px;white-space:nowrap}
  .content{padding:20px 24px 32px;display:flex;flex-direction:column;gap:18px}

  /* ── KPI 타일 ───────────────────────────────────────────────────────── */
  .kpis{display:grid;grid-template-columns:1.35fr 1fr 1fr 1fr;gap:12px}
  .tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;
        padding:15px 16px;min-width:0;box-shadow:var(--shadow)}
  .tile .lb{font-size:11.5px;color:var(--dim);margin-bottom:6px}
  .tile .v{font-size:23px;font-weight:600;letter-spacing:-.02em;line-height:1.15}
  .tile.hero{background:linear-gradient(180deg,#f7fbff,#ffffff);
             border-color:#cfe0f6}
  .tile.hero .lb{color:var(--accent-ink)}
  .tile.hero .v{font-size:44px;letter-spacing:-.03em;line-height:1.05;
                color:var(--accent-ink)}
  .tile .sub{font-size:11.5px;color:var(--faint);margin-top:5px}
  .tile .v small{font-size:14px;font-weight:500;color:var(--dim);margin-left:3px}

  /* ── 카드 ───────────────────────────────────────────────────────────── */
  .grid2{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(320px,1fr);
         gap:18px;align-items:start}
  .card{background:var(--surface);border:1px solid var(--line);border-radius:12px;
        min-width:0;display:flex;flex-direction:column;box-shadow:var(--shadow)}
  .card > h2{font-size:12.5px;margin:0;font-weight:600;color:var(--dim);
             padding:14px 16px;border-bottom:1px solid var(--line);
             display:flex;align-items:center;gap:9px}
  .card > h2 .cnt{margin-left:auto;font-weight:500;color:var(--faint);font-size:11.5px}
  .card .pad{padding:14px 16px}

  /* ── S3 브라우저 ────────────────────────────────────────────────────── */
  .bar{display:flex;align-items:center;gap:9px;flex-wrap:wrap;
       padding:12px 16px;border-bottom:1px solid var(--line)}
  .crumb{display:flex;align-items:center;gap:5px;font-size:12.5px;min-width:0;flex:1}
  .crumb a{color:var(--accent-ink);text-decoration:none;cursor:pointer;
           white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .crumb a:hover{text-decoration:underline}
  .crumb .sep{color:var(--faint);flex:none}
  .crumb .cur{color:var(--fg);font-weight:600;white-space:nowrap}
  .divider{width:1px;height:18px;background:var(--line);flex:none}
  .toggle{display:inline-flex;align-items:center;gap:6px;font-size:12px;
          color:var(--dim);cursor:pointer;white-space:nowrap;margin:0}
  .toggle:hover{color:var(--fg)}
  .toggle input{margin:0}
  .search{position:relative;flex:none}
  .search input{background:var(--bg);border:1px solid var(--line);color:var(--fg);
    border-radius:7px;padding:6px 10px 6px 26px;font-size:12.5px;width:170px}
  .search input:focus{outline:none;border-color:var(--accent);
    box-shadow:0 0 0 3px var(--accent-dim)}
  .search::before{content:'\2315';position:absolute;left:8px;top:5px;
                  color:var(--faint);font-size:13px}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  thead th{text-align:left;font-weight:600;font-size:11px;color:var(--faint);
           padding:8px 16px;border-bottom:1px solid var(--line);white-space:nowrap;
           letter-spacing:.02em;background:#fafcfe}
  tbody td{padding:8px 16px;border-bottom:1px solid #f1f5f9}
  tbody tr:last-child td{border-bottom:0}
  tbody tr:hover{background:#f8fbff}
  tbody tr.sel{background:var(--accent-sel)}
  td.num{text-align:right;font-variant-numeric:tabular-nums;color:var(--dim);
         white-space:nowrap}
  td.when{color:var(--dim);white-space:nowrap;font-variant-numeric:tabular-nums}
  .key{display:flex;align-items:center;gap:8px;min-width:0}
  .key .ico{flex:none;width:14px;text-align:center;color:var(--accent);font-size:11px}
  .key .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .key a{color:var(--accent-ink);text-decoration:none;cursor:pointer;font-weight:500}
  .key a:hover{text-decoration:underline}
  /* 상태 배지: 색은 점이 나르고 글자는 잉크 색으로 둔다 (밝은 표면 대비) */
  .tag{display:inline-flex;align-items:center;gap:5px;font-size:10.5px;
       padding:2px 8px;border-radius:20px;border:1px solid var(--line);
       color:var(--dim);white-space:nowrap;background:var(--surface)}
  .tag::before{content:'';width:5px;height:5px;border-radius:50%;
               background:var(--faint);flex:none}
  .tag.done{border-color:#bfe3bf}   .tag.done::before{background:var(--good)}
  .tag.run{border-color:#cfe0f6;color:var(--accent-ink)}
  .tag.run::before{background:var(--accent)}
  .tag.err{border-color:#f2c9c9;color:var(--critical-ink)}
  .tag.err::before{background:var(--critical)}
  .tag.plain::before{display:none}
  .empty{text-align:center;color:var(--faint);padding:38px 16px;font-size:12.5px}
  .bar .n{font-size:12.5px;color:var(--dim);margin-left:auto;white-space:nowrap}
  .bar .n b{color:var(--fg)}

  /* ── 폼 ─────────────────────────────────────────────────────────────── */
  .opts{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:11px}
  details > summary{cursor:pointer;font-size:12px;color:var(--dim);
                    list-style:none;display:flex;align-items:center;gap:7px}
  details > summary::before{content:'▸';color:var(--faint);font-size:10px}
  details[open] > summary::before{content:'▾'}
  details > summary:hover{color:var(--accent-ink)}
  details > summary #dsum{color:var(--faint);font-size:11.5px}
  label{display:block;font-size:11px;color:var(--faint);margin-bottom:4px}
  select,input[type=number]{width:100%;background:var(--surface);color:var(--fg);
    border:1px solid var(--line);border-radius:7px;padding:6px 8px;font-size:12.5px}
  select:focus,input[type=number]:focus{outline:none;border-color:var(--accent);
    box-shadow:0 0 0 3px var(--accent-dim)}
  .chk{display:flex;align-items:center;gap:7px;font-size:12.5px;padding-top:18px;
       color:var(--dim)}
  button{background:var(--accent);color:#fff;border:0;border-radius:8px;
    padding:8px 15px;font-size:13px;font-weight:600;cursor:pointer;white-space:nowrap}
  button:hover:not(:disabled){background:var(--accent-ink)}
  button:disabled{opacity:.45;cursor:not-allowed}
  button.ghost{background:var(--surface);color:var(--dim);
               border:1px solid var(--line);font-weight:500}
  button.ghost:hover:not(:disabled){background:var(--raise);color:var(--accent-ink);
    border-color:#cfe0f6}
  input[type=checkbox]{accent-color:var(--accent);width:14px;height:14px;cursor:pointer}

  /* ── 작업 목록 ──────────────────────────────────────────────────────── */
  .job{padding:13px 16px;border-bottom:1px solid #f1f5f9}
  .job:last-child{border-bottom:0}
  .jhead{display:flex;align-items:baseline;gap:9px}
  /* 목록에서만 지우는 것이라 '삭제' 라고 쓰면 S3 원본을 지우는 것으로 읽힌다.
     라벨 없는 x 로 두고 설명은 툴팁에 둔다. */
  .jx{flex:none;border:0;background:none;cursor:pointer;padding:0 2px;
      font-size:15px;line-height:1;color:var(--faint)}
  .jx:hover{color:var(--critical)}
  .jname{font-weight:600;font-size:12.5px;overflow:hidden;
         text-overflow:ellipsis;white-space:nowrap;flex:1}
  .bar2{height:6px;background:var(--accent-dim);border-radius:99px;
        overflow:hidden;margin:9px 0 7px}
  .bar2>i{display:block;height:100%;background:var(--accent);width:0;
          border-radius:99px;transition:width .35s ease}
  .job.ok .bar2>i{background:var(--good)}
  .meta{font-size:11.5px;color:var(--dim);font-variant-numeric:tabular-nums}
  .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:10px}
  .stats div{background:var(--bg);border-radius:7px;padding:6px 8px}
  .stats span{display:block;color:var(--faint);font-size:10.5px}
  .stats b{font-size:12.5px;font-weight:600}
  .warn{background:var(--critical-bg);border:1px solid #f2c9c9;border-radius:8px;
        padding:9px 11px;margin-top:10px;color:var(--critical-ink);
        font-size:11.5px;line-height:1.5}
  .warn b{display:block;margin-bottom:2px}
  a.dl{display:inline-block;background:var(--accent);color:#fff;text-decoration:none;
       border-radius:7px;padding:6px 13px;font-weight:600;font-size:12px}
  a.dl:hover{background:var(--accent-ink)}
  video{width:100%;border-radius:8px;margin-top:10px;background:#000}
  .row{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap}

  /* ── 좌측 하단 상태 박스 ────────────────────────────────────────────── */
  .hud{border-top:1px solid var(--line);padding:13px 14px;flex:none;background:#fafcfe}
  .hud .hd{display:flex;align-items:center;gap:8px;margin-bottom:11px}
  .hud .hd h3{font-size:11.5px;margin:0;font-weight:600;flex:1;letter-spacing:-.01em}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--faint);flex:none}
  .dot.ok{background:var(--good);box-shadow:0 0 0 3px rgba(12,163,12,.14)}
  .dot.run{background:var(--accent);box-shadow:0 0 0 3px var(--accent-dim);
           animation:pulse 1.5s ease-in-out infinite}
  .dot.err{background:var(--critical);box-shadow:0 0 0 3px rgba(208,59,59,.14)}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
  .ringwrap{position:relative;display:flex;align-items:center;justify-content:center;
            flex:none}
  .ringwrap .pct{position:absolute;font-size:12.5px;font-weight:600;
                 color:var(--accent-ink)}
  .ring{display:flex;align-items:center;gap:12px}
  .ring svg{transform:rotate(-90deg)}
  .ring circle{fill:none;stroke-width:5;stroke-linecap:round}
  .ring .trk{stroke:var(--accent-dim)}
  .ring .fil{stroke:var(--accent);transition:stroke-dasharray .4s ease}
  .who{min-width:0;flex:1}
  .who .nm{font-size:12px;font-weight:600;overflow:hidden;text-overflow:ellipsis;
           white-space:nowrap;margin-bottom:2px}
  .who .st{font-size:11px;color:var(--dim);font-variant-numeric:tabular-nums}
  .chip{display:inline-block;font-size:10px;padding:1px 6px;border-radius:20px;
        border:1px solid var(--line);color:var(--faint);margin-right:4px;
        background:var(--surface)}
  .chip.on{color:var(--accent-ink);border-color:#cfe0f6;background:var(--accent-dim)}
  .hgrid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:11px}
  .hgrid div{background:var(--surface);border:1px solid var(--line);
             border-radius:7px;padding:6px 8px}
  .hgrid span{display:block;color:var(--faint);font-size:10px}
  .hgrid b{font-size:12px;font-weight:600;font-variant-numeric:tabular-nums}
  .idle{color:var(--faint);font-size:11.5px;text-align:center;padding:4px 0 2px}

  @media (max-width:1080px){
    .kpis{grid-template-columns:1fr 1fr}
    .grid2{grid-template-columns:minmax(0,1fr)}
  }
  @media (max-width:760px){ .app{grid-template-columns:1fr} .side{display:none} }
</style>
</head>
<body>
<div class="app">
  <aside class="side">
    <div class="brand"><span class="mk"></span>face-anonymizer</div>
    <nav>
      <div class="grp">작업</div>
      <a class="on">파일 브라우저</a>
      <div class="grp">보기</div>
      <a onclick="setFilter('')" id="nav-all">전체</a>
      <a onclick="setFilter('running')" id="nav-running">수행중</a>
      <a onclick="setFilter('queued')" id="nav-queued">대기</a>
      <a onclick="setFilter('done')" id="nav-done">완료</a>
      <a onclick="setFilter('failed')" id="nav-failed">실패</a>
    </nav>
    <div class="hud">
      <div class="hd">
        <span class="dot" id="hdot"></span>
        <h3 id="htitle">연결 중…</h3>
      </div>
      <div id="hbody"></div>
    </div>
  </aside>

  <main>
    <div class="top">
      <h1 id="page">파일 브라우저</h1>
      <span class="badge" id="dev">—</span>
    </div>

    <div class="content">
      <div class="kpis" id="kpis"></div>

      <div class="grid2">
        <section class="card">
          <div class="bar">
            <div class="crumb" id="crumb"></div>
            <div class="search"><input id="q" placeholder="이름으로 검색" autocomplete="off"></div>
            <button class="ghost" onclick="loadObjects()">새로고침</button>
            <span class="divider"></span>
            <label class="toggle"><input type="checkbox" id="skipdone" checked><span>처리된 건 건너뛰기</span></label>
            <span class="n" id="picked">선택된 항목 없음</span>
            <button id="go" disabled>비식별화 시작</button>
          </div>
          <div id="browser"></div>
          <details class="pad" style="border-top:1px solid var(--line)">
            <summary>상세 설정 <span id="dsum"></span></summary>
            <div class="opts" style="margin-top:12px">
              <div><label>익명화 방식</label>
                <select id="method">
                  <option value="mosaic">모자이크</option>
                  <option value="blur">블러</option>
                  <option value="box">단색 박스</option>
                </select></div>
              <div><label>검출 임계값</label>
                <input type="number" id="conf" value="0.25" min="0.05" max="0.95" step="0.05"></div>
              <div><label>추론 해상도</label>
                <select id="imgsz">
                  <option value="960">960</option>
                  <option value="1280" selected>1280</option>
                  <option value="1600">1600</option>
                </select></div>
              <div><label>배치 크기</label>
                <input type="number" id="batch" value="16" min="1" max="64"></div>
              <div class="chk"><input type="checkbox" id="audio" checked><span>오디오 유지</span></div>
            </div>
          </details>
        </section>

        <section class="card">
          <h2>작업 <span class="cnt" id="jobcnt"></span></h2>
          <div id="jobs"></div>
        </section>
      </div>
    </div>
  </main>
</div>

<script>
const $ = s => document.querySelector(s);
const go = $('#go');
let timer = null, filter = '';

function fmt(s) {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), x = s % 60;
  return h ? `${h}:${String(m).padStart(2,'0')}:${String(x).padStart(2,'0')}`
           : `${m}:${String(x).padStart(2,'0')}`;
}
function humanSize(b) {
  if (b == null) return '—';
  const u = ['B','KB','MB','GB','TB']; let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return `${b.toFixed(i ? 1 : 0)} ${u[i]}`;
}
function humanTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso); if (isNaN(d)) return '—';
  const p = n => String(n).padStart(2,'0');
  return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())} `
       + `${p(d.getHours())}:${p(d.getMinutes())}`;
}

// ── S3 브라우저 ────────────────────────────────────────────────────────────
//
// 서버 계약:
//   GET /api/s3/objects?prefix=videos/2026-08/
//     -> { bucket, prefix, folders:["videos/2026-08/raw/"],
//          objects:[{key, size, modified, processed?}] }
//   POST /api/jobs (form) s3_key=<key> + 기존 옵션
//
// 연결 전이면 404 가 온다. 안내만 띄우고 직접 업로드로 쓸 수 있게 둔다.
const S3 = { bucket:'', prefix:'', folders:[], objects:[], selected:new Set(), error:null };
const VIDEO_RE = /\.(mp4|mov|mkv|avi|webm|m4v)$/i;

async function loadObjects(prefix) {
  if (prefix !== undefined) { S3.prefix = prefix; S3.selected.clear(); }
  try {
    const r = await fetch('/api/s3/objects?prefix=' + encodeURIComponent(S3.prefix));
    if (!r.ok) throw new Error(r.status === 404 ? 'not-configured' : 'HTTP ' + r.status);
    const d = await r.json();
    S3.bucket = d.bucket || ''; S3.folders = d.folders || [];
    S3.objects = d.objects || []; S3.error = null;
  } catch (e) {
    S3.error = e.message; S3.folders = []; S3.objects = [];
  }
  renderBrowser();
}

function crumbs() {
  const el = $('#crumb');
  if (!S3.bucket) { el.innerHTML = '<span class="cur">S3</span>'; return; }
  const parts = S3.prefix.split('/').filter(Boolean);
  let acc = '';
  const items = [`<a onclick="loadObjects('')">${S3.bucket}</a>`];
  parts.forEach((p, i) => {
    acc += p + '/'; const path = acc;
    items.push(i === parts.length - 1 ? `<span class="cur">${p}</span>`
                                      : `<a onclick="loadObjects('${path}')">${p}</a>`);
  });
  el.innerHTML = items.join('<span class="sep">/</span>');
}

function renderBrowser() {
  crumbs();
  const el = $('#browser');
  if (S3.error === 'not-configured') {
    el.innerHTML = `<div class="empty">S3 가 연결되지 않았습니다<br>
      <span style="font-size:11.5px">직접 업로드로 처리할 수 있습니다</span></div>`;
    return updatePicked();
  }
  if (S3.error) {
    el.innerHTML = `<div class="empty">목록을 불러오지 못했습니다 (${S3.error})</div>`;
    return updatePicked();
  }
  const q = ($('#q').value || '').toLowerCase();
  const folders = S3.folders.filter(f => f.toLowerCase().includes(q));
  const objects = S3.objects.filter(o => o.key.toLowerCase().includes(q));
  if (!folders.length && !objects.length) {
    el.innerHTML = '<div class="empty">항목이 없습니다</div>';
    return updatePicked();
  }
  // 폴더도 파일과 똑같이 체크해서 고른다. 폴더를 고르면 그 안의 영상 전부가
  // 들어간다 — 별도의 '폴더 전체 제출' 버튼을 두면 같은 일에 버튼이 둘이 된다.
  const selectable = folders.concat(objects.filter(o => VIDEO_RE.test(o.key))
                                           .map(o => o.key));
  const allSel = selectable.length && selectable.every(k => S3.selected.has(k));

  const rows = folders.map(f => {
    const name = f.replace(S3.prefix, '').replace(/\/$/, '');
    const sel = S3.selected.has(f);
    return `<tr class="${sel ? 'sel' : ''}">
      <td><input type="checkbox" ${sel ? 'checked' : ''}
            onchange="toggle('${f}', this.checked)"></td>
      <td><div class="key"><span class="ico">▸</span>
        <span class="nm"><a onclick="loadObjects('${f}')">${name}/</a></span></div></td>
      <td class="num">—</td><td class="when">—</td>
      <td><span class="tag plain">폴더</span></td></tr>`;
  }).concat(objects.map(o => {
    const name = o.key.replace(S3.prefix, '');
    const isVideo = VIDEO_RE.test(o.key), sel = S3.selected.has(o.key);
    const tag = o.processed ? '<span class="tag done">처리됨</span>'
              : (isVideo ? '' : '<span class="tag plain">영상 아님</span>');
    return `<tr class="${sel ? 'sel' : ''}">
      <td>${isVideo ? `<input type="checkbox" ${sel ? 'checked' : ''}
            onchange="toggle('${o.key}', this.checked)">` : ''}</td>
      <td><div class="key"><span class="ico">▤</span>
        <span class="nm" title="${o.key}">${name}</span></div></td>
      <td class="num">${humanSize(o.size)}</td>
      <td class="when">${humanTime(o.modified)}</td>
      <td>${tag}</td></tr>`;
  }));

  el.innerHTML = `<table><thead><tr>
      <th style="width:38px">${selectable.length
        ? `<input type="checkbox" ${allSel ? 'checked' : ''}
             onchange="toggleAll(this.checked)">` : ''}</th>
      <th>이름</th><th style="width:88px;text-align:right">크기</th>
      <th style="width:132px">마지막 수정</th><th style="width:86px"></th>
    </tr></thead><tbody>${rows.join('')}</tbody></table>`;
  updatePicked();
}

const isFolder = k => k.endsWith('/');
function toggle(key, on) { on ? S3.selected.add(key) : S3.selected.delete(key); renderBrowser(); }
function toggleAll(on) {
  const q = ($('#q').value || '').toLowerCase();
  S3.folders.filter(f => f.toLowerCase().includes(q))
    .forEach(f => on ? S3.selected.add(f) : S3.selected.delete(f));
  S3.objects.filter(o => VIDEO_RE.test(o.key) && o.key.toLowerCase().includes(q))
    .forEach(o => on ? S3.selected.add(o.key) : S3.selected.delete(o.key));
  renderBrowser();
}
function split() {
  const all = [...S3.selected];
  return { files: all.filter(k => !isFolder(k)), dirs: all.filter(isFolder) };
}
function updatePicked() {
  const { files, dirs } = split();
  const bits = [];
  if (files.length) bits.push(`영상 <b>${files.length}개</b>`);
  if (dirs.length) bits.push(`폴더 <b>${dirs.length}개</b>`);
  $('#picked').innerHTML = bits.length ? bits.join(' · ') + ' 선택됨'
                                       : '선택된 항목 없음';
  go.disabled = !bits.length;
}

// ── 제출 ───────────────────────────────────────────────────────────────────
// 서버가 RFC 9457 problem+json 을 준다. title/detail/hint 를 그대로 보여 주면
// "왜 안 되는지" 와 "무엇을 하면 되는지" 가 같이 전달된다.
function problemText(p) {
  if (!p || !p.title) return '알 수 없는 오류';
  return [p.title, p.detail, p.hint].filter(Boolean).join('\n');
}
function explain(status, text) {
  let p = null; try { p = JSON.parse(text); } catch (_) {}
  return p && p.title ? problemText(p) : (text || `HTTP ${status}`);
}
// 제출 진입점은 POST /api/jobs 하나다. 한 건이든 여러 건이든 폴더든 같은 요청,
// 같은 응답 — 화면에서도 분기가 필요 없다.
async function submitJobs(body, label) {
  go.disabled = true;
  const was = go.textContent;
  go.textContent = label;
  const r = await fetch('/api/jobs', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, params: paramObject() }),
  });
  go.textContent = was;

  if (r.status !== 202) { alert(explain(r.status, await r.text())); poll();
                          updatePicked(); return; }
  const d = await r.json();
  S3.selected.clear(); renderBrowser(); poll();
  if (d.rejected && d.rejected.length) {
    alert(`${d.accepted.length}건 접수 · ${d.rejected.length}건 거절\n\n`
      + d.rejected.map(x => `${x.s3_key}\n  ${problemText(x.error)}`).join('\n\n'));
  }
}

go.onclick = () => {
  const { files, dirs } = split();
  if (!files.length && !dirs.length) return;
  const skip = $('#skipdone').checked;

  // 폴더는 안에 몇 개가 들었는지 화면에 안 보인다. 그것만 한 번 확인받는다.
  if (dirs.length) {
    const lines = [`선택한 항목을 비식별화합니다.`, ''];
    dirs.forEach(d => lines.push(`폴더   ${d}`));
    if (files.length) lines.push(`영상   ${files.length}개`);
    lines.push('', `이미 처리된 건  ${skip ? '건너뜀' : '다시 처리'}`);
    if (!confirm(lines.join('\n'))) return;
  }
  const label = dirs.length ? '제출 중…' : `제출 중… ${files.length}건`;
  return submitJobs({ s3_keys: files, s3_prefix: dirs, skip_processed: skip },
                    label);
};

function paramObject() {
  return { method: $('#method').value, conf: +$('#conf').value,
           imgsz: +$('#imgsz').value, batch_size: +$('#batch').value,
           keep_audio: $('#audio').checked };
}
$('#q').addEventListener('input', renderBrowser);

function setFilter(f) {
  filter = f;
  for (const k of ['','running','queued','done','failed'])
    document.getElementById('nav-' + (k || 'all')).classList.toggle('on', k === f);
  poll();
}

// ── KPI · 상태 박스 · 작업 목록 ────────────────────────────────────────────
const RAD = 24, CIRC = 2 * Math.PI * RAD;
function ring(pct) {
  return `<div class="ringwrap">
    <svg width="58" height="58" viewBox="0 0 58 58">
      <circle class="trk" cx="29" cy="29" r="${RAD}"></circle>
      <circle class="fil" cx="29" cy="29" r="${RAD}"
              stroke-dasharray="${CIRC*pct/100} ${CIRC}"></circle>
    </svg><span class="pct">${pct}%</span></div>`;
}

function avgSeconds(jobs) {
  const d = jobs.filter(j => j.status === 'done' && j.result).slice(0, 5);
  return d.length ? d.reduce((a, j) => a + j.result.seconds, 0) / d.length : null;
}

// 작업 한 건에 걸리는 시간. 완료 기록이 있으면 그 평균이 제일 정확하고,
// 없으면 지금 도는 작업의 진행률로 되짚는다. 첫 배치에서는 완료가 하나도
// 없으니 평균만 보면 끝날 때까지 '계산 중' 이다 — 그게 이 화면의 버그였다.
function jobSeconds(jobs, running) {
  const avg = avgSeconds(jobs);
  if (avg) return avg;
  if (running && running.overall >= 5 && running.job_elapsed > 0)
    return running.job_elapsed * 100 / running.overall;
  return null;
}

function kpis(jobs, st) {
  const running = jobs.find(j => j.status === 'running');
  const queued = jobs.filter(j => j.status === 'queued');
  const done = jobs.filter(j => j.status === 'done');
  const failed = jobs.filter(j => j.status === 'failed');
  const avg = avgSeconds(jobs);
  const per = jobSeconds(jobs, running);
  const remain = (running ? 1 : 0) + queued.length;
  // 수행중 작업은 '이 작업이 끝나기까지'(job_eta), 대기 작업은 한 건 소요 x 건수.
  const etaAll = running && per != null
    ? (running.job_eta || 0) + queued.length * per : null;
  const rf = done.length && done[0].result ? done[0].result.realtime_factor : null;

  $('#kpis').innerHTML = `
    <div class="tile hero">
      <div class="lb">전체 남은 시간</div>
      <div class="v">${etaAll != null ? fmt(etaAll) : (remain ? '계산 중' : '없음')}</div>
      <div class="sub">${remain ? `남은 작업 ${remain}건` : '처리할 작업이 없습니다'}</div>
    </div>
    <div class="tile">
      <div class="lb">수행중 · 대기</div>
      <div class="v">${running ? 1 : 0}<small>/</small>${queued.length}</div>
      <div class="sub">${running ? running.name : '유휴'}</div>
    </div>
    <div class="tile">
      <div class="lb">완료</div>
      <div class="v">${done.length}${failed.length
        ? `<small style="color:var(--critical)"> · 실패 ${failed.length}</small>` : ''}</div>
      <div class="sub">${avg ? `평균 ${fmt(avg)}`
        : (per ? `추정 ${fmt(per)}` : '기록 없음')}</div>
    </div>
    <div class="tile">
      <div class="lb">처리 속도</div>
      <div class="v">${rf ? rf.toFixed(2) : '—'}<small>× 실시간</small></div>
      <div class="sub">${st && st.free_mb != null
        ? `디스크 여유 ${(st.free_mb/1024).toFixed(1)} GB` : ''}</div>
    </div>`;
}

function hud(jobs, st) {
  const dot = $('#hdot'), title = $('#htitle'), body = $('#hbody');
  if (!st) {
    dot.className = 'dot err'; title.textContent = '서버 연결 끊김';
    body.innerHTML = '<div class="idle">응답이 없습니다</div>'; return;
  }
  if (!st.ready) {
    dot.className = 'dot err'; title.textContent = '준비 중';
    body.innerHTML = `<div class="idle">${st.model_error || '모델을 올리는 중'}</div>`; return;
  }
  const running = jobs.find(j => j.status === 'running');
  const queued = jobs.filter(j => j.status === 'queued');
  const remain = (running ? 1 : 0) + queued.length;
  const avg = avgSeconds(jobs);

  dot.className = 'dot ' + (running ? 'run' : 'ok');
  title.textContent = running ? `수행중 · 남은 ${remain}건`
                              : (remain ? `대기 ${remain}건` : '유휴');
  if (!running) {
    body.innerHTML = `<div class="idle">${remain ? '곧 시작합니다' : '처리할 작업이 없습니다'}</div>
      <div class="hgrid">
        <div><span>대기</span><b>${queued.length}건</b></div>
        <div><span>디스크</span><b>${st.free_mb != null
          ? (st.free_mb/1024).toFixed(1) + ' GB' : '—'}</b></div>
      </div>`;
    return;
  }
  const stage = running.stage === 'detect' ? '검출' : running.stage === 'render' ? '렌더' : '준비';
  body.innerHTML = `
    <div class="ring">${ring(running.overall)}
      <div class="who">
        <div class="nm" title="${running.name}">${running.name}</div>
        <div class="st">
          <span class="chip ${running.stage==='detect'?'on':''}">검출</span>
          <span class="chip ${running.stage==='render'?'on':''}">렌더</span>
        </div>
        <div class="st" style="margin-top:3px">${stage} ${running.percent}% ·
          ${running.fps.toFixed(0)} f/s</div>
      </div></div>
    <div class="hgrid">
      <div><span>이 작업</span><b>${fmt(running.eta)}</b></div>
      <div><span>전체</span><b>${avg ? fmt(running.eta + queued.length*avg) : '—'}</b></div>
    </div>`;
}

const WARN_TEXT = { 'no-detections':
  '얼굴이 하나도 검출되지 않았습니다 — 원본이 그대로 출력됐습니다. 임계값을 낮추거나 영상 회전을 확인하세요.' };

function card(j) {
  const cls = j.status === 'done' ? 'ok' : '';
  const label = { queued:'대기', running:'수행중', done:'완료',
                  failed:'실패', cancelled:'취소됨' }[j.status] || j.status;
  const tagcls = j.status === 'done' ? 'done'
               : (j.status === 'failed' || j.status === 'cancelled') ? 'err'
               : j.status === 'running' ? 'run' : '';
  let body = '';
  if (j.status === 'running') {
    const stage = { transcode:'변환 → H.264', detect:'검출', render:'렌더' }[j.stage]
                || '준비';
    body = `<div class="bar2"><i style="width:${j.overall}%"></i></div>
      <div class="meta">${stage} ${j.percent}% · ${j.fps.toFixed(0)} f/s ·
        남은 시간 ${fmt(j.eta)}</div>
      <div class="row"><button class="ghost" onclick="cancel('${j.id}')">취소</button></div>`;
  } else if (j.status === 'queued') {
    const retry = j.attempts ? ` · 재시도 ${j.attempts}/${j.max_attempts}` : '';
    body = `<div class="bar2"><i style="width:0"></i></div>
      <div class="meta">앞에 ${j.queued_ahead}건${retry}</div>
      <div class="row"><button class="ghost" onclick="cancel('${j.id}')">취소</button></div>`;
  } else if (j.status === 'failed' || j.status === 'cancelled') {
    const e = j.error || {};
    body = `<div class="warn">
        <b>${e.title || '실패'}${j.attempts ? ` · ${j.attempts}회 시도` : ''}
           ${e.code ? `<span class="tag err" style="float:right">${e.code}</span>` : ''}</b>
        ${e.detail || ''}${e.hint ? `<br><span style="opacity:.8">${e.hint}</span>` : ''}
      </div>
      `;
  } else {
    const r = j.result, t = r.timing;
    const warns = (r.warnings || []).length
      ? `<div class="warn"><b>확인 필요</b>${
          r.warnings.map(w => WARN_TEXT[w] || w).join('<br>')}</div>` : '';
    body = `<div class="bar2"><i style="width:100%"></i></div>${warns}
      <div class="stats">
        <div><span>처리 시간</span><b>${fmt(r.seconds)}</b></div>
        <div><span>속도</span><b>${r.fps} f/s</b></div>
        <div><span>실시간 대비</span><b>${r.realtime_factor}×</b></div>
        <div><span>프레임</span><b>${r.frames}</b></div>
        <div><span>검출</span><b>${r.raw_boxes}</b></div>
        <div><span>보간</span><b>${r.filled_boxes}</b></div>
      </div>
      <div class="meta" style="margin-top:8px">검출 ${fmt(t.detect)} · 추적 ${fmt(t.track)}
        · 렌더 ${fmt(t.render)} · ${r.video.width}×${r.video.height} · ${r.method}</div>
      <div class="row">
        <a class="dl" href="/api/jobs/${j.id}/download">내려받기</a>
        <button class="ghost" onclick="preview('${j.id}')">미리보기</button>
      </div><div id="pv-${j.id}"></div>`;
  }
  // 끝난 작업만 지울 수 있다. 대기·수행중은 '취소' 가 따로 있고, 둘을 같은
  // 자리에 두면 무엇이 멈추고 무엇이 사라지는지 구분이 안 된다.
  const over = ['done', 'failed', 'cancelled'].includes(j.status);
  const x = over ? `<button class="jx" onclick="del('${j.id}')"
      title="목록에서 지웁니다. S3 의 원본과 결과물은 그대로 남습니다">&times;</button>` : '';
  return `<div class="job ${cls}" id="job-${j.id}">
    <div class="jhead">${x}<div class="jname" title="${j.name}">${j.name}</div>
      <span class="tag ${tagcls}">${label}</span></div>${body}</div>`;
}

function preview(id) {
  const el = document.getElementById('pv-' + id);
  el.innerHTML = el.innerHTML ? '' : `<video controls src="/api/jobs/${id}/download"></video>`;
}
async function del(id) { await fetch('/api/jobs/' + id, { method:'DELETE' }); poll(); }
async function cancel(id) {
  const r = await fetch(`/api/jobs/${id}/cancel`, { method:'POST' });
  if (!r.ok) alert(explain(r.status, await r.text()));
  poll();
}

async function poll() {
  const [jobs, st] = await Promise.all([
    fetch('/api/jobs').then(r => r.json()).catch(() => null),
    fetch('/api/status').then(r => r.json()).catch(() => null),
  ]);
  hud(jobs || [], st);
  if (!jobs) return;
  kpis(jobs, st);
  const shown = filter ? jobs.filter(j => j.status === filter) : jobs;
  $('#jobcnt').textContent = `${shown.length}건`;
  const open = [...document.querySelectorAll('video')].map(v => v.parentElement.id);
  $('#jobs').innerHTML = shown.length ? shown.map(card).join('')
    : '<div class="empty">작업이 없습니다</div>';
  open.forEach(pid => {
    const el = document.getElementById(pid);
    if (el && !el.innerHTML)
      el.innerHTML = `<video controls src="/api/jobs/${pid.slice(3)}/download"></video>`;
  });
  const busy = jobs.some(j => j.status === 'running' || j.status === 'queued');
  clearTimeout(timer);
  timer = setTimeout(poll, busy ? 700 : 5000);
}

fetch('/api/health').then(r => r.json()).then(h => {
  $('#dev').textContent = h.model_loaded
    ? `${h.device} · half=${h.half} · imgsz ${h.imgsz}` : '모델 준비 중';
}).catch(() => $('#dev').textContent = '연결 실패');

// 컨트롤 초깃값은 서버에서 받는다. 화면에 박아 두면 서버 설정을 바꿔도
// 화면은 옛 값을 보내서 둘이 조용히 어긋난다.
fetch('/api/defaults').then(r => r.json()).then(d => {
  if (d.method) $('#method').value = d.method;
  if (d.conf != null) $('#conf').value = d.conf;
  if (d.imgsz != null) {
    if (![...$('#imgsz').options].some(o => +o.value === d.imgsz))
      $('#imgsz').add(new Option(d.imgsz, d.imgsz));
    $('#imgsz').value = d.imgsz;
  }
  if (d.batch_size != null) $('#batch').value = d.batch_size;
  if (d.keep_audio != null) $('#audio').checked = !!d.keep_audio;
  $('#dsum').textContent =
    `(기본 ${d.method} · conf ${d.conf} · ${d.imgsz} · batch ${d.batch_size})`;
}).catch(() => { $('#dsum').textContent = '(서버 기본값을 쓰지 못함)'; });

setFilter('');
loadObjects();
</script>
</body>
</html>
"""
