"""웹 UI (단일 HTML 문자열).

빌드 도구도 CDN 의존도 없이 파일 하나로 끝낸다. 서버가 파일을 못 찾는 사고를
없애려고 패키지 데이터가 아니라 파이썬 모듈 상수로 들고 있다.
"""

INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>face-anonymizer</title>
<style>
  :root{
    --bg:#0f1115; --panel:#171a21; --line:#252a34; --fg:#e6e8ec;
    --muted:#8b93a1; --accent:#5b9dff; --ok:#3ecf8e; --err:#ff6b6b;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
            "Helvetica Neue","Apple SD Gothic Neo","Noto Sans KR",sans-serif}
  .wrap{max-width:860px;margin:0 auto;padding:32px 20px 64px}
  h1{font-size:20px;margin:0 0 4px;letter-spacing:-.01em}
  .sub{color:var(--muted);font-size:13px;margin-bottom:24px}
  .sub b{color:var(--fg);font-weight:600}
  .panel{background:var(--panel);border:1px solid var(--line);
         border-radius:12px;padding:18px}
  #drop{border:1.5px dashed var(--line);border-radius:10px;padding:34px 16px;
        text-align:center;cursor:pointer;transition:.15s;color:var(--muted)}
  #drop:hover,#drop.over{border-color:var(--accent);color:var(--fg);
                         background:rgba(91,157,255,.06)}
  #drop b{color:var(--fg)}
  .opts{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
        gap:14px;margin-top:18px}
  label{display:block;font-size:12px;color:var(--muted);margin-bottom:5px}
  select,input[type=number]{width:100%;background:#0e1116;color:var(--fg);
    border:1px solid var(--line);border-radius:7px;padding:7px 9px;font-size:13px}
  .chk{display:flex;align-items:center;gap:7px;font-size:13px;padding-top:20px}
  button{background:var(--accent);color:#04070d;border:0;border-radius:8px;
    padding:10px 18px;font-size:14px;font-weight:600;cursor:pointer}
  button:disabled{opacity:.45;cursor:not-allowed}
  button.ghost{background:transparent;color:var(--muted);
    border:1px solid var(--line);font-weight:500}
  .row{display:flex;gap:10px;align-items:center;margin-top:16px}
  .job{background:var(--panel);border:1px solid var(--line);border-radius:12px;
       padding:16px 18px;margin-top:14px}
  .jhead{display:flex;justify-content:space-between;align-items:baseline;gap:12px}
  .jname{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .badge{font-size:11px;padding:2px 8px;border-radius:20px;border:1px solid var(--line);
         color:var(--muted);white-space:nowrap}
  .badge.run{color:var(--accent);border-color:var(--accent)}
  .badge.done{color:var(--ok);border-color:var(--ok)}
  .badge.err{color:var(--err);border-color:var(--err)}
  .bar{height:6px;background:#0b0e13;border-radius:99px;overflow:hidden;margin:12px 0 8px}
  .bar>i{display:block;height:100%;background:var(--accent);width:0;
         transition:width .3s ease}
  .job.done .bar>i{background:var(--ok)}
  .meta{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));
         gap:10px;margin-top:12px;font-size:12px}
  .stats div{background:#0e1116;border-radius:8px;padding:8px 10px}
  .stats span{display:block;color:var(--muted);font-size:11px}
  .stats b{font-size:15px;font-weight:600;font-variant-numeric:tabular-nums}
  video{width:100%;border-radius:8px;margin-top:12px;background:#000}
  .err{color:var(--err);font-size:13px;margin-top:8px;word-break:break-all}
  .warn{background:rgba(255,107,107,.1);border:1px solid var(--err);
        border-radius:8px;padding:10px 12px;margin-top:12px;
        color:var(--err);font-size:12.5px;line-height:1.5}
  .warn b{display:block;margin-bottom:3px}
  a.dl{display:inline-block;background:var(--ok);color:#04140d;text-decoration:none;
       border-radius:8px;padding:8px 15px;font-weight:600;font-size:13px}
</style>
</head>
<body>
<div class="wrap">
  <h1>face-anonymizer</h1>
  <div class="sub" id="health">서버 확인 중…</div>

  <div class="panel">
    <div id="drop">
      <b>영상을 끌어다 놓거나 클릭</b>해서 선택<br>
      <span style="font-size:12px">mp4 · mov · mkv · avi · webm</span>
    </div>
    <input type="file" id="file" accept="video/*" hidden>

    <div class="opts">
      <div>
        <label>익명화 방식</label>
        <select id="method">
          <option value="mosaic">모자이크</option>
          <option value="blur">블러</option>
          <option value="box">단색 박스</option>
        </select>
      </div>
      <div>
        <label>검출 임계값 (conf)</label>
        <input type="number" id="conf" value="0.25" min="0.05" max="0.95" step="0.05">
      </div>
      <div>
        <label>추론 해상도</label>
        <select id="imgsz">
          <option value="960">960</option>
          <option value="1280" selected>1280</option>
          <option value="1600">1600</option>
        </select>
      </div>
      <div>
        <label>배치 크기</label>
        <input type="number" id="batch" value="16" min="1" max="64" step="1">
      </div>
      <div class="chk"><input type="checkbox" id="audio" checked><span>오디오 유지</span></div>
    </div>

    <div class="row">
      <button id="go" disabled>처리 시작</button>
      <span class="meta" id="picked">선택된 파일 없음</span>
    </div>
  </div>

  <div id="jobs"></div>
</div>

<script>
const $ = s => document.querySelector(s);
const drop = $('#drop'), fileInput = $('#file'), go = $('#go');
let picked = null, timer = null;

fetch('/api/health').then(r => r.json()).then(h => {
  $('#health').innerHTML = h.model_loaded
    ? `모델 로드됨 · <b>${h.device}</b> · half=${h.half} · imgsz ${h.imgsz}`
    : `대기 중 · device <b>${h.device}</b> · 첫 요청 때 모델을 올립니다`;
}).catch(() => $('#health').textContent = '서버에 연결할 수 없습니다');

drop.onclick = () => fileInput.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => {
  e.preventDefault(); drop.classList.remove('over');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
};
fileInput.onchange = () => fileInput.files.length && setFile(fileInput.files[0]);

function setFile(f) {
  picked = f;
  $('#picked').textContent = `${f.name} · ${(f.size / 1048576).toFixed(1)} MB`;
  go.disabled = false;
}

function fmt(s) {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), x = s % 60;
  return h ? `${h}:${String(m).padStart(2,'0')}:${String(x).padStart(2,'0')}`
           : `${m}:${String(x).padStart(2,'0')}`;
}

go.onclick = () => {
  if (!picked) return;
  const fd = new FormData();
  fd.append('file', picked);
  fd.append('method', $('#method').value);
  fd.append('conf', $('#conf').value);
  fd.append('imgsz', $('#imgsz').value);
  fd.append('batch_size', $('#batch').value);
  fd.append('keep_audio', $('#audio').checked);

  go.disabled = true; go.textContent = '업로드 중… 0%';
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/jobs');
  xhr.upload.onprogress = e => {
    if (e.lengthComputable)
      go.textContent = `업로드 중… ${Math.round(100 * e.loaded / e.total)}%`;
  };
  xhr.onload = () => {
    go.textContent = '처리 시작'; go.disabled = false;
    if (xhr.status === 202) { poll(); }
    else {
      let m = xhr.responseText;
      try { m = JSON.parse(m).detail; } catch (_) {}
      if (xhr.status === 429) m = '대기열이 가득 찼습니다. 잠시 후 다시 시도하세요.';
      else if (xhr.status === 503) m = '서버가 아직 준비되지 않았습니다. ' + m;
      alert(m);
    }
  };
  xhr.onerror = () => { go.textContent = '처리 시작'; go.disabled = false;
                        alert('업로드 실패'); };
  xhr.send(fd);
};

function card(j) {
  const cls = j.status === 'done' ? 'done'
            : j.status === 'failed' ? 'err'
            : j.status === 'running' ? 'run' : '';
  const label = { queued: '대기', running: '수행중', done: '완료', failed: '실패' }[j.status];
  let body = '';
  if (j.status === 'running') {
    const stage = j.stage === 'detect' ? '검출' : j.stage === 'render' ? '렌더' : '준비';
    body = `<div class="bar"><i style="width:${j.overall}%"></i></div>
      <div class="meta">${stage} ${j.percent}% · 전체 ${j.overall}% ·
        ${j.fps.toFixed(1)} f/s · 남은 시간 ${fmt(j.eta)}</div>`;
  } else if (j.status === 'queued') {
    const retry = j.attempts ? ` · 재시도 ${j.attempts}/${j.max_attempts}` : '';
    body = `<div class="bar"><i style="width:0"></i></div>
      <div class="meta">앞에 ${j.queued_ahead}건 대기 중${retry}</div>`;
  } else if (j.status === 'failed') {
    body = `<div class="err">${j.error} (${j.attempts}회 시도)</div>`;
  } else {
    const r = j.result, t = r.timing;
    const W = { 'no-detections':
                  '얼굴이 하나도 검출되지 않았습니다 — 원본이 그대로 출력됐습니다. '
                  + '임계값(conf)을 낮추거나 영상 회전을 확인하세요.' };
    const warn = (r.warnings || []).length ? `<div class="warn"><b>확인 필요</b>${
        r.warnings.map(w => W[w] || w).join('<br>')}</div>` : '';
    body = `<div class="bar"><i style="width:100%"></i></div>${warn}
      <div class="stats">
        <div><span>처리 시간</span><b>${fmt(r.seconds)}</b></div>
        <div><span>처리 속도</span><b>${r.fps} f/s</b></div>
        <div><span>실시간 대비</span><b>${r.realtime_factor}x</b></div>
        <div><span>프레임</span><b>${r.frames}</b></div>
        <div><span>검출 박스</span><b>${r.raw_boxes}</b></div>
        <div><span>보간 박스</span><b>${r.filled_boxes}</b></div>
        <div><span>검출된 프레임</span><b>${(r.detection_rate*100).toFixed(1)}%</b></div>
      </div>
      <div class="meta" style="margin-top:10px">
        검출 ${fmt(t.detect)} · 추적 ${fmt(t.track)} · 렌더 ${fmt(t.render)} ·
        오디오 ${fmt(t.audio)} (${r.audio}) · ${r.video.width}x${r.video.height}
        @${r.video.fps}fps · ${r.method}</div>
      <div class="row">
        <a class="dl" href="/api/jobs/${j.id}/download">내려받기</a>
        <button class="ghost" onclick="preview('${j.id}')">미리보기</button>
        <button class="ghost" onclick="del('${j.id}')">삭제</button>
      </div>
      <div id="pv-${j.id}"></div>`;
  }
  return `<div class="job ${cls}" id="job-${j.id}">
    <div class="jhead"><div class="jname">${j.name}</div>
      <span class="badge ${cls}">${label}</span></div>${body}</div>`;
}

function preview(id) {
  const el = document.getElementById('pv-' + id);
  el.innerHTML = el.innerHTML
    ? '' : `<video controls src="/api/jobs/${id}/download"></video>`;
}

async function del(id) {
  await fetch('/api/jobs/' + id, { method: 'DELETE' });
  poll();
}

async function poll() {
  const jobs = await fetch('/api/jobs').then(r => r.json()).catch(() => null);
  if (!jobs) return;
  const open = [...document.querySelectorAll('video')].map(v => v.parentElement.id);
  $('#jobs').innerHTML = jobs.map(card).join('');
  open.forEach(preview_restore);
  const busy = jobs.some(j => j.status === 'running' || j.status === 'queued');
  clearTimeout(timer);
  timer = setTimeout(poll, busy ? 700 : 5000);
}
function preview_restore(pid) {
  const el = document.getElementById(pid);
  if (el && !el.innerHTML)
    el.innerHTML = `<video controls src="/api/jobs/${pid.slice(3)}/download"></video>`;
}
poll();
</script>
</body>
</html>
"""
