const board = document.getElementById('board');
async function refresh(){ board.innerHTML = await (await fetch('/viewstate')).text(); }

// Coalesce SSE-driven repaints: a thinking-delta storm is many frames per second — one trailing
// repaint per window keeps the board live without a /viewstate fetch per delta. The projection
// stays server-side (inv 2): this only re-fetches rendered HTML, never builds it.
let repaintTimer = null;
function scheduleRefresh(){
  if(repaintTimer) return;
  repaintTimer = setTimeout(async () => { repaintTimer = null; await refresh(); }, 120);
}

// Transient cues: a CSS class flicked on the wrapper, gone after the animation. Presentation
// only — durable truth is the re-fetched board (a reload owes these nothing).
function cue(cls){
  board.classList.remove(cls);
  void board.offsetWidth; // restart the animation if the class re-lands mid-flight
  board.classList.add(cls);
  setTimeout(() => board.classList.remove(cls), 600);
}

board.addEventListener('click', async (ev) => {
  const btn = ev.target.closest('button[data-verdict]');
  if(!btn) return;
  const verdict = btn.dataset.verdict;
  const rEl = btn.closest('.prey').querySelector('.reason');
  const note = rEl ? rEl.value.trim() : '';
  if(verdict === 'release' && !note){ rEl.focus(); return; }
  const body = {prey_id: btn.dataset.preyId, verdict};
  if(verdict === 'release') body.reason = note;
  if(verdict === 'amber' && note) body.provenance = note;
  await fetch('/verdict', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)});
  refresh(); // a human action repaints immediately — no debounce
});

(async () => {
  await refresh();
  const snap = await (await fetch('/snapshot')).json();
  const feed = document.getElementById('feed');
  const es = new EventSource('/events?since=' + snap.latest_id);
  es.onmessage = (e) => {
    feed.textContent += e.data + '\n';
    feed.scrollTop = feed.scrollHeight;
    try {
      const frame = JSON.parse(e.data); // cue routing only — never rendered as markup (inv 3)
      if(frame.type === 'error') cue('damage');
      else if(frame.type === 'run_finished' || frame.type === 'verdict') cue('pulse');
    } catch { /* keep-alive / non-JSON payloads: no cue */ }
    scheduleRefresh();
  };
})();
