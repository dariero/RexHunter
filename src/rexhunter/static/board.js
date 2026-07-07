const board = document.getElementById('board');
async function refresh(){ board.innerHTML = await (await fetch('/viewstate')).text(); }
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
  refresh();
});
(async () => {
  await refresh();
  const snap = await (await fetch('/snapshot')).json();
  const feed = document.getElementById('feed');
  const es = new EventSource('/events?since=' + snap.latest_id);
  es.onmessage = (e) => {
    feed.textContent += e.data + '\n';
    feed.scrollTop = feed.scrollHeight;
    refresh();
  };
})();
