const $ = id => document.getElementById(id);
const form = $('form');

function setMeta(m){
  if (!m) return;
  $('r2').textContent = m.r2_test ?? '—';
  $('rows').textContent = Number(m.rows ?? 0).toLocaleString();
  $('mae').textContent = m.mae ?? '0.35';

  if (Array.isArray(m.countries) && m.countries.length) {
    $('country').innerHTML = m.countries
      .map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`)
      .join('');
    if (m.countries.includes('India')) $('country').value = 'India';
  }
}

function escapeHtml(value){
  return String(value)
    .replaceAll('&','&amp;')
    .replaceAll('<','&lt;')
    .replaceAll('>','&gt;')
    .replaceAll('"','&quot;')
    .replaceAll("'","&#039;");
}

fetch('/api/meta')
  .then(r => {
    if (!r.ok) throw new Error('Metadata unavailable');
    return r.json();
  })
  .then(setMeta)
  .catch(() => {
    $('r2').textContent = '—';
    $('rows').textContent = '—';
  });

function describe(score){
  if (score < 5) {
    return {
      label:'Lower range',
      text:'The estimated score is in the lower portion of the model’s 1–10 range.'
    };
  }
  if (score < 7) {
    return {
      label:'Middle range',
      text:'The estimated score is around the middle portion of the model’s observed range.'
    };
  }
  return {
    label:'Higher range',
    text:'The estimated score is in the higher portion of the model’s 1–10 range.'
  };
}

function numericPayload(){
  const d = Object.fromEntries(new FormData(form));

  ['age','daily_unlocks'].forEach(k => d[k] = parseInt(d[k],10));
  ['avg_daily_usage_hours','study_hours','physical_activity_hours','sleep_hours_per_night']
    .forEach(k => d[k] = parseFloat(d[k]));

  return d;
}

form.addEventListener('submit', async e => {
  e.preventDefault();

  const err = $('err');
  err.hidden = true;

  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  const btn = $('go');
  btn.disabled = true;
  btn.textContent = 'Running prediction…';

  try {
    const res = await fetch('/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(numericPayload())
    });

    if (!res.ok) {
      let message = 'The prediction request could not be completed.';
      try {
        const body = await res.json();
        if (body.detail) message = Array.isArray(body.detail)
          ? body.detail.map(x => x.msg).join(', ')
          : body.detail;
      } catch {}
      throw new Error(message);
    }

    const data = await res.json();
    const score = Number(data.predicted_mental_health_score);

    if (!Number.isFinite(score)) throw new Error('The API returned an invalid prediction.');

    const safeScore = Math.min(10, Math.max(1, score));
    const info = describe(safeScore);

    $('empty').style.display = 'none';
    $('out').classList.add('show');

    $('num').textContent = safeScore.toFixed(1);
    $('band').textContent = info.label;
    $('insightText').textContent = info.text;

    requestAnimationFrame(() => {
      $('pin').style.left = ((safeScore - 1) / 9 * 100) + '%';
    });

    $('out').scrollIntoView({behavior:'smooth', block:'nearest'});
  } catch (ex) {
    err.textContent = ex.message || 'The server could not be reached. Please check that FastAPI is running.';
    err.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate wellbeing prediction →';
  }
});