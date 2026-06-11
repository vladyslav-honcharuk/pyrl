// Wires the UI to the backend and the 3D scene.
import { Lab } from './scene.js';
import { LearningCurve, ValueGrid, OptionGrid } from './charts.js';

const $ = sel => document.querySelector(sel);

const state = {
  task: null,
  preset: 'basic',
  dopamine: 0,
  infDopamine: 0,
  species: 'macaque',
  targetL: 0,
  targetR: 24,
  trialNo: 0,
  auto: false,
  trained: false,
};

const lab = new Lab($('#scene'));
const learn = new LearningCurve($('#learncurve'));
const valueGrid = new ValueGrid($('#valuegrid'));
let optionGrid = null;

async function boot() {
  const task = await (await fetch('/api/task')).json();
  state.task = task;
  state.trained = task.has_checkpoint;

  // Presets dropdown
  const presetSel = $('#preset');
  for (const [key, meta] of Object.entries(task.presets)) {
    const opt = document.createElement('option');
    opt.value = key; opt.textContent = meta.label;
    presetSel.appendChild(opt);
  }
  presetSel.value = state.preset;
  $('#preset-desc').textContent = task.presets[state.preset].description;

  // Option grid editor
  optionGrid = new OptionGrid($('#optiongrid'), task.color_vector, (l, r) => {
    state.targetL = l; state.targetR = r;
    updateOptionReadout();
    showStaticOptions();
  });
  updateOptionReadout();
  showStaticOptions();

  if (state.trained) enableTrialControls();
  connectStream();
  bindControls();
}

function optionDesc(idx) {
  const [p, m] = state.task.value_vector[idx];
  return { p, m, ev: p * m };
}

function riskLabel() {
  const a = optionDesc(state.targetL), b = optionDesc(state.targetR);
  // The "risky" option is the lower-probability / higher-magnitude one.
  if (Math.abs(a.p - b.p) < 1e-6) return null;
  return a.p < b.p ? 'L' : 'R';
}

function updateOptionReadout() {
  const a = optionDesc(state.targetL), b = optionDesc(state.targetR);
  const risky = riskLabel();
  $('#option-readout').innerHTML =
    `L[#${state.targetL}] p=${a.p} ×${a.m.toFixed(1)} EV=${a.ev.toFixed(2)}` +
    `${risky === 'L' ? ' (risky)' : ''}<br>` +
    `R[#${state.targetR}] p=${b.p} ×${b.m.toFixed(1)} EV=${b.ev.toFixed(2)}` +
    `${risky === 'R' ? ' (risky)' : ''}`;
}

function showStaticOptions() {
  lab.setOptions(state.task.color_vector[state.targetL],
                 state.task.color_vector[state.targetR]);
}

function connectStream() {
  const es = new EventSource('/api/train/stream');
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.type === 'state') onTrainState(m);
    else if (m.type === 'progress') onProgress(m);
  };
}

function onTrainState(m) {
  const el = $('#train-state');
  el.className = 'hint ' + m.state;
  if (m.state === 'training') {
    el.textContent = `Training ${state.task.presets[m.meta.preset].label} · kappa=${m.meta.kappa}…`;
    $('#train-btn').disabled = true;
    $('#stop-btn').disabled = false;
  } else if (m.state === 'done') {
    el.textContent = 'Training complete. Run trials to watch the agent.';
    $('#train-btn').disabled = false;
    $('#stop-btn').disabled = true;
    state.trained = true;
    enableTrialControls();
    refreshHeatmap();
  } else if (m.state === 'error') {
    el.textContent = 'Training failed — check the server console.';
    $('#train-btn').disabled = false; $('#stop-btn').disabled = true;
  } else {
    $('#train-btn').disabled = false; $('#stop-btn').disabled = true;
  }
}

function onProgress(rec) {
  if (rec.iter === 0 || (learn.data.length && rec.iter < learn.data[learn.data.length-1].iter)) {
    learn.reset();
  }
  learn.push(rec);
}

function enableTrialControls() {
  $('#trial-btn').disabled = false;
  $('#auto-btn').disabled = false;
}

function bindControls() {
  // Species
  $('#species').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    [...$('#species').children].forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    state.species = b.dataset.species;
    lab.setSpecies(state.species);
    showStaticOptions();
  });

  $('#preset').addEventListener('change', e => {
    state.preset = e.target.value;
    $('#preset-desc').textContent = state.task.presets[state.preset].description;
  });

  $('#dopamine').addEventListener('input', e => {
    state.dopamine = parseFloat(e.target.value);
    $('#dopamine-val').textContent = state.dopamine.toFixed(2);
  });
  $('#inf-dopamine').addEventListener('input', e => {
    state.infDopamine = parseFloat(e.target.value);
    $('#inf-dopamine-val').textContent = state.infDopamine.toFixed(2);
  });

  $('#train-btn').addEventListener('click', startTraining);
  $('#stop-btn').addEventListener('click', () => fetch('/api/train/stop', { method: 'POST' }));
  $('#trial-btn').addEventListener('click', () => runTrial());
  $('#auto-btn').addEventListener('click', toggleAuto);
}

async function startTraining() {
  learn.reset();
  await fetch('/api/train', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ preset: state.preset, dopamine: state.dopamine, seed: 1 }),
  });
}

async function runTrial() {
  if (!state.trained || lab.anim) return;
  const res = await (await fetch('/api/trial', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      target_l: state.targetL, target_r: state.targetR,
      inference_dopamine: state.infDopamine, preset: state.preset,
    }),
  })).json();
  if (!res.ok) { $('#train-state').textContent = res.message; return; }
  state.trialNo++;
  $('#hud-trial').textContent = `trial ${state.trialNo}`;
  playTrial(res.trial);
}

function playTrial(trial) {
  lab.playTrial(trial,
    (phase) => { $('#hud-phase').textContent = phase; },
    (t) => { onTrialDone(t); });
  $('#hud-choice').textContent = '…';
}

function onTrialDone(trial) {
  const risky = riskLabel();
  const choseRisky = risky && trial.choice === risky;
  const choiceTxt = trial.choice === 'L' ? 'LEFT' : (trial.choice === 'R' ? 'RIGHT' : 'no choice');
  $('#hud-choice').textContent = `${choiceTxt} · ${trial.reward > 0 ? '+' + trial.reward.toFixed(1) : 'no reward'}`;

  const tag = trial.choice
    ? `<span class="chip ${choseRisky ? 'risky' : 'safe'}">${choseRisky ? 'risky' : 'safe'}</span>`
    : '';
  $('#trial-readout').innerHTML = `
    <div class="big">${choiceTxt} ${tag}</div>
    <div>reward: <b>${trial.reward.toFixed(2)}</b></div>
    <div class="hint">LEFT EV ${trial.ev_l.toFixed(2)} · RIGHT EV ${trial.ev_r.toFixed(2)}</div>
    <div class="hint">value estimate: ${(trial.timeline.at(-1).value).toFixed(2)}</div>`;

  if (state.auto) setTimeout(() => runTrial(), 600);
}

function toggleAuto() {
  state.auto = !state.auto;
  $('#auto-btn').textContent = state.auto ? 'Stop auto' : 'Auto-run';
  $('#auto-btn').classList.toggle('active', state.auto);
  if (state.auto) runTrial();
}

async function refreshHeatmap() {
  try {
    const res = await (await fetch('/api/heatmap', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inference_dopamine: state.infDopamine, preset: state.preset }),
    })).json();
    if (res.ok) valueGrid.set(res.grid);
  } catch (e) { /* ignore */ }
}

boot();
