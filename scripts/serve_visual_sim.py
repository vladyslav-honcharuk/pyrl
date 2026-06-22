#!/usr/bin/env python
"""
Bridge server between the `visual_sim` 3D environment (browser) and the pyrl
training/inference code (PyTorch).

The browser cannot import PyTorch, so this small stdlib-only server wraps the
existing `pyrl` code and exposes it over HTTP/JSON. It also serves the static
3D app so everything lives on one origin (no CORS).

Run:
    python scripts/serve_visual_sim.py
    # then open http://localhost:8000

This is being built up incrementally. Endpoints so far:
    GET /                 -> the visual_sim app (static files)
    GET /api/models       -> list of trained weight checkpoints
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# Repo root = two levels up from this file (scripts/serve_visual_sim.py).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

# Where the 3D app lives. Override with VISUAL_SIM_DIR if you moved it.
VISUAL_SIM_DIR = os.environ.get(
    'VISUAL_SIM_DIR',
    os.path.expanduser('~/Desktop/visual_sim'),
)

# Port: argv[1] overrides $PORT overrides default.
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get('PORT', '8000'))

# Task spec that every gambling checkpoint was trained against.
TASK_FILE = os.path.join(REPO_ROOT, 'tasks', 'gambling.py')

# Model pre-selected in the picker (relative to REPO_ROOT). Override with env.
DEFAULT_MODEL = os.environ.get(
    'DEFAULT_MODEL',
    'data_progression3/d1d2_plasticity_opal04/weights/'
    'gambling_d1d2_plasticity_pos_reg_opal04/'
    'gambling_d1d2_plasticity_pos_reg_opal04.pkl',
)

# Currently loaded model, shared across request threads. Inference is not yet
# wired up; for now we just hold the trainer + its metadata.
ACTIVE = {'pg': None, 'model': None, 'meta': None, 'path': None}
ACTIVE_LOCK = threading.Lock()


def find_model_checkpoints():
    """Find trained policy/baseline weight checkpoints under data*/ dirs.

    A checkpoint lives at `<data_root>/.../weights/<name>/<name>.pkl`. We skip
    `*_copy.pkl` (scratch copies made by the `run` action) and the unrelated
    trial-activity dumps under `.../trials/`.
    """
    models = []
    for entry in sorted(os.listdir(REPO_ROOT)):
        if not entry.startswith('data'):
            continue
        data_root = os.path.join(REPO_ROOT, entry)
        if not os.path.isdir(data_root):
            continue
        for dirpath, _dirnames, filenames in os.walk(data_root):
            # Only descend into the weights subtree.
            if os.sep + 'weights' + os.sep not in dirpath + os.sep:
                continue
            for fn in filenames:
                if not fn.endswith('.pkl') or fn.endswith('_copy.pkl'):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, REPO_ROOT)
                # Group = the path segment just under the data root, useful for
                # the picker (e.g. "hardwired_kappa_0.5", "d1d2_vmod", ...).
                tail = os.path.relpath(full, data_root).split(os.sep)
                group = tail[0] if len(tail) > 1 else entry
                models.append({
                    'name': os.path.splitext(fn)[0],
                    'path': rel,
                    'data_root': entry,
                    'group': group,
                    'size': os.path.getsize(full),
                    'mtime': os.path.getmtime(full),
                    'is_default': rel == DEFAULT_MODEL,
                })
    models.sort(key=lambda m: m['name'])
    return models


def _resolve_checkpoint(rel_path):
    """Validate a client-supplied checkpoint path and return its absolute form.

    Guards against path traversal: the resolved file must live under REPO_ROOT
    and be an existing .pkl.
    """
    if not rel_path:
        rel_path = DEFAULT_MODEL
    full = os.path.abspath(os.path.join(REPO_ROOT, rel_path))
    if not full.startswith(REPO_ROOT + os.sep):
        raise ValueError('path escapes repo root')
    if not full.endswith('.pkl') or not os.path.isfile(full):
        raise ValueError(f'not a checkpoint file: {rel_path}')
    return full


def load_model(rel_path):
    """Load a checkpoint into a trainer and extract display metadata.

    Returns (pg, meta). Wraps the same Model/get_pg path the train.py `info`
    action uses, so behavior matches the rest of the project.
    """
    from pyrl.model import Model

    full = _resolve_checkpoint(rel_path)
    rel = os.path.relpath(full, REPO_ROOT)

    model = Model(TASK_FILE)
    pg = model.get_pg(full, seed=1, device='cpu')

    inputs = list(pg.config.get('inputs', {}).keys())
    save = getattr(pg, 'save', {}) or {}
    meta = {
        'name': os.path.splitext(os.path.basename(full))[0],
        'path': rel,
        'network_type': pg.config.get('network_type'),
        'N': pg.config.get('N'),
        'kappa': float(getattr(pg, 'kappa', 0.0)),
        'kappa_mode': getattr(pg, 'kappa_mode', None),
        'opponent_modulation': bool(pg.config.get('use_opponent_modulation', False)),
        'pathway_specific_plasticity': bool(
            pg.config.get('pathway_specific_plasticity', False)),
        'inputs': inputs,
        'has_context': 'CONTEXT' in inputs,
        'actions': list(pg.config.get('actions', {}).keys()),
        'best_iter': save.get('best_iter'),
        'best_reward': save.get('best_reward'),
    }
    return pg, meta


def run_one_trial(pg, target_l, target_r, context=0.0):
    """Run a single real trial of the loaded model and summarize the outcome.

    target_l / target_r are option indices 0-24 into the task's 5x5 grid
    (row = success probability, col = reward magnitude).

    context is the tonic-dopamine level c fed to the network at runtime. It
    drives D1/D2 opponent modulation, shifting risk preference: c < 0 is
    risk-averse, c > 0 risk-seeking. This is the main runtime risk switch.
    """
    trial = pg.task.get_condition(
        pg.rng, pg.dt, context={'target_l': int(target_l), 'target_r': int(target_r)})
    results = pg.run_trials([trial], return_states=True, context_input=float(context))

    perf = results['perf']
    choice = perf.choices[0]            # 'L', 'R', or None (no decision)
    t_choice = perf.t_choices[0]        # timestep index of the choice, or None
    correct = perf.corrects[0]

    R, M, Z_b = results['R'], results['M'], results['Z_b']
    reward = float((R[:, 0] * M[:, 0]).sum().item())

    # Critic's predicted value approaching the decision. The terminal choice
    # step is masked (Z_b == 0 there), so we read the step just before the
    # commit, which carries the critic's expectation for the trial.
    Tmax = Z_b.shape[0]
    if t_choice is not None:
        v_idx = max(0, min(int(t_choice) - 1, Tmax - 1))
    else:
        valid = int(M[:, 0].sum().item())
        v_idx = max(0, valid - 1)
    value = float(Z_b[v_idx, 0].item())

    ev_l = float(trial['prob_l'] * trial['size_l'])
    ev_r = float(trial['prob_r'] * trial['size_r'])

    # --- timing trajectory: real task epochs + where the net committed -------
    dur = trial['durations']
    fixation_ms = float(dur['fixation'][1] - dur['fixation'][0])
    stimulus_ms = float(dur['stimulus'][1] - dur['stimulus'][0])
    decision_ms = float(dur['decision'][1] - dur['decision'][0])
    decision_idx = trial['epochs']['decision']
    dec_start = int(decision_idx[0])
    dec_len = max(1, len(decision_idx))

    commit_frac = None
    t_choice_ms = None
    if t_choice is not None:
        # Fraction through the decision window at which the action committed.
        commit_frac = max(0.0, min(1.0, (int(t_choice) - dec_start) / dec_len))
        t_choice_ms = float(int(t_choice) * pg.dt)

    # --- developing preference (gaze lean) over stimulus -> commit -----------
    # Signed relative policy preference (pL - pR)/(pL + pR) per timestep, +1 = L.
    # This rises toward the chosen side as the decision forms, so the eyes can
    # track it. The terminal choice step is masked (Z == 0), so we stop before.
    actions = pg.config.get('actions', {})
    iL, iR = actions.get('CHOOSE-LEFT'), actions.get('CHOOSE-RIGHT')
    Z = results['Z']
    stim_onset = int(trial['epochs']['stimulus'][0])
    if t_choice is not None:
        gaze_end = max(stim_onset, int(t_choice) - 1)
    else:
        gaze_end = int(trial['epochs']['decision'][-1]) - 1

    raw_lean = []
    if iL is not None and iR is not None:
        for tt in range(stim_onset, gaze_end + 1):
            pL = float(Z[tt, 0, iL]); pR = float(Z[tt, 0, iR])
            raw_lean.append(max(-1.0, min(1.0, (pL - pR) / (pL + pR + 1e-6))))

    # Resample to a fixed length the scene can index by phase fraction.
    N = 24
    if not raw_lean:
        gaze_lean = [0.0] * N
    elif len(raw_lean) == 1:
        gaze_lean = [raw_lean[0]] * N
    else:
        gaze_lean = []
        for i in range(N):
            f = i / (N - 1) * (len(raw_lean) - 1)
            lo = int(f); hi = min(lo + 1, len(raw_lean) - 1)
            gaze_lean.append(round(raw_lean[lo] * (1 - (f - lo)) + raw_lean[hi] * (f - lo), 3))

    trajectory = {
        'decided': choice in ('L', 'R'),
        'fixation_ms': fixation_ms,
        'stimulus_ms': stimulus_ms,
        'decision_ms': decision_ms,
        't_choice_ms': t_choice_ms,     # ms from trial start to commit
        'commit_frac': commit_frac,     # 0..1 within the decision window
        'gaze_lean': gaze_lean,         # 24 samples, +1 = leaning LEFT
    }

    return {
        'target_l': int(target_l),
        'target_r': int(target_r),
        'choice': choice,                 # 'L' | 'R' | None
        'decided': choice in ('L', 'R'),
        't_choice': None if t_choice is None else int(t_choice),
        'correct': bool(correct),
        'rewarded': reward > 0,
        'reward': reward,
        'value': value,
        'prob_l': float(trial['prob_l']), 'size_l': float(trial['size_l']), 'ev_l': ev_l,
        'prob_r': float(trial['prob_r']), 'size_r': float(trial['size_r']), 'ev_r': ev_r,
        'trajectory': trajectory,
    }


def compute_value_grid(pg, context=0.0, repeats=2):
    """Critic's predicted value for each of the 25 options at tonic-DA `context`.

    Each option is shown on both sides (target_l == target_r == o) so the
    reading reflects that option alone; we average a couple of stochastic
    rollouts for stability. Returned as grid[row][col] where row = probability
    index (0 = p0.1 .. 4 = p0.9) and col = reward-magnitude index.
    """
    grid = [[0.0] * 5 for _ in range(5)]
    for o in range(25):
        vals = [run_one_trial(pg, o, o, context=context)['value'] for _ in range(repeats)]
        grid[o // 5][o % 5] = sum(vals) / len(vals)
    return grid


# ======================================================================
# Training jobs: launch scripts/training/train.py as a subprocess and stream
# its real progress. Keeps pyrl/ untouched; Stop = terminate the process.
# ======================================================================
TRAIN_JOBS = {}            # job_id -> dict
TRAIN_LOCK = threading.Lock()

# Flags that reproduce the opal04 (D1/D2 OpAL) architecture.
OPAL_TRAIN_FLAGS = [
    '--opponent-modulation', '--positive-policy-readout',
    '--pathway-specific-plasticity',
    '--opal-d1-negative-scale', '0.4', '--opal-d2-positive-scale', '0.4',
]

_RE_AFTER = re.compile(r'After (\d+) updates')
_RE_MEANR = re.compile(r'Mean reward:\s*([-\d.eE]+)')
_RE_PCORR = re.compile(r'P\(correct\|choice\):.*=\s*([\d.]+)')


def _train_reader(job):
    """Read the training subprocess stdout, parse per-checkpoint progress, and
    push {iter, meanReward, pCorrect} records onto the job queue."""
    proc, q = job['proc'], job['queue']
    cur = None
    for raw in proc.stdout:
        line = raw.decode('utf-8', 'replace').rstrip()
        m = _RE_AFTER.search(line)
        if m:
            if cur is not None:
                q.put(cur)
            cur = {'iter': int(m.group(1)), 'meanReward': None, 'pCorrect': None}
            continue
        if cur is not None:
            m = _RE_MEANR.search(line)
            if m:
                cur['meanReward'] = float(m.group(1))
                continue
            m = _RE_PCORR.search(line)
            if m:
                cur['pCorrect'] = float(m.group(1))
    if cur is not None:
        q.put(cur)
    job['done'] = True
    q.put({'__done__': True, 'savefile': job['savefile'], 'code': proc.poll()})


def start_training(seed=1, max_iter=400, checkfreq=10, n_validation=100):
    """Launch a fresh opal04-style training run to a temp data root."""
    job_id = uuid.uuid4().hex[:8]
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    group = f'data_uitrain/{stamp}_{job_id}'
    suffix = '_uitrain'
    name = 'gambling' + suffix
    savefile_rel = f'{group}/weights/{name}/{name}.pkl'
    cmd = [
        sys.executable, os.path.join('scripts', 'training', 'train.py'),
        'tasks/gambling.py', '--data-root', group, '--suffix', suffix,
        '--device', 'cpu', '--seed', str(seed),
        '--max-iter', str(max_iter), '--checkfreq', str(checkfreq),
        '--n-validation', str(n_validation),
    ] + OPAL_TRAIN_FLAGS + ['train']
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env)
    job = {
        'id': job_id, 'proc': proc, 'queue': queue.Queue(),
        'savefile': savefile_rel, 'done': False, 'max_iter': max_iter,
    }
    with TRAIN_LOCK:
        TRAIN_JOBS[job_id] = job
    threading.Thread(target=_train_reader, args=(job,), daemon=True).start()
    return job


class Handler(SimpleHTTPRequestHandler):
    """Serve the static 3D app, plus a small JSON API under /api/."""

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode('utf-8')) if raw else {}

    def do_GET(self):
        route = self.path.split('?', 1)[0]
        if route == '/api/models':
            try:
                self._send_json({
                    'models': find_model_checkpoints(),
                    'default': DEFAULT_MODEL,
                })
            except Exception as e:  # surface errors as JSON, not a stack-dump page
                self._send_json({'error': str(e)}, status=500)
            return
        if route == '/api/active':
            with ACTIVE_LOCK:
                self._send_json({'meta': ACTIVE['meta']})
            return
        if route == '/api/train/stream':
            self._stream_training()
            return
        if route == '/api/value_grid':
            try:
                qs = parse_qs(urlparse(self.path).query)
                context = float(qs.get('context', ['0'])[0])
                with ACTIVE_LOCK:
                    pg = ACTIVE['pg']
                    if pg is None:
                        raise ValueError('no model loaded; POST /api/load first')
                    grid = compute_value_grid(pg, context=context)
                self._send_json({'grid': grid, 'context': context})
            except Exception as e:
                self._send_json({'error': str(e)}, status=400)
            return
        # Fall through to static file serving (handled via directory= below).
        super().do_GET()

    def _stream_training(self):
        """Server-sent-events stream of a training job's live progress."""
        qs = parse_qs(urlparse(self.path).query)
        job = TRAIN_JOBS.get(qs.get('job', [''])[0])
        if not job:
            self._send_json({'error': 'unknown job'}, status=404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Connection', 'close')
        self.end_headers()
        q = job['queue']
        try:
            while True:
                try:
                    item = q.get(timeout=1.0)
                except queue.Empty:
                    self.wfile.write(b': ping\n\n')   # heartbeat / disconnect check
                    self.wfile.flush()
                    continue
                if isinstance(item, dict) and item.get('__done__'):
                    self.wfile.write(('event: done\ndata: ' + json.dumps(item) + '\n\n').encode())
                    self.wfile.flush()
                    break
                self.wfile.write(('data: ' + json.dumps(item) + '\n\n').encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        route = self.path.split('?', 1)[0]
        if route == '/api/load':
            try:
                body = self._read_json_body()
                pg, meta = load_model(body.get('path'))
                with ACTIVE_LOCK:
                    ACTIVE.update(pg=pg, model=None, meta=meta, path=meta['path'])
                self._send_json({'meta': meta})
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({'error': str(e)}, status=400)
            return
        if route == '/api/train':
            try:
                body = self._read_json_body()
                job = start_training(
                    seed=int(body.get('seed', 1)),
                    max_iter=int(body.get('maxIter', 400)),
                    checkfreq=int(body.get('checkfreq', 10)),
                    n_validation=int(body.get('nValidation', 100)),
                )
                self._send_json({'job': job['id'], 'savefile': job['savefile'],
                                 'max_iter': job['max_iter']})
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({'error': str(e)}, status=400)
            return
        if route == '/api/train/stop':
            try:
                body = self._read_json_body()
                job = TRAIN_JOBS.get(body.get('job'))
                if job and job['proc'].poll() is None:
                    job['proc'].terminate()
                self._send_json({'stopped': True})
            except Exception as e:
                self._send_json({'error': str(e)}, status=400)
            return
        if route == '/api/trial':
            try:
                body = self._read_json_body()
                with ACTIVE_LOCK:
                    pg = ACTIVE['pg']
                    if pg is None:
                        raise ValueError('no model loaded; POST /api/load first')
                    out = run_one_trial(
                        pg, body.get('target_l', 0), body.get('target_r', 0),
                        context=float(body.get('context', 0.0)))
                self._send_json(out)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({'error': str(e)}, status=400)
            return
        self._send_json({'error': 'not found'}, status=404)

    def end_headers(self):
        # Dev server: never let the browser cache the app files, so edits to
        # the JS/HTML/CSS always show up on reload.
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        super().end_headers()

    def log_message(self, fmt, *args):  # quieter console
        sys.stderr.write("[serve] " + (fmt % args) + "\n")


def main():
    if not os.path.isdir(VISUAL_SIM_DIR):
        print(f"visual_sim dir not found: {VISUAL_SIM_DIR}")
        print("Set VISUAL_SIM_DIR to point at it.")
        sys.exit(1)

    handler = partial(Handler, directory=VISUAL_SIM_DIR)
    server = ThreadingHTTPServer(('127.0.0.1', PORT), handler)
    print(f"Serving 3D app from {VISUAL_SIM_DIR}")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Open http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == '__main__':
    main()
