"""Backend for the 3D gambling-task simulation.

Runs the *real* pyrl actor-critic RNN:
  * Training is launched as a subprocess (``scripts/training/train.py``) so the
    trainer's ``sys.exit`` at completion cannot kill the server; stdout is parsed
    and streamed to the browser as Server-Sent Events (the live learning curve).
  * Trial inference loads the latest checkpoint in-process and calls
    ``run_trials`` to drive the monkey's choices, with an inference-time dopamine
    offset applied so risk preference can be nudged without retraining.

Run:  python simulation/server.py   (then open http://127.0.0.1:5000)
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

from pyrl import utils
from pyrl.model import Model
import presets as preset_mod

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
TASK_SPEC = os.path.join(os.path.dirname(__file__), 'sim_task.py')
TRAIN_SCRIPT = os.path.join(REPO_ROOT, 'scripts', 'training', 'train.py')
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'workspace')
SUFFIX = '_sim'
MODEL_NAME = 'sim_task' + SUFFIX

app = Flask(__name__, static_folder=None)


# ---------------------------------------------------------------------------
# Task constants (mirror tasks/gambling.py) exposed to the frontend.
# ---------------------------------------------------------------------------
import tasks.gambling as gambling


def task_constants():
    return {
        'inputs': list(gambling.inputs.keys()),
        'actions': list(gambling.actions.keys()),
        'value_vector': gambling.value_vector.tolist(),
        'color_vector': gambling.color_vector.tolist(),
        'reward_scale': gambling.REWARD_SCALE,
        'durations': {
            'fixation': gambling.fixation,
            'stimulus': gambling.stimulus,
            'decision': gambling.decision,
            'tmax': gambling.tmax,
        },
        'probabilities': sorted(set(gambling.value_vector[:, 0].tolist())),
    }


# ---------------------------------------------------------------------------
# Training manager: one live training run at a time, streamed via SSE.
# ---------------------------------------------------------------------------
class TrainManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.thread = None
        self.subscribers = []          # list[queue.Queue]
        self.history = []              # parsed iteration records
        self.state = 'idle'            # idle | training | done | error
        self.meta = {}

    def _broadcast(self, event):
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def subscribe(self):
        q = queue.Queue(maxsize=1000)
        self.subscribers.append(q)
        # Replay current state so a late subscriber catches up.
        q.put_nowait({'type': 'state', 'state': self.state, 'meta': self.meta})
        for rec in self.history:
            q.put_nowait({'type': 'progress', **rec})
        return q

    def unsubscribe(self, q):
        if q in self.subscribers:
            self.subscribers.remove(q)

    def start(self, preset, dopamine, seed):
        with self.lock:
            if self.state == 'training':
                return False, 'A training run is already in progress.'
            self.history = []
            self.state = 'training'
            self.meta = {'preset': preset, 'dopamine': dopamine, 'seed': seed,
                         'kappa': preset_mod.kappa_for_dopamine(dopamine)}
            argv = preset_mod.train_command(
                sys.executable, TRAIN_SCRIPT, TASK_SPEC, preset, dopamine,
                DATA_ROOT, SUFFIX, seed, device='cpu')
            self.proc = subprocess.Popen(
                argv, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            self.thread = threading.Thread(target=self._read_output, daemon=True)
            self.thread.start()
            self._broadcast({'type': 'state', 'state': 'training', 'meta': self.meta})
            return True, 'started'

    def stop(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
            self.state = 'idle'
            self._broadcast({'type': 'state', 'state': 'idle', 'meta': self.meta})

    def _read_output(self):
        cur = {}
        iter_re = re.compile(r'^After (\d+) updates')
        mean_re = re.compile(r'Mean reward:\s*([-\d.eE+]+)')
        best_re = re.compile(r'Best reward:\s*([-\d.eE+]+)\s*\(iteration (\d+)\)')
        choice_re = re.compile(r'P\(choice\):\s*\d+/\d+\s*=\s*([\d.]+)')
        correct_re = re.compile(r'P\(correct\|choice\):\s*\d+/\d+\s*=\s*([\d.]+)')
        prederr_re = re.compile(r'Prediction error:\s*([-\d.eE+]+)')

        def flush(cur):
            if 'iter' in cur:
                self.history.append(cur.copy())
                self._broadcast({'type': 'progress', **cur})

        for line in self.proc.stdout:
            line = line.rstrip()
            m = iter_re.search(line)
            if m:
                flush(cur)
                cur = {'iter': int(m.group(1))}
                continue
            for rx, key, cast in (
                (mean_re, 'mean_reward', float),
                (choice_re, 'p_choice', float),
                (correct_re, 'p_correct', float),
                (prederr_re, 'pred_error', float),
            ):
                mm = rx.search(line)
                if mm:
                    cur[key] = cast(mm.group(1))
            mb = best_re.search(line)
            if mb:
                cur['best_reward'] = float(mb.group(1))
                cur['best_iter'] = int(mb.group(2))
        flush(cur)
        code = self.proc.wait()
        self.state = 'done' if code in (0, None) else 'error'
        self._broadcast({'type': 'state', 'state': self.state, 'meta': self.meta,
                         'returncode': code})


trainer = TrainManager()


# ---------------------------------------------------------------------------
# Inference: load latest checkpoint and run one trial through the real RNN.
# ---------------------------------------------------------------------------
_model_cache = {}


def checkpoint_path():
    return os.path.join(DATA_ROOT, 'weights', MODEL_NAME, MODEL_NAME + '.pkl')


def load_pg(inference_dopamine, preset):
    path = checkpoint_path()
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    key = (mtime, round(float(inference_dopamine), 3), preset)
    if key in _model_cache:
        return _model_cache[key]

    model = Model(TASK_SPEC)
    pg = model.get_pg(path, seed=1, device='cpu')

    overrides = preset_mod.inference_overrides(preset, inference_dopamine)
    for attr, val in overrides.items():
        setattr(pg, attr, val)

    _model_cache.clear()
    _model_cache[key] = pg
    return pg


def run_one_trial(pg, target_l, target_r):
    """Run a single trial and return per-timestep state for animation."""
    import torch

    trial = pg.task.get_condition(pg.rng, pg.dt,
                                  context={'target_l': int(target_l),
                                           'target_r': int(target_r)})
    results = pg.run_trials([trial], progress_bar=False, return_states=True)

    A = results['A'][:, 0, :].cpu().numpy()      # (T, Nout) one-hot
    Z_b = results['Z_b'][:, 0].cpu().numpy()     # (T,) value estimate
    R = results['R'][:, 0].cpu().numpy()         # (T,) reward
    actions = np.argmax(A, axis=1).tolist()      # per-timestep action index

    action_names = list(gambling.actions.keys())
    epochs = trial['epochs']

    # Find the choice timestep / outcome.
    choice = None
    choice_t = None
    for t, a in enumerate(actions):
        if action_names[a] in ('CHOOSE-LEFT', 'CHOOSE-RIGHT'):
            choice = 'L' if action_names[a] == 'CHOOSE-LEFT' else 'R'
            choice_t = t
            break
    total_reward = float(np.sum(R))

    def phase_at(t):
        if t in epochs['fixation']:
            return 'fixation'
        if t in epochs['stimulus']:
            return 'stimulus'
        return 'decision'

    timeline = [{'t': t, 'phase': phase_at(t), 'action': action_names[a],
                 'value': float(Z_b[t])} for t, a in enumerate(actions)]

    ev_l = float(trial['prob_l'] * trial['size_l'])
    ev_r = float(trial['prob_r'] * trial['size_r'])

    return {
        'target_l': int(target_l), 'target_r': int(target_r),
        'prob_l': float(trial['prob_l']), 'size_l': float(trial['size_l']),
        'prob_r': float(trial['prob_r']), 'size_r': float(trial['size_r']),
        'ev_l': ev_l, 'ev_r': ev_r,
        'color_l': trial['color_l'].tolist(), 'color_r': trial['color_r'].tolist(),
        'choice': choice, 'choice_t': choice_t,
        'reward': total_reward,
        'timeline': timeline,
    }


def value_heatmap(pg, n_samples=200):
    """Estimate the agent's predicted value over the 5x5 option grid."""
    import torch
    grid = np.zeros((5, 5))
    count = np.zeros((5, 5))
    trials = [pg.task.get_condition(pg.rng, pg.dt) for _ in range(n_samples)]
    results = pg.run_trials(trials, progress_bar=False, return_states=True)
    A = np.argmax(results['A'].cpu().numpy(), axis=2)        # (T, n)
    Z_b = results['Z_b'].cpu().numpy()                       # (T, n)
    action_names = list(gambling.actions.keys())
    for i, trial in enumerate(trials):
        col_times = np.where((A[:, i] == 1) | (A[:, i] == 2))[0]
        if not len(col_times):
            continue
        t0 = col_times[0]
        choice = A[t0, i]
        target = trial['target_l'] if choice == 1 else trial['target_r']
        row, col = target // 5, target % 5
        grid[row, col] += Z_b[t0, i]
        count[row, col] += 1
    out = np.where(count > 0, grid / np.maximum(count, 1), np.nan)
    return [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in out]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/static/<path:fname>')
def static_files(fname):
    return send_from_directory(STATIC_DIR, fname)


@app.route('/api/task')
def api_task():
    payload = task_constants()
    payload['presets'] = {k: {'label': v['label'], 'description': v['description']}
                          for k, v in preset_mod.PRESETS.items()}
    payload['has_checkpoint'] = os.path.exists(checkpoint_path())
    return jsonify(payload)


@app.route('/api/train', methods=['POST'])
def api_train():
    data = request.get_json(force=True)
    preset = data.get('preset', 'basic')
    dopamine = float(data.get('dopamine', 0.0))
    seed = int(data.get('seed', 1))
    ok, msg = trainer.start(preset, dopamine, seed)
    return jsonify({'ok': ok, 'message': msg, 'state': trainer.state})


@app.route('/api/train/stop', methods=['POST'])
def api_train_stop():
    trainer.stop()
    return jsonify({'ok': True, 'state': trainer.state})


@app.route('/api/train/stream')
def api_train_stream():
    def gen():
        q = trainer.subscribe()
        try:
            while True:
                try:
                    event = q.get(timeout=15)
                    yield f'data: {json.dumps(event)}\n\n'
                except queue.Empty:
                    yield ': keep-alive\n\n'
        finally:
            trainer.unsubscribe(q)
    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/trial', methods=['POST'])
def api_trial():
    data = request.get_json(force=True)
    target_l = int(data.get('target_l', 0))
    target_r = int(data.get('target_r', 24))
    inference_dopamine = float(data.get('inference_dopamine', 0.0))
    preset = data.get('preset', 'basic')

    pg = load_pg(inference_dopamine, preset)
    if pg is None:
        return jsonify({'ok': False, 'message': 'No trained model yet. Train one first.'}), 400
    result = run_one_trial(pg, target_l, target_r)
    return jsonify({'ok': True, 'trial': result})


@app.route('/api/heatmap', methods=['POST'])
def api_heatmap():
    data = request.get_json(force=True)
    inference_dopamine = float(data.get('inference_dopamine', 0.0))
    preset = data.get('preset', 'basic')
    pg = load_pg(inference_dopamine, preset)
    if pg is None:
        return jsonify({'ok': False, 'message': 'No trained model yet.'}), 400
    return jsonify({'ok': True, 'grid': value_heatmap(pg)})


@app.route('/api/status')
def api_status():
    return jsonify({'state': trainer.state, 'meta': trainer.meta,
                    'has_checkpoint': os.path.exists(checkpoint_path()),
                    'history_len': len(trainer.history)})


if __name__ == '__main__':
    os.makedirs(DATA_ROOT, exist_ok=True)
    print('Gambling-task 3D simulation server')
    print('Open http://127.0.0.1:5000 in your browser.')
    app.run(host='127.0.0.1', port=5000, threaded=True, debug=False)
