# Task: Build a 3D Monkey Simulation of the Risk-Preference Gambling Task

## Role & objective

You are building, from scratch, an interactive **3D web simulation** of the
risk-sensitive reinforcement-learning gambling task in this repository. A monkey
agent — **driven by the project's real `pyrl` actor-critic RNN, not a
re-implementation** — sits in a primate chair, watches two coloured gambling
options on a screen, and chooses between them. The user can train the agent
live and watch it learn, edit the environment, switch primate species, and
sweep dopamine to make the agent risk-averse or risk-seeking.

Deliver a working app the user can run with `python simulation/server.py` and
open at `http://127.0.0.1:5000`.

---

## Required reading (do this first)

Before writing any code, read these files to understand the real system you are
wrapping. Do **not** guess at their contents — they define correctness.

- `tasks/gambling.py` — the task: 7 inputs (FIXATION + RGB×2), 3 actions
  (FIXATE / CHOOSE-LEFT / CHOOSE-RIGHT), 25 options in a 5×5 grid
  (`value_vector` = [probability, reward_size]), `color_vector` RGB encodings,
  epoch durations (fixation/stimulus/decision), and `get_condition` /
  `get_step`. Note `get_condition(rng, dt, context={})` accepts
  `context={'target_l': i, 'target_r': j}` to force specific options.
- `pyrl/model.py` — `Model(modelfile)` loads a task spec; `model.get_pg(...)`
  returns an `ActorCriticTrainer`. Read `Model.train` and `Model.get_pg`.
- `pyrl/configs.py` — every config key and default. The risk knob is `kappa`
  (clipped to [-0.9, 0.9]); inference-time dopamine is `opto_stim_offset` /
  `recent_rpe_stim_offset`. D1/D2 and phasic-DA flags live here too.
- `pyrl/rollouts.py` — `run_trials(trials, return_states=True)` returns a dict
  with `A` (actions, one-hot, shape (T, n, Nout)), `Z_b` (value estimate,
  (T, n)), `R` (reward, (T, n)), `perf`. This is how you drive the monkey.
- `pyrl/learning.py` — `_train` prints validation lines you will parse for the
  live learning curve: `After N updates`, `Mean reward:`, `P(choice):`,
  `P(correct|choice):`, `Prediction error:`, `Best reward: X (iteration Y)`.
- `scripts/training/train.py` — the CLI entry point. Note the **positional
  `action` argument comes immediately after the model file**
  (`train.py <task> train --flags...`), the `--data-root` / `--suffix` /
  `--seed` / `--device` / `--kappa` options, and the full set of model flags.

### Critical constraints discovered in the codebase

1. **Never train in-process.** `pyrl/learning.py::_train` calls `sys.exit(0)`
   when it reaches `max_iter`, which would kill your server. Launch training as
   a **subprocess** of `scripts/training/train.py` and parse its stdout.
2. **Run inference in-process.** Loading a checkpoint and calling `run_trials`
   is safe and fast; do that directly in the Flask process.
3. **`train.py` argument order matters.** `action` ('train') must come right
   after the task-spec path, before the optional flags, or argparse rejects it.
4. **Beware the repo-root `.gitignore`.** It contains a broad `*html` rule that
   will silently exclude your `index.html` from `git add`. Force-add it
   (`git add -f`) and verify it is tracked before committing.
5. There may be **no trained `.pkl` weights** present. The app must train the
   network live; design around that rather than assuming pre-trained models.

---

## Functional requirements

### A. Subject (species)
Three selectable primates as **procedural 3D models** (no external assets):
**marmoset**, **macaque**, **chimpanzee**. They must differ visibly in size,
fur colour, face skin, and ear/tuft/tail morphology (e.g. marmoset = small with
white ear tufts; macaque = mid-size with round ears; chimpanzee = large, dark,
big ears, no tail). The model is seated in a chair facing the screen.

### B. Models (the three headline conditions)
Expose exactly these three presets, mapped to **real `train.py` flag sets**:

| Preset           | Meaning                                  | Flags |
|------------------|------------------------------------------|-------|
| `basic`          | plain actor-critic                       | (none) |
| `d1d2_04`        | OpAL-style D1/D2 opponent plasticity     | `--opponent-modulation --positive-policy-readout --pathway-specific-plasticity --opal-alpha-d1 0.4 --opal-alpha-d2 0.4` |
| `d1d2_04_phasic` | D1/D2 **+ between-trial phasic dopamine**| the `d1d2_04` flags **plus** `--use-recent-rpe-modulation --recent-rpe-gain 0.5 --recent-rpe-decay 0.6` |

Verify these flag names against `scripts/training/train.py`'s argument parser —
adjust if the parser differs.

### C. Baseline dopamine → risk preference
A slider in **[-1, +1]**. Map it to `kappa` (× 0.9, clamped to [-0.9, 0.9]) and
pass it as `--kappa` at training time. Lower (negative) = risk-averse; higher
(positive) = risk-seeking. Show the resulting kappa value in the UI.

### D. Live (inference-time) dopamine
A **separate** slider in [-1, +1] that shifts risk preference on an
already-trained model **without retraining**, by setting `opto_stim_offset`
(and `recent_rpe_stim_offset` for the D1/D2 presets) on the loaded trainer
before `run_trials`.

### E. Watch it learn
- Launch training via subprocess; stream parsed validation records to the
  browser using **Server-Sent Events** (`text/event-stream`).
- Draw a **live learning curve** (mean reward + P(correct) vs iteration) that
  updates each checkpoint.
- Provide a **5×5 predicted-value heatmap**: sample many trials through the
  trained net, bin the value estimate at the choice timestep by the chosen
  option's grid cell (row = win probability, col = reward magnitude).

### F. Manual environment editing
A clickable **5×5 option grid** rendered with the real `color_vector`. First
click sets the LEFT target, next sets RIGHT, alternating. Selected cells feed
`target_l` / `target_r` into `get_condition`. Show p / magnitude / EV and which
side is the "risky" (lower-probability) option.

### G. Trial playback
"Run trial" and "Auto-run". For each trial: call the backend, get a per-timestep
timeline (phase, action, value) plus choice/reward, and animate the monkey —
fixate during fixation, scan during stimulus, gaze + reach toward the chosen
side during decision, highlight the chosen option, flash reward on win. A small
HUD shows trial number / phase / choice+reward.

---

## Architecture

```
simulation/
├── server.py          Flask: subprocess training + SSE stream; in-process run_trials inference
├── presets.py         PRESETS table; dopamine→kappa and dopamine→opto-offset mappings; train argv builder
├── sim_task.py        `from tasks.gambling import *` then shrink N / max_iter / n_validation for a watchable live run
├── requirements.txt   flask  (torch/numpy/scipy/matplotlib come from repo root)
├── .gitignore         workspace/  __pycache__/
├── README.md
└── static/
    ├── index.html     (force-add past the *html ignore rule)
    ├── css/style.css
    └── js/
        ├── app.js     UI wiring, fetch/SSE, trial loop
        ├── scene.js   Three.js lab: room, chair, screen, trial animation
        ├── monkey.js  procedural marmoset / macaque / chimpanzee builders
        └── charts.js  learning curve + value heatmap + clickable option grid (plain canvas 2D)
```

**Frontend stack:** Three.js (r0.160) via an ES-module import map from a CDN
(`three` + `three/addons/`), `OrbitControls` for the camera. Charts are plain
2D `<canvas>` — no chart library.

### Backend HTTP API (suggested)
- `GET  /` → `index.html`; `GET /static/<path>` → assets.
- `GET  /api/task` → task constants (inputs, actions, `value_vector`,
  `color_vector`, durations, probabilities) + preset labels/descriptions +
  `has_checkpoint`.
- `POST /api/train` `{preset, dopamine, seed}` → starts a run (reject if one is
  active).
- `POST /api/train/stop` → terminate the subprocess.
- `GET  /api/train/stream` → SSE; emit `{type:'state', state}` and
  `{type:'progress', iter, mean_reward, p_choice, p_correct, ...}`. Replay
  current state + history to late subscribers; send keep-alives.
- `POST /api/trial` `{target_l, target_r, inference_dopamine, preset}` → run one
  trial through the real net, return timeline + choice + reward + EVs + colours.
- `POST /api/heatmap` `{inference_dopamine, preset}` → 5×5 value grid.
- `GET  /api/status` → state + whether a checkpoint exists.

**Training manager:** one run at a time, guarded by a lock; a reader thread
parses subprocess stdout with regexes and broadcasts to a list of subscriber
queues. On process exit, set state `done`/`error`.

**Inference:** locate the checkpoint at
`workspace/weights/<modelname>/<modelname>.pkl`; cache the loaded `pg` keyed by
(checkpoint mtime, rounded inference-dopamine, preset) so repeated trials are
fast and re-load when retrained.

### `sim_task.py`
Re-export the real task and only reduce capacity/budget for responsiveness,
e.g. `N = 50`, `baseline_N = 50`, `max_iter = 600`, `n_validation = 100`,
`checkfreq = 10`. Document that bumping toward `configs.py` values yields more
converged behaviour at the cost of a longer (still streamed) run.

---

## Acceptance criteria — verify each before declaring done

1. `python -m py_compile` passes for all backend `.py`; `node --check` passes
   for each JS file (skip gracefully if node is absent).
2. Server boots; `GET /` returns **200** (confirm `index.html` is actually
   committed/tracked, not just on disk), and `/api/task` returns the 25 options
   and 3 presets.
3. `POST /api/train` followed by `GET /api/train/stream` yields parsed
   `progress` events with increasing `iter`, and a checkpoint `.pkl` is written.
4. After training, `POST /api/trial` returns a full per-timestep timeline with
   a choice and reward; `POST /api/heatmap` returns a populated 5×5 grid.
5. The frontend loads the 3D scene, all three species render and swap, both
   dopamine sliders work, the option grid is clickable, the learning curve
   animates during training, and trials animate the monkey's choice.

> You cannot visually confirm the 3D render without a browser + CDN access.
> Validate everything testable headlessly (compile checks, API round-trips,
> checkpoint creation, trial/heatmap payloads), and explicitly flag the visual
> render as the one item the user should confirm in a browser.

---

## Deliverable & git workflow

- Put everything under `simulation/`. Keep the trained-model `workspace/`
  out of git.
- Commit the dependency-light backend and the static frontend. **Double-check
  `git ls-files simulation/static` includes `index.html`** — the `*html`
  ignore rule will otherwise drop it.
- Do not open a pull request unless explicitly asked.
- Write a `simulation/README.md` covering: what it does, how to run it, and a
  table mapping each UI control to its real `pyrl` mechanism.
