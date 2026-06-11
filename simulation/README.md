# 3D Monkey Simulation of the Risk-Preference Gambling Task

An interactive 3D lab where a monkey agent — **driven by the project's real
`pyrl` actor-critic RNN** — sits in a primate chair, watches the two coloured
gambling options on a screen, and chooses between them. You can train the agent
live and watch the learning process, edit the environment, switch species, and
sweep dopamine to make the agent risk-averse or risk-seeking.

> The agent is the actual recurrent network from `pyrl`, not a re-implementation.
> Training runs as a subprocess of `scripts/training/train.py` and its progress
> is streamed to the browser; trial choices are produced by loading the trained
> checkpoint and calling `run_trials` on the real network.

## What you can do

- **Pick a subject**: marmoset, macaque, or chimpanzee (procedural 3D models with
  species-appropriate size, colouring, ears/tufts, and tail).
- **Pick a model** — the three headline conditions:
  - `basic` — plain actor-critic.
  - `d1d2_04` — OpAL-style D1/D2 opponent plasticity (`opal_alpha_d1 = opal_alpha_d2 = 0.4`).
  - `d1d2_04_phasic` — the above plus between-trial phasic dopamine
    (leaky previous-trial RPE biasing the next trial).
- **Set baseline dopamine** (−1 … +1): baked in as the learning-asymmetry `kappa`
  at training time. Lower = risk-averse, higher = risk-seeking.
- **Train and watch it learn**: a live learning curve (mean reward and
  P(correct)) updates each validation checkpoint, and a 5×5 value heatmap shows
  the agent's predicted value over the option grid.
- **Edit the environment**: click cells in the 5×5 option grid to set the LEFT
  and RIGHT targets for the next trial (probability × reward magnitude).
- **Nudge dopamine live**: a separate inference-time dopamine slider applies an
  optogenetic-style offset so you can shift risk preference on an already-trained
  model without retraining.
- **Run trials / auto-run**: watch the monkey fixate, view the stimuli, and reach
  toward its choice, with the reward outcome flashed on screen.

## Running it

From the repository root:

```bash
pip install -r requirements.txt          # project deps: torch, numpy, scipy, matplotlib
pip install -r simulation/requirements.txt   # flask
python simulation/server.py
```

Then open <http://127.0.0.1:5000>. Choose a model and baseline dopamine, click
**Train agent**, watch the learning curve, then **Run trial**.

## How the pieces map to `pyrl`

| Simulation control        | Real `pyrl` mechanism                                              |
|---------------------------|-------------------------------------------------------------------|
| Model preset              | `train.py` flag sets (see `presets.py`)                           |
| Baseline dopamine         | `--kappa` (learning asymmetry, clipped to [-0.9, 0.9])           |
| Live (inference) dopamine | `opto_stim_offset` / `recent_rpe_stim_offset` on the trained model|
| Learning curve            | parsed from the trainer's validation output over SSE             |
| Trial choice / value      | `ActorCriticTrainer.run_trials(..., return_states=True)`         |
| Environment editor        | `target_l` / `target_r` passed into `gambling.get_condition`     |

The task itself (epochs, 25 options, RGB encodings, probabilistic reward) comes
straight from `tasks/gambling.py`; `sim_task.py` only shrinks the network and
iteration budget so training is watchable in real time.

## Files

```
simulation/
├── server.py          Flask backend: training (subprocess + SSE) and inference
├── presets.py         model presets + dopamine→config mapping
├── sim_task.py        the real gambling task with a smaller, watchable budget
├── requirements.txt   flask
└── static/
    ├── index.html
    ├── css/style.css
    └── js/
        ├── app.js      UI wiring, API/SSE, trial loop
        ├── scene.js    Three.js lab (room, chair, screen, trial animation)
        ├── monkey.js   procedural marmoset / macaque / chimpanzee
        └── charts.js   learning curve, value heatmap, clickable option grid
```
