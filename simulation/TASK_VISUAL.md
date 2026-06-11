# Task: Build a Polished 3D Monkey Gambling-Task Simulation (Visual Prototype)

## Objective

Build a **standalone, great-looking 3D web simulation** of a primate
decision-making experiment. A monkey sits in a lab chair facing a screen that
shows two coloured "gambling" options; on each trial the monkey looks at the
options and reaches toward one, and a reward is revealed. The user can switch
the monkey's species, edit which options appear, train a (mock) agent and watch
a live "learning" curve, and drag a dopamine slider that biases the agent toward
risky or safe choices.

This is a **visual prototype**. The agent's choices are produced by a
**self-contained mock model written in JavaScript** — a placeholder with a clean
interface that will later be swapped for a real trained model. Do **not** read,
import, or depend on any other code in this repository. Everything you need is
described below. Spend your effort on making the 3D scene, animation, and UI
look genuinely polished.

Deliver something the user can open in a browser and immediately find beautiful
and fun to interact with.

---

## Tech stack

- **Three.js (r0.160+)** loaded as ES modules via a CDN import map
  (`three` and `three/addons/`), with `OrbitControls` for the camera.
- Plain **HTML/CSS/JS** — no build step, no framework, no bundler. The app must
  run by opening `index.html` directly (or via any trivial static server).
- 2D charts drawn on plain `<canvas>` — no chart library.
- All 3D content is **procedural geometry** (Three.js primitives). No external
  models, textures, or image assets.

Suggested layout:

```
visual_sim/
├── index.html
├── css/style.css
└── js/
    ├── app.js       UI wiring, state, the trial loop, mock "learning" loop
    ├── scene.js     Three.js lab: room, chair, screen, lighting, trial animation
    ├── monkey.js    procedural marmoset / macaque / chimpanzee builders
    ├── charts.js    learning curve, value heatmap, clickable option grid
    └── agent.js     the MOCK model behind a clean, swappable interface
```

---

## The task being visualised (self-contained spec — no code needed)

There are **25 gambling options** arranged in a **5×5 grid**:

- **Rows = win probability**: `0.1, 0.3, 0.5, 0.7, 0.9` (top row riskiest).
- **Columns = reward magnitude**: 5 increasing levels.
- Each option has an **expected value** EV = probability × magnitude. Use a
  layout where EV is roughly constant within a column and grows across columns,
  so that within any column the **low-probability/high-magnitude option is the
  "risky" one** and the high-probability/low-magnitude option is the "safe" one.
  A simple workable scheme: pick 5 base EVs (e.g. 1.0, 1.4, 1.8, 2.2, 2.6) and
  set magnitude = EV / probability for each cell.
- Give each of the 25 cells a distinct **RGB colour** forming a smooth 2D
  gradient across the grid (e.g. hue varying with probability, brightness with
  magnitude), so options are visually discriminable. Define these colours
  yourself; they are the only thing shown on the screen as the "stimuli".

**A trial** has three phases, animated in sequence:

1. **Fixation** — only a central white dot on the screen; the monkey looks at it.
2. **Stimulus** — two coloured discs appear, one on the left, one on the right
   (the two chosen options); the monkey scans between them.
3. **Decision** — the monkey gazes toward and reaches for one disc. The chosen
   disc is highlighted; if the trial is rewarded, a gold flash/coin appears.

Whether a trial is rewarded is drawn randomly against the chosen option's win
probability (magnitude only affects the reward size shown).

---

## The MOCK agent (`agent.js`) — clean interface for later replacement

Implement a placeholder agent with this exact shape so a real model can be
dropped in later without touching the rest of the app:

```js
// js/agent.js
export class MockAgent {
  constructor() { this.trained = false; this.skill = 0; /* 0..1 */ }

  // Returns one synthetic training record. Call repeatedly to simulate learning.
  trainStep(iter) { /* return {iter, meanReward, pCorrect} */ }

  // Decide between option L and R given a dopamine bias in [-1, 1].
  // Must return { choice: 'L'|'R', value: number, rewarded: boolean, reward: number }.
  decide(optionL, optionR, dopamine) { /* ... */ }

  // Predicted "value" for every cell, as a 5x5 array (for the heatmap).
  valueGrid(dopamine) { /* ... */ }
}
```

Reasonable mock behaviour (good enough to look alive and respond to the sliders):

- **Learning curve:** over ~60–120 steps, `meanReward` and `pCorrect` rise along
  a noisy saturating curve (e.g. `1 - exp(-iter/τ)` plus small noise), and
  `skill` ramps 0→1. Before training, the agent chooses ~randomly.
- **Choice rule:** compute each option's EV. As `skill` rises, weight choices
  toward higher EV (softmax on EV with temperature shrinking as skill grows).
- **Dopamine bias:** `dopamine > 0` nudges the agent toward the **risky** (lower
  probability, higher magnitude) option; `dopamine < 0` toward the **safe**
  option. Implement as an additive term on the risky option's score scaled by
  dopamine. This is what makes the slider visibly change behaviour.
- **`value`:** return something plausible (e.g. the chosen option's EV plus
  noise), used for the heatmap and the per-trial readout.

Keep all randomness and logic inside `agent.js`. The rest of the app should only
ever call `trainStep`, `decide`, and `valueGrid`.

---

## 3D scene requirements (spend the most effort here)

A convincing, attractive **primate-lab** scene:

- **Room:** a floor and back wall with tasteful, low-key materials; soft fog for
  depth. A clean dark/clinical palette.
- **Lighting:** a warm key directional light casting **soft shadows**
  (`PCFSoftShadowMap`), a cooler fill light, and gentle ambient. Make it look
  intentionally lit, not flat.
- **Primate chair:** a seat, back, and legs in a metallic/neutral material,
  positioned so the monkey sits in it facing the screen.
- **Stimulus monitor:** a framed screen in front of the monkey showing a faint
  emissive panel, a central fixation dot, and two option discs (left/right) whose
  colours are set from the chosen options. Add a subtle glow/emissive so the
  discs read as "lit". A reward flash plane (gold) that fades in/out on wins.
- **Camera:** `OrbitControls` with sensible limits (can't go under the floor),
  damping on, framed on the monkey + screen. Resizes with the window.

### The monkey (`monkey.js`) — three species, procedural

Each species is a `THREE.Group` of primitives with **named pivots** the scene
can animate:

- `userData.head` — head pivot, for gaze.
- `userData.armL`, `userData.armR` — shoulder pivots, for reaching/pointing.

Build a seated body (torso, belly, folded legs), a head with face patch, muzzle,
eyes (sclera + pupil), and species-appropriate features. Make the three clearly
distinct:

- **Marmoset** — small, light brown fur, prominent **white ear tufts**, long
  curling tail.
- **Macaque** — mid-size, tan fur, **round ears**, pinkish face, medium tail.
- **Chimpanzee** — large, near-black fur, **big ears**, dark face, no tail,
  bulkier build.

Scale each species differently so swapping is obviously visible.

### Trial animation (`scene.js`)

Given a trial (which two options, the chosen side, whether rewarded), animate
over ~1–2 seconds with smooth easing/lerping (no instant jumps):

- **Fixation:** head faces the centre dot; arms at rest; only the fixation dot lit.
- **Stimulus:** two discs appear; the head gently scans left/right.
- **Decision:** the head turns toward the chosen side and the corresponding arm
  reaches/points toward that disc; the chosen disc gets a highlight ring.
- **Outcome:** on a win, the reward plane flashes gold (and/or a coin pops);
  on a loss, a subtle neutral cue. Then everything eases back to rest.
- When idle (no trial running), add a soft "breathing" bob and occasional
  micro-movements so the monkey looks alive.

---

## UI / controls

A clean two-rail overlay over the 3D canvas (frosted-glass panels, rounded
corners, soft shadows, a tasteful dark theme with one accent colour). Left rail
= controls, right rail = analytics, plus a small bottom HUD.

**Left rail — controls:**
- **Subject:** segmented buttons — Marmoset / Macaque / Chimpanzee. Switching
  rebuilds the monkey instantly.
- **Baseline dopamine** slider [-1 … +1] with a "risk-averse ↔ risk-seeking"
  label and a numeric readout. (This is the *training-time* bias for the mock.)
- **Train agent** / **Stop** buttons. Training kicks off the mock learning loop
  that drives the live curve.
- **Environment editor:** a clickable **5×5 option grid** (rendered in the real
  option colours). First click sets the LEFT option, next sets RIGHT, then
  repeats. Show each option's probability, magnitude, EV, and which side is
  "risky".
- **Live dopamine** slider [-1 … +1] — biases the *trained* agent's choices in
  real time (separate from the baseline/training slider).
- **Run trial** / **Auto-run** buttons.

**Right rail — analytics:**
- **Learning curve:** animated line chart of mean reward + P(correct) vs
  iteration, updating live during training (two colours, small legend, axes).
- **Predicted value (5×5):** a heatmap (viridis-style colormap) of the agent's
  `valueGrid`, with row labels = probabilities and a "reward →" axis hint.
- **This trial:** a readout of the choice (LEFT/RIGHT), a safe/risky chip, the
  reward, and the two EVs.

**Bottom HUD:** trial number · current phase · last choice + reward, in a
rounded pill.

---

## Interaction flow

1. Page loads → 3D lab visible, monkey idling, option grid populated, charts
   showing empty/placeholder states.
2. User picks a species, sets baseline dopamine, clicks **Train agent** → the
   learning curve animates upward over a few seconds; when it finishes, trial
   controls enable and the value heatmap fills in.
3. User edits the LEFT/RIGHT options on the grid, sets live dopamine, clicks
   **Run trial** (or Auto-run) → the monkey performs the trial in 3D, the HUD and
   "this trial" readout update, and moving the live-dopamine slider visibly
   shifts how often the monkey picks the risky option.

---

## Polish bar (this is the point of the task)

- Smooth, eased animations everywhere — no popping or teleporting.
- Cohesive colour theme; legible typography; consistent spacing and rounded
  panels with subtle blur and shadow.
- Soft shadows and considered lighting in the 3D scene.
- Responsive to window resize; no layout breakage.
- Runs at a smooth frame rate; clean up / lerp toward targets each frame rather
  than setting state abruptly.
- No console errors. Degrade gracefully if WebGL is unavailable (show a message).

---

## Acceptance criteria

1. Opening `index.html` shows the lit 3D lab with a seated monkey and a screen.
2. All three species render and swap correctly and look distinct.
3. **Train agent** produces a live, animated learning curve and then fills the
   value heatmap.
4. The option grid is clickable and updates the discs shown on the 3D screen.
5. Running a trial animates fixation → stimulus → decision → outcome, with the
   monkey gazing and reaching toward its choice and a reward flash on wins.
6. Both dopamine sliders work; the live one visibly changes risky-vs-safe choice
   rates.
7. `agent.js` is the **only** place choice/learning logic lives, behind the
   `trainStep` / `decide` / `valueGrid` interface, ready to be replaced by a real
   model later.

> Note: the real trained model is **not** part of this task. Build the mock
> agent behind the interface above; the user will plug in the real model
> afterward.
