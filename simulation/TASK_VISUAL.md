# Task: Build a Polished 3D Visualization of a Decision-Making Experiment (Visual Prototype)

## Objective

Build a **standalone, great-looking 3D web visualization** of a classic
reinforcement-learning decision task. An animated character sits at a desk
facing a display that shows two coloured option cards; on each trial the
character looks at the options and selects one, and the outcome is revealed.
The user can switch the character's appearance, edit which options appear,
"train" a placeholder model and watch a live learning curve, and drag a slider
that biases the model toward higher-variance or lower-variance choices.

This is a **front-end visual prototype** for an educational/research dashboard.
The model's choices are produced by a **self-contained placeholder written in
JavaScript** — a simple stub with a clean interface that will later be replaced
by a real trained model. Do **not** read, import, or depend on any other code in
this repository. Everything needed is described below. Focus your effort on
making the 3D scene, animation, and UI look genuinely polished.

Deliver something the user can open in a browser and immediately find clean,
clear, and pleasant to interact with.

---

## Tech stack

- **Three.js (r0.160+)** loaded as ES modules via a CDN import map
  (`three` and `three/addons/`), with `OrbitControls` for the camera.
- Plain **HTML/CSS/JS** — no build step, no framework, no bundler. The app must
  run by opening `index.html` directly (or via any trivial static file server).
- 2D charts drawn on plain `<canvas>` — no chart library.
- All 3D content is **procedural geometry** (Three.js primitives). No external
  models, textures, or image assets.

Suggested layout:

```
visual_sim/
├── index.html
├── css/style.css
└── js/
    ├── app.js       UI wiring, state, the trial loop, placeholder "learning" loop
    ├── scene.js     Three.js scene: room, desk, display, lighting, trial animation
    ├── character.js procedural character builders (three style variants)
    ├── charts.js    learning curve, value heatmap, clickable option grid
    └── agent.js     the PLACEHOLDER model behind a clean, swappable interface
```

---

## The task being visualised (self-contained spec — no external code needed)

There are **25 options** arranged in a **5×5 grid**:

- **Rows = success probability**: `0.1, 0.3, 0.5, 0.7, 0.9` (top row most uncertain).
- **Columns = payoff size**: 5 increasing levels.
- Each option has an **expected value** EV = probability × payoff. Use a layout
  where EV is roughly constant within a column and grows across columns, so that
  within any column the **low-probability/high-payoff option is the
  "high-variance" one** and the high-probability/low-payoff option is the
  "low-variance" one. A simple workable scheme: pick 5 base EVs
  (e.g. 1.0, 1.4, 1.8, 2.2, 2.6) and set payoff = EV / probability for each cell.
- Give each of the 25 cells a distinct **RGB colour** forming a smooth 2D
  gradient across the grid (e.g. hue varying with probability, brightness with
  payoff), so options are visually discriminable. Define these colours yourself;
  they are the only thing shown on the display as the "options".

**A trial** has three phases, animated in sequence:

1. **Ready** — only a central white dot on the display; the character looks at it.
2. **Options shown** — two coloured cards appear, one on the left, one on the
   right (the two selected options); the character glances between them.
3. **Selection** — the character looks toward and reaches for one card. The
   chosen card is highlighted; if the trial pays off, a gold flash/coin appears.

Whether a trial pays off is drawn randomly against the chosen option's success
probability (payoff size only affects the amount shown on a win).

---

## The PLACEHOLDER model (`agent.js`) — clean interface for later replacement

Implement a placeholder behind this exact interface so a real model can be
dropped in later without touching the rest of the app:

```js
// js/agent.js
export class PlaceholderAgent {
  constructor() { this.trained = false; this.skill = 0; /* 0..1 */ }

  // Returns one synthetic training record. Call repeatedly to simulate learning.
  trainStep(iter) { /* return {iter, meanReward, pCorrect} */ }

  // Choose between option L and R given a bias parameter in [-1, 1].
  // Must return { choice: 'L'|'R', value: number, rewarded: boolean, reward: number }.
  decide(optionL, optionR, bias) { /* ... */ }

  // Predicted "value" for every cell, as a 5x5 array (for the heatmap).
  valueGrid(bias) { /* ... */ }
}
```

Reasonable placeholder behaviour (enough to look alive and respond to the sliders):

- **Learning curve:** over ~60–120 steps, `meanReward` and `pCorrect` rise along
  a noisy saturating curve (e.g. `1 - exp(-iter/τ)` plus small noise), and
  `skill` ramps 0→1. Before training, choices are ~random.
- **Choice rule:** compute each option's EV. As `skill` rises, weight choices
  toward higher EV (softmax on EV with temperature shrinking as skill grows).
- **Bias parameter:** `bias > 0` nudges the model toward the **high-variance**
  (lower probability, higher payoff) option; `bias < 0` toward the
  **low-variance** option. Implement as an additive term on the high-variance
  option's score scaled by `bias`. This is what makes the slider visibly change
  behaviour.
- **`value`:** return something plausible (e.g. the chosen option's EV plus
  noise), used for the heatmap and the per-trial readout.

Keep all randomness and logic inside `agent.js`. The rest of the app should only
ever call `trainStep`, `decide`, and `valueGrid`.

---

## 3D scene requirements (spend the most effort here)

A clean, attractive **research-lab** scene:

- **Room:** a floor and back wall with tasteful, low-key materials; soft fog for
  depth. A clean, modern palette.
- **Lighting:** a warm key directional light casting **soft shadows**
  (`PCFSoftShadowMap`), a cooler fill light, and gentle ambient. Make it look
  intentionally lit, not flat.
- **Desk + chair:** a desk and seat in a metallic/neutral material, positioned so
  the character sits facing the display.
- **Display:** a framed screen in front of the character showing a faint emissive
  panel, a central "ready" dot, and two option cards (left/right) whose colours
  are set from the selected options. Add a subtle glow/emissive so the cards read
  as "lit". A payoff flash plane (gold) that fades in/out on wins.
- **Camera:** `OrbitControls` with sensible limits (can't go under the floor),
  damping on, framed on the character + display. Resizes with the window.

### The character (`character.js`) — three style variants, procedural

Each variant is a `THREE.Group` of primitives with **named pivots** the scene
can animate:

- `userData.head` — head pivot, for gaze direction.
- `userData.armL`, `userData.armR` — shoulder pivots, for reaching/pointing.

Build a seated body (torso, midsection, folded legs), a head with a face area,
eyes, and distinguishing features. Make the three variants clearly distinct in
**size, colour, and silhouette** so swapping is obviously visible — for example a
small light-toned variant, a mid-size tan variant, and a large dark bulky
variant. Choose whatever stylised forms you like; the only requirement is that
they look polished and are easy to tell apart.

### Trial animation (`scene.js`)

Given a trial (which two options, the chosen side, whether it paid off), animate
over ~1–2 seconds with smooth easing/lerping (no instant jumps):

- **Ready:** head faces the centre dot; arms at rest; only the ready dot lit.
- **Options shown:** two cards appear; the head gently glances left/right.
- **Selection:** the head turns toward the chosen side and the corresponding arm
  reaches/points toward that card; the chosen card gets a highlight ring.
- **Outcome:** on a win, the payoff plane flashes gold (and/or a coin pops); on a
  loss, a subtle neutral cue. Then everything eases back to rest.
- When idle (no trial running), add a soft "breathing" bob and occasional
  micro-movements so the character looks alive.

---

## UI / controls

A clean two-rail overlay over the 3D canvas (frosted-glass panels, rounded
corners, soft shadows, a tasteful theme with one accent colour). Left rail =
controls, right rail = analytics, plus a small bottom status bar.

**Left rail — controls:**
- **Character:** segmented buttons for the three style variants. Switching
  rebuilds the character instantly.
- **Baseline bias** slider [-1 … +1] with a "low-variance ↔ high-variance" label
  and a numeric readout. (This is the *training-time* bias for the placeholder.)
- **Train model** / **Stop** buttons. Training kicks off the placeholder learning
  loop that drives the live curve.
- **Environment editor:** a clickable **5×5 option grid** (rendered in the real
  option colours). First click sets the LEFT option, next sets RIGHT, then
  repeats. Show each option's probability, payoff, EV, and which side is the
  "high-variance" one.
- **Live bias** slider [-1 … +1] — biases the *trained* model's choices in real
  time (separate from the baseline/training slider).
- **Run trial** / **Auto-run** buttons.

**Right rail — analytics:**
- **Learning curve:** animated line chart of mean reward + accuracy vs iteration,
  updating live during training (two colours, small legend, axes).
- **Predicted value (5×5):** a heatmap (viridis-style colormap) of the model's
  `valueGrid`, with row labels = probabilities and a "payoff →" axis hint.
- **This trial:** a readout of the choice (LEFT/RIGHT), a low/high-variance chip,
  the payoff, and the two EVs.

**Bottom status bar:** trial number · current phase · last choice + payoff, in a
rounded pill.

---

## Interaction flow

1. Page loads → 3D scene visible, character idling, option grid populated, charts
   showing empty/placeholder states.
2. User picks a character variant, sets the baseline bias, clicks **Train model**
   → the learning curve animates upward over a few seconds; when it finishes,
   trial controls enable and the value heatmap fills in.
3. User edits the LEFT/RIGHT options on the grid, sets the live bias, clicks **Run
   trial** (or Auto-run) → the character performs the trial in 3D, the status bar
   and "this trial" readout update, and moving the live-bias slider visibly shifts
   how often the character picks the high-variance option.

---

## Polish bar (this is the point of the task)

- Smooth, eased animations everywhere — no popping or teleporting.
- Cohesive colour theme; legible typography; consistent spacing and rounded
  panels with subtle blur and shadow.
- Soft shadows and considered lighting in the 3D scene.
- Responsive to window resize; no layout breakage.
- Runs at a smooth frame rate; lerp toward targets each frame rather than setting
  state abruptly.
- No console errors. Degrade gracefully if WebGL is unavailable (show a message).

---

## Acceptance criteria

1. Opening `index.html` shows the lit 3D scene with a seated character and a
   display.
2. All three character variants render and swap correctly and look distinct.
3. **Train model** produces a live, animated learning curve and then fills the
   value heatmap.
4. The option grid is clickable and updates the cards shown on the 3D display.
5. Running a trial animates ready → options → selection → outcome, with the
   character gazing and reaching toward its choice and a payoff flash on wins.
6. Both bias sliders work; the live one visibly changes the high-vs-low-variance
   choice rate.
7. `agent.js` is the **only** place choice/learning logic lives, behind the
   `trainStep` / `decide` / `valueGrid` interface, ready to be replaced by a real
   model later.

> Note: the real trained model is **not** part of this task. Build the
> placeholder behind the interface above; the user will plug in the real model
> afterward.
