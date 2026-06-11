// Plain-canvas 2D charts for the analytics rail + the clickable option grid.

export class LearningCurve {
  constructor(canvas) {
    this.ctx = canvas.getContext('2d');
    this.w = canvas.width; this.h = canvas.height;
    this.data = [];
  }
  reset() { this.data = []; this.draw(); }
  push(rec) { this.data.push(rec); this.draw(); }
  draw() {
    const { ctx, w, h } = this;
    ctx.clearRect(0, 0, w, h);
    const pad = { l: 34, r: 8, t: 10, b: 20 };
    const x0 = pad.l, x1 = w - pad.r, y0 = h - pad.b, y1 = pad.t;
    // axes
    ctx.strokeStyle = '#2d333b'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x0, y0); ctx.lineTo(x1, y0); ctx.stroke();
    if (this.data.length < 2) {
      ctx.fillStyle = '#8b949e'; ctx.font = '11px monospace';
      ctx.fillText('waiting for training…', x0 + 8, (y0 + y1) / 2);
      return;
    }
    const iters = this.data.map(d => d.iter);
    const imin = Math.min(...iters), imax = Math.max(...iters);
    const rewards = this.data.map(d => d.mean_reward ?? 0);
    let rmin = Math.min(...rewards), rmax = Math.max(...rewards);
    if (rmax - rmin < 1e-6) { rmax += 1; rmin -= 1; }
    const sx = v => x0 + (x1 - x0) * (imax === imin ? 0.5 : (v - imin) / (imax - imin));
    const syR = v => y0 - (y0 - y1) * (v - rmin) / (rmax - rmin);
    const syP = v => y0 - (y0 - y1) * v; // P in [0,1]

    // gridline labels (reward)
    ctx.fillStyle = '#8b949e'; ctx.font = '9px monospace'; ctx.textAlign = 'right';
    ctx.fillText(rmax.toFixed(1), x0 - 4, y1 + 8);
    ctx.fillText(rmin.toFixed(1), x0 - 4, y0);
    ctx.textAlign = 'left';

    const line = (vals, sy, color) => {
      ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.beginPath();
      let started = false;
      this.data.forEach((d, i) => {
        const v = vals(d); if (v == null || isNaN(v)) return;
        const px = sx(d.iter), py = sy(v);
        started ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        started = true;
      });
      ctx.stroke();
    };
    line(d => d.mean_reward, syR, '#f0b429');
    line(d => d.p_correct, syP, '#58a6ff');

    // x label
    ctx.fillStyle = '#8b949e'; ctx.font = '9px monospace';
    ctx.fillText(`iter ${imax}`, x1 - 44, y0 + 14);
  }
}

export class ValueGrid {
  constructor(canvas) {
    this.ctx = canvas.getContext('2d');
    this.w = canvas.width; this.h = canvas.height;
    this.grid = null;
  }
  set(grid) { this.grid = grid; this.draw(); }
  draw() {
    const { ctx, w, h } = this;
    ctx.clearRect(0, 0, w, h);
    const pad = 24;
    const cw = (w - pad) / 5, ch = (h - pad) / 5;
    if (!this.grid) {
      ctx.fillStyle = '#8b949e'; ctx.font = '11px monospace';
      ctx.fillText('run trials to estimate values', 10, h / 2);
      return;
    }
    let lo = Infinity, hi = -Infinity;
    for (const row of this.grid) for (const v of row) {
      if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-6) hi = lo + 1;
    const probs = [0.1, 0.3, 0.5, 0.7, 0.9];
    for (let r = 0; r < 5; r++) {
      for (let c = 0; c < 5; c++) {
        const v = this.grid[r][c];
        const x = pad + c * cw, y = r * ch;
        if (v == null) { ctx.fillStyle = '#161b22'; }
        else {
          const t = (v - lo) / (hi - lo);
          ctx.fillStyle = viridis(t);
        }
        ctx.fillRect(x + 1, y + 1, cw - 2, ch - 2);
        if (v != null) {
          ctx.fillStyle = '#0b0f14'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
          ctx.fillText(v.toFixed(2), x + cw / 2, y + ch / 2 + 3);
        }
      }
      ctx.fillStyle = '#8b949e'; ctx.font = '9px monospace'; ctx.textAlign = 'right';
      ctx.fillText(probs[r].toFixed(1), pad - 3, r * ch + ch / 2 + 3);
    }
    ctx.textAlign = 'center'; ctx.fillStyle = '#8b949e';
    ctx.fillText('reward →', w / 2 + pad / 2, h - 4);
  }
}

// Clickable 5x5 option grid (the environment editor). Cell index = row*5 + col.
export class OptionGrid {
  constructor(canvas, colorVector, onPick) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.colors = colorVector;
    this.onPick = onPick;
    this.selL = 0; this.selR = 24;
    canvas.addEventListener('click', e => this._click(e));
    this.draw();
  }
  _click(e) {
    const rect = this.canvas.getBoundingClientRect();
    const cx = (e.clientX - rect.left) / rect.width * this.canvas.width;
    const cy = (e.clientY - rect.top) / rect.height * this.canvas.height;
    const cell = this.canvas.width / 5;
    const col = Math.floor(cx / cell), row = Math.floor(cy / cell);
    if (col < 0 || col > 4 || row < 0 || row > 4) return;
    const idx = row * 5 + col;
    // cycle: first click sets LEFT, next sets RIGHT
    if (this._next === 'R') { this.selR = idx; this._next = 'L'; }
    else { this.selL = idx; this._next = 'R'; }
    this.draw();
    if (this.onPick) this.onPick(this.selL, this.selR);
  }
  draw() {
    const { ctx, canvas } = this;
    const cell = canvas.width / 5;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
      const idx = r * 5 + c;
      const col = this.colors[idx];
      ctx.fillStyle = `rgb(${col[0]*255|0},${col[1]*255|0},${col[2]*255|0})`;
      ctx.fillRect(c * cell + 2, r * cell + 2, cell - 4, cell - 4);
      if (idx === this.selL || idx === this.selR) {
        ctx.strokeStyle = idx === this.selL ? '#58a6ff' : '#3fb950';
        ctx.lineWidth = 3;
        ctx.strokeRect(c * cell + 3, r * cell + 3, cell - 6, cell - 6);
        ctx.fillStyle = idx === this.selL ? '#58a6ff' : '#3fb950';
        ctx.font = 'bold 11px monospace'; ctx.textAlign = 'center';
        ctx.fillText(idx === this.selL ? 'L' : 'R', c * cell + cell / 2, r * cell + cell / 2 + 4);
      }
    }
  }
}

function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
  ];
  const x = t * (stops.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
