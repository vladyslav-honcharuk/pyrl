// Three.js scene: lab room, chair, monkey facing a screen that shows the two
// gambling options. Exposes methods to swap species, set the displayed options,
// and animate a trial (gaze + reach toward the chosen side, reward flash).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { buildMonkey } from './monkey.js';

export class Lab {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0e13);
    this.scene.fog = new THREE.Fog(0x0a0e13, 9, 22);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    this.camera.position.set(2.6, 2.3, 5.2);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 1.4, -1.5);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.52;
    this.controls.minDistance = 2.5;
    this.controls.maxDistance = 12;

    this._buildRoom();
    this._buildScreen();
    this.setSpecies('macaque');

    this.clock = new THREE.Clock();
    this.anim = null;            // active trial animation state
    this._resize();
    window.addEventListener('resize', () => this._resize());
    this._loop();
  }

  _buildRoom() {
    const lights = new THREE.Group();
    const amb = new THREE.AmbientLight(0x4a5568, 0.9);
    lights.add(amb);
    const key = new THREE.DirectionalLight(0xfff3e0, 1.5);
    key.position.set(4, 8, 6);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 1; key.shadow.camera.far = 30;
    key.shadow.camera.left = -8; key.shadow.camera.right = 8;
    key.shadow.camera.top = 8; key.shadow.camera.bottom = -8;
    lights.add(key);
    const fill = new THREE.DirectionalLight(0x6ea8ff, 0.5);
    fill.position.set(-5, 4, 2);
    lights.add(fill);
    this.scene.add(lights);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(40, 40),
      new THREE.MeshStandardMaterial({ color: 0x12171f, roughness: 0.95 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);

    const backWall = new THREE.Mesh(
      new THREE.PlaneGeometry(40, 14),
      new THREE.MeshStandardMaterial({ color: 0x0e131a, roughness: 1 }));
    backWall.position.set(0, 7, -4.2);
    this.scene.add(backWall);

    // Primate chair
    const chairMat = new THREE.MeshStandardMaterial({ color: 0x3a4250, roughness: 0.6, metalness: 0.3 });
    const seat = new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.25, 1.4), chairMat);
    seat.position.set(0, 0.65, 0.1);
    seat.castShadow = true; seat.receiveShadow = true;
    this.scene.add(seat);
    const back = new THREE.Mesh(new THREE.BoxGeometry(1.5, 1.8, 0.22), chairMat);
    back.position.set(0, 1.55, 0.78);
    back.castShadow = true;
    this.scene.add(back);
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.65, 10), chairMat);
        leg.position.set(sx * 0.6, 0.32, 0.1 + sz * 0.55);
        this.scene.add(leg);
      }
    }
  }

  _buildScreen() {
    // A monitor in front of the monkey showing the two options.
    const frame = new THREE.Mesh(
      new THREE.BoxGeometry(3.6, 2.3, 0.16),
      new THREE.MeshStandardMaterial({ color: 0x05070a, roughness: 0.5, metalness: 0.5 }));
    frame.position.set(0, 1.9, -2.9);
    this.scene.add(frame);

    const screenMat = new THREE.MeshStandardMaterial({
      color: 0x05080d, roughness: 0.35, emissive: 0x0a0f16, emissiveIntensity: 1 });
    const screen = new THREE.Mesh(new THREE.PlaneGeometry(3.3, 2.0), screenMat);
    screen.position.set(0, 1.9, -2.81);
    this.scene.add(screen);

    // Fixation point
    this.fixPoint = new THREE.Mesh(
      new THREE.CircleGeometry(0.08, 16),
      new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 1.4 }));
    this.fixPoint.position.set(0, 1.9, -2.79);
    this.scene.add(this.fixPoint);

    // Two option discs (left/right)
    const makeOption = (x) => {
      const disc = new THREE.Mesh(
        new THREE.CircleGeometry(0.42, 32),
        new THREE.MeshStandardMaterial({ color: 0x222222, emissive: 0x000000, emissiveIntensity: 1.2 }));
      disc.position.set(x, 1.9, -2.79);
      disc.visible = false;
      this.scene.add(disc);
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.46, 0.54, 32),
        new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 }));
      ring.position.set(x, 1.9, -2.78);
      this.scene.add(ring);
      return { disc, ring, x };
    };
    this.optL = makeOption(-1.0);
    this.optR = makeOption(1.0);

    // Reward flash plane
    this.reward = new THREE.Mesh(
      new THREE.PlaneGeometry(3.3, 2.0),
      new THREE.MeshBasicMaterial({ color: 0xf0b429, transparent: true, opacity: 0 }));
    this.reward.position.set(0, 1.9, -2.78);
    this.scene.add(this.reward);
  }

  setSpecies(species) {
    if (this.monkey) this.scene.remove(this.monkey);
    this.monkey = buildMonkey(species);
    this.monkey.position.set(0, 0.78, 0.1);
    this.scene.add(this.monkey);
  }

  setOptions(colorL, colorR) {
    const c = (rgb) => new THREE.Color(rgb[0], rgb[1], rgb[2]);
    for (const [opt, col] of [[this.optL, colorL], [this.optR, colorR]]) {
      if (!col) { opt.disc.visible = false; continue; }
      opt.disc.visible = true;
      opt.disc.material.color = c(col);
      opt.disc.material.emissive = c(col);
    }
  }

  clearOptions() {
    this.optL.disc.visible = false;
    this.optR.disc.visible = false;
  }

  // timeline: [{t, phase, action, value}], choice: 'L'|'R'|null, reward, colors
  playTrial(trial, onPhase, onDone) {
    this.setOptions(trial.color_l, trial.color_r);
    const dur = 0.06; // seconds per task timestep
    this.anim = {
      trial, t0: this.clock.getElapsedTime(), dur,
      n: trial.timeline.length, onPhase, onDone, done: false, lastPhase: null,
    };
  }

  _animateTrial(now) {
    const a = this.anim;
    const idx = Math.min(Math.floor((now - a.t0) / a.dur), a.n - 1);
    const step = a.trial.timeline[idx];
    const head = this.monkey.userData.head;

    // Gaze: fixation -> center; stimulus -> scan; decision -> toward choice
    let targetYaw = 0;
    this.fixPoint.material.opacity = step.phase === 'decision' ? 0.15 : 1;
    if (step.phase === 'decision' && a.trial.choice) {
      targetYaw = a.trial.choice === 'L' ? 0.4 : -0.4;
    } else if (step.phase === 'stimulus') {
      targetYaw = Math.sin(now * 4) * 0.18;
    }
    head.rotation.y += (targetYaw - head.rotation.y) * 0.18;
    head.rotation.x += (-0.05 - head.rotation.x) * 0.1;

    // Reach when choice happens
    const armR = this.monkey.userData.armR, armL = this.monkey.userData.armL;
    const reaching = (a.trial.choice_t != null && idx >= a.trial.choice_t);
    const arm = a.trial.choice === 'L' ? armL : armR;
    const other = a.trial.choice === 'L' ? armR : armL;
    const rest = 0;
    if (reaching && a.trial.choice) {
      arm.rotation.x += (-1.35 - arm.rotation.x) * 0.2;
      arm.rotation.z += ((a.trial.choice === 'L' ? 0.5 : -0.5) - arm.rotation.z) * 0.2;
    } else {
      armR.rotation.x += (rest - armR.rotation.x) * 0.15;
      armL.rotation.x += (rest - armL.rotation.x) * 0.15;
      armR.rotation.z += (rest - armR.rotation.z) * 0.15;
      armL.rotation.z += (rest - armL.rotation.z) * 0.15;
    }

    // Highlight chosen option
    const chosen = a.trial.choice === 'L' ? this.optL : (a.trial.choice === 'R' ? this.optR : null);
    for (const opt of [this.optL, this.optR]) {
      const tgt = (opt === chosen && reaching) ? 0.95 : 0;
      opt.ring.material.opacity += (tgt - opt.ring.material.opacity) * 0.2;
      opt.ring.material.color.set(a.trial.reward > 0 ? 0xf0b429 : 0xffffff);
    }

    if (step.phase !== a.lastPhase) {
      a.lastPhase = step.phase;
      if (a.onPhase) a.onPhase(step.phase, step, a.trial);
    }

    // End: reward flash, then settle
    if (idx >= a.n - 1) {
      if (!a.done) {
        a.done = true; a.flashStart = now;
        if (a.onDone) a.onDone(a.trial);
      }
      const f = Math.max(0, 1 - (now - a.flashStart) * 1.5);
      this.reward.material.opacity = (a.trial.reward > 0 ? 0.5 : 0.08) * f;
      if (now - a.flashStart > 1.4) { this.anim = null; }
    }
  }

  _idleBreath(now) {
    if (this.monkey) {
      this.monkey.position.y = 0.78 + Math.sin(now * 1.4) * 0.012;
      const head = this.monkey.userData.head;
      head.rotation.y += (0 - head.rotation.y) * 0.05;
    }
  }

  _resize() {
    const w = window.innerWidth, h = window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _loop() {
    requestAnimationFrame(() => this._loop());
    const now = this.clock.getElapsedTime();
    if (this.anim) this._animateTrial(now); else this._idleBreath(now);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
