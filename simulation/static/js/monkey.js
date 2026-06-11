// Procedural low-poly primate models for three species.
// Each builder returns a THREE.Group with named handles used for animation:
//   group.userData.head      -> head pivot (gaze)
//   group.userData.armL      -> left upper-arm pivot (reach/point)
//   group.userData.armR      -> right upper-arm pivot
import * as THREE from 'three';

const SPECIES = {
  marmoset: {
    fur: 0x6b5644, faceSkin: 0xe9d9c5, tuft: 0xf4efe6,
    scale: 0.62, headR: 0.46, bodyH: 0.95, ears: 'tuft', muzzle: 0.18,
  },
  macaque: {
    fur: 0x8a7a5c, faceSkin: 0xc98f7a, tuft: null,
    scale: 1.0, headR: 0.52, bodyH: 1.25, ears: 'round', muzzle: 0.34,
  },
  chimpanzee: {
    fur: 0x2b2620, faceSkin: 0x5a4034, tuft: null,
    scale: 1.28, headR: 0.6, bodyH: 1.55, ears: 'big', muzzle: 0.3,
  },
};

function mat(color, rough = 0.85) {
  return new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: 0.02 });
}

export function buildMonkey(species) {
  const cfg = SPECIES[species] || SPECIES.macaque;
  const g = new THREE.Group();
  const fur = mat(cfg.fur);
  const skin = mat(cfg.faceSkin, 0.7);

  // ---- Torso (seated) ----
  const torso = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.42, cfg.bodyH * 0.55, 6, 14), fur);
  torso.position.y = cfg.bodyH * 0.5;
  torso.scale.set(1.05, 1, 0.8);
  torso.castShadow = true;
  g.add(torso);

  const belly = new THREE.Mesh(new THREE.SphereGeometry(0.4, 16, 14), mat(cfg.fur, 0.9));
  belly.position.set(0, cfg.bodyH * 0.42, 0.12);
  belly.scale.set(1, 1.1, 0.85);
  g.add(belly);

  // ---- Head pivot ----
  const head = new THREE.Group();
  head.position.set(0, cfg.bodyH + cfg.headR * 0.75, 0.04);
  g.add(head);

  const skull = new THREE.Mesh(new THREE.SphereGeometry(cfg.headR, 20, 18), fur);
  skull.castShadow = true;
  head.add(skull);

  // Face patch
  const face = new THREE.Mesh(new THREE.SphereGeometry(cfg.headR * 0.86, 18, 16), skin);
  face.position.set(0, -0.02, cfg.headR * 0.42);
  face.scale.set(0.92, 1, 0.6);
  head.add(face);

  // Muzzle
  const muzzle = new THREE.Mesh(
    new THREE.SphereGeometry(cfg.headR * cfg.muzzle, 14, 12), skin);
  muzzle.position.set(0, -cfg.headR * 0.38, cfg.headR * 0.72);
  muzzle.scale.set(1, 0.8, 1.1);
  head.add(muzzle);

  // Eyes
  const eyeMat = mat(0x14110d, 0.3);
  const whiteMat = mat(0xf7f3ec, 0.4);
  for (const sx of [-1, 1]) {
    const sclera = new THREE.Mesh(new THREE.SphereGeometry(cfg.headR * 0.17, 12, 12), whiteMat);
    sclera.position.set(sx * cfg.headR * 0.34, cfg.headR * 0.12, cfg.headR * 0.78);
    head.add(sclera);
    const pupil = new THREE.Mesh(new THREE.SphereGeometry(cfg.headR * 0.09, 10, 10), eyeMat);
    pupil.position.set(sx * cfg.headR * 0.34, cfg.headR * 0.12, cfg.headR * 0.9);
    head.add(pupil);
  }

  // Ears / tufts
  for (const sx of [-1, 1]) {
    if (cfg.ears === 'tuft' && cfg.tuft != null) {
      const tuft = new THREE.Mesh(new THREE.ConeGeometry(cfg.headR * 0.28, cfg.headR * 0.7, 8),
        mat(cfg.tuft, 0.95));
      tuft.position.set(sx * cfg.headR * 0.7, cfg.headR * 0.4, 0);
      tuft.rotation.z = sx * 0.5;
      head.add(tuft);
    } else {
      const r = cfg.ears === 'big' ? cfg.headR * 0.4 : cfg.headR * 0.26;
      const ear = new THREE.Mesh(new THREE.CircleGeometry(r, 14), skin);
      ear.position.set(sx * cfg.headR * 0.92, cfg.headR * 0.05, 0);
      ear.rotation.y = sx * Math.PI / 2;
      ear.material.side = THREE.DoubleSide;
      head.add(ear);
    }
  }

  // ---- Arms (pivot at shoulder) ----
  function makeArm(side) {
    const pivot = new THREE.Group();
    pivot.position.set(side * 0.42, cfg.bodyH * 0.78, 0.05);
    const upper = new THREE.Mesh(new THREE.CapsuleGeometry(0.13, 0.5, 5, 10), fur);
    upper.position.set(side * 0.05, -0.3, 0.05);
    upper.rotation.z = -side * 0.25;
    upper.castShadow = true;
    pivot.add(upper);
    const hand = new THREE.Mesh(new THREE.SphereGeometry(0.14, 10, 10), skin);
    hand.position.set(side * 0.12, -0.62, 0.18);
    pivot.add(hand);
    g.add(pivot);
    return pivot;
  }
  const armL = makeArm(-1);
  const armR = makeArm(1);

  // ---- Legs (folded, seated) ----
  for (const sx of [-1, 1]) {
    const thigh = new THREE.Mesh(new THREE.CapsuleGeometry(0.16, 0.42, 5, 10), fur);
    thigh.position.set(sx * 0.24, cfg.bodyH * 0.16, 0.32);
    thigh.rotation.x = 1.15;
    thigh.castShadow = true;
    g.add(thigh);
    const foot = new THREE.Mesh(new THREE.SphereGeometry(0.15, 10, 10), skin);
    foot.position.set(sx * 0.26, 0.06, 0.6);
    g.add(foot);
  }

  // Tail (longer/curlier for marmoset & macaque, stub for chimp)
  if (species !== 'chimpanzee') {
    const tail = new THREE.Mesh(new THREE.CapsuleGeometry(0.07, cfg.bodyH, 5, 8),
      mat(cfg.fur, 0.95));
    tail.position.set(0.3, cfg.bodyH * 0.3, -0.35);
    tail.rotation.set(0.6, 0, -0.8);
    g.add(tail);
  }

  g.scale.setScalar(cfg.scale);
  g.userData = { head, armL, armR, species };
  return g;
}

export const SPECIES_NAMES = Object.keys(SPECIES);
