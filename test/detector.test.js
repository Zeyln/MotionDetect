// Runs the detector on synthetic frames: a textured scene viewed through a "camera" that pans,
// optionally with a square that moves (and maybe stops) in the scene. Every case runs at 320x240
// and 640x480, with mild sensor noise, at 25 fps. Usage: node test/detector.test.js
'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
const { createDetector } = require('../detector.js');

const SCENE_W = 900;
const SCENE_H = 700;
const FRAME_MS = 40;

// The OpenCV.js module object has a `then` method, so it is wrapped to stop promises from
// trying to adopt it (which never settles).
async function loadCv() {
  const cv = require(path.join(__dirname, '..', 'vendor', 'opencv.js'));
  if (!cv.Mat) await new Promise((resolve) => (cv.onRuntimeInitialized = resolve));
  return { cv };
}

function makeRandom(seed) {
  return () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
}

// Deterministic blotchy texture so the feature tracker has corners to lock onto.
function makeScene(blobs = 2500) {
  const rand = makeRandom(12345);
  const scene = new Uint8Array(SCENE_W * SCENE_H).fill(128);
  for (let n = 0; n < blobs; n++) {
    const cx = rand() * SCENE_W;
    const cy = rand() * SCENE_H;
    const s = 3 + rand() * 12;
    const v = Math.floor(rand() * 255);
    for (let y = Math.max(0, cy - s | 0); y < Math.min(SCENE_H, cy + s); y++)
      for (let x = Math.max(0, cx - s | 0); x < Math.min(SCENE_W, cx + s); x++) scene[y * SCENE_W + x] = v;
  }
  return scene;
}

// Renders a k*320 x k*240 frame of the scene with the camera at (camX, camY) in scene units, so higher
// k samples the same view more finely. `obj` is in frame pixels; `gain` simulates exposure changes.
function renderFrame(scene, k, camX, camY, obj, gain, rand) {
  const W = 320 * k;
  const H = 240 * k;
  const out = new Uint8ClampedArray(W * H * 4);
  const p = (xx, yy) => scene[yy * SCENE_W + xx];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      // Bilinear sample so sub-pixel camera motion looks like a real camera.
      const sx = x / k + camX;
      const sy = y / k + camY;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const fx = sx - x0;
      const fy = sy - y0;
      let v =
        p(x0, y0) * (1 - fx) * (1 - fy) + p(x0 + 1, y0) * fx * (1 - fy) +
        p(x0, y0 + 1) * (1 - fx) * fy + p(x0 + 1, y0 + 1) * fx * fy;
      if (obj) {
        // Sub-pixel coverage, so a still object doesn't jitter against the smoothly sampled scene.
        const cover =
          Math.max(0, Math.min(x + 1, obj.x + obj.s) - Math.max(x, obj.x)) *
          Math.max(0, Math.min(y + 1, obj.y + obj.s) - Math.max(y, obj.y));
        if (cover > 0) {
          const cell = 8 * k;
          const lx = Math.max(0, x + 0.5 - obj.x);
          const ly = Math.max(0, y + 0.5 - obj.y);
          // Checkerboard, or for a tiny object (a bird against the sky) the opposite of whatever is
          // behind it: a single-colour speck can vanish against a matching patch of texture.
          const ov = obj.contrast ? (v < 128 ? 250 : 10)
            : Math.floor(lx / cell) % 2 === Math.floor(ly / cell) % 2 ? 250 : 10;
          v = v * (1 - cover) + ov * cover;
        }
      }
      const noise = (rand() + rand() + rand() - 1.5) * 8; // roughly Gaussian, sigma about 4
      const i = 4 * (y * W + x);
      out[i] = out[i + 1] = out[i + 2] = v * gain + noise;
      out[i + 3] = 255;
    }
  }
  return out;
}

// object: { vx, size = 40, stopAt } in scene units per frame; it sits still from frame `stopAt` on.
function run(cv, defaultScene, k, c) {
  const { pan, object, scene = defaultScene, exposureJump = null, flicker = null, frames = 30, options } = c;
  const det = createDetector(cv, options);
  const rand = makeRandom(99);
  const results = [];
  let ms = 0;
  for (let f = 0; f < frames; f++) {
    const camX = 100 + pan.x * f;
    const camY = 100 + pan.y * f + 2 * Math.sin(f / 3); // a little hand shake
    let obj = null;
    let moving = false;
    if (object) {
      const steps = object.stopAt == null ? f : Math.min(f, object.stopAt);
      moving = object.stopAt == null || f <= object.stopAt;
      const size = object.size || 40;
      const sceneX = 140 + object.vx * steps;
      const sceneY = 190;
      obj = { x: (sceneX - camX) * k, y: (sceneY - camY) * k, s: size * k, contrast: size < 8 };
    }
    const gain = exposureJump && f >= exposureJump.frame ? exposureJump.gain : 1;
    const frame = renderFrame(scene, k, camX, camY, obj, gain, rand);
    if (flicker && f % flicker.every === 2) {
      // A one-frame glitch (light reflection, compression artifact) over a patch of the frame.
      for (let y = 60 * k; y < 100 * k; y++)
        for (let x = 200 * k; x < 260 * k; x++)
          for (let ch = 0; ch < 3; ch++) frame[4 * (y * 320 * k + x) + ch] += 70;
    }
    const t0 = performance.now();
    const r = det.processFrame(frame, 320 * k, 240 * k, f * FRAME_MS);
    ms += performance.now() - t0;
    results.push({ ...r, f, obj, moving });
  }
  det.dispose();
  return { results: results.slice(5), ms: ms / frames }; // let the detector settle
}

function contains(box, obj, slack) {
  const cx = obj.x + obj.s / 2;
  const cy = obj.y + obj.s / 2;
  return cx > box.x - slack && cx < box.x + box.w + slack && cy > box.y - slack && cy < box.y + box.h + slack;
}

(async () => {
  const { cv } = await loadCv();
  const scene = makeScene();
  const still = { x: 0, y: 0 };
  const panning = { x: 4, y: 1.5 };
  const cases = [
    { name: 'still camera, nothing moving', pan: still },
    { name: 'panning camera, nothing moving', pan: panning },
    { name: 'still camera, object moving', pan: still, object: { vx: 6 } },
    { name: 'panning camera, object moving', pan: panning, object: { vx: 6 } },
    // Plain wall: almost nothing for the tracker to lock onto except the moving object itself.
    { name: 'low-texture scene, object moving', pan: still, object: { vx: 6 }, scene: makeScene(0) },
    // Auto exposure brightens the whole picture by 20% from one frame to the next.
    { name: 'exposure jump while panning', pan: panning, exposureJump: { frame: 15, gain: 1.2 } },
    // A bird in the distance: 5x5 px at 320 px.
    { name: 'panning camera, tiny object moving', pan: panning, object: { vx: 4, size: 5 } },
    // The target stops; its box must stay on it, following the scene while the camera pans.
    { name: 'still camera, object stops', pan: still, object: { vx: 6, stopAt: 14 } },
    { name: 'panning camera, object stops', pan: panning, object: { vx: 6, stopAt: 14 } },
    // Held boxes disappear once the hold time runs out (400 ms = 10 frames here).
    { name: 'held box expires', pan: still, object: { vx: 6, stopAt: 14 }, options: { holdMs: 400 } },
    // Moves twice its own size per frame, so the spot it just left is a separate blob each frame.
    // Those must not become boxes (or held boxes) trailing along its path.
    { name: 'panning camera, fast small object (no trail)', pan: panning, object: { vx: 16, size: 8 } },
    // Single-frame brightness glitches over a patch must not produce boxes.
    { name: 'panning camera, one-frame glitches', pan: panning, flicker: { every: 4 } },
  ];

  let failed = 0;
  for (const k of [1, 2]) {
    for (const c of cases) {
      const { results, ms } = run(cv, scene, k, c);
      const holdMs = c.options?.holdMs ?? 3000;
      const stopAt = c.object?.stopAt;
      const lost = results.filter((r) => r.mode !== 'ok').length;
      let falseBoxes = 0;
      let boxed = 0; // frames where the object should be boxed and is
      let expected = 0;
      let heldOk = 0; // after stopping: frames where the box is marked held
      let lingering = 0; // boxes still shown after the hold time ran out
      let maxExcess = 0; // how far a moving target's box extends beyond the object, in frame px
      for (const r of results) {
        const expired = stopAt != null && (r.f - stopAt) * FRAME_MS > holdMs + FRAME_MS;
        const hit = r.obj && r.boxes.find((b) => contains(b, r.obj, 10 * k));
        for (const b of r.boxes) {
          if (b === hit) continue;
          if (expired) lingering++;
          else falseBoxes++;
        }
        if (!r.obj) continue;
        const inView = r.obj.x >= 0 && r.obj.y >= 0 && r.obj.x + r.obj.s <= 320 * k && r.obj.y + r.obj.s <= 240 * k;
        if (!inView) continue; // it has left the frame; any box it leaves is checked as a stray above
        if (expired) {
          if (hit) lingering++;
          continue;
        }
        expected++;
        if (hit) boxed++;
        if (hit && r.moving) maxExcess = Math.max(maxExcess, hit.w - r.obj.s, hit.h - r.obj.s);
        if (hit && stopAt != null && r.f > stopAt + 1 && hit.held) heldOk++;
      }
      const afterStop = stopAt == null ? 0 : results.filter((r) => r.f > stopAt + 1 && (r.f - stopAt) * FRAME_MS <= holdMs).length;

      const name = `${320 * k}px  ${c.name}`;
      const summary = `notOk=${lost} boxed=${boxed}/${expected} false=${falseBoxes}` +
        (c.object ? ` excess=${maxExcess}px` : '') +
        (stopAt != null ? ` held=${heldOk}/${afterStop}` : '') +
        (c.options?.holdMs ? ` lingering=${lingering}` : '') +
        ` ${ms.toFixed(1)}ms/frame`;
      try {
        assert.equal(lost, 0, 'every frame should track');
        assert.ok(falseBoxes <= 1, 'at most one stray box across the run');
        if (c.object) {
          assert.ok(boxed >= expected - 1, 'object boxed in nearly every frame, moving or held');
          // A tight box spans the object's old and new positions: its size plus one frame of travel.
          assert.ok(maxExcess <= (c.object.vx + 4) * k, 'box should hug the object');
        }
        if (stopAt != null) assert.ok(heldOk >= afterStop - 1, 'box marked held once the object stops');
        if (c.options?.holdMs) assert.equal(lingering, 0, 'no boxes after the hold time');
        console.log(`PASS  ${name}  (${summary})`);
      } catch (e) {
        failed++;
        console.log(`FAIL  ${name}  (${summary})\n      ${e.message}`);
      }
    }
  }
  process.exit(failed ? 1 : 0);
})();
