// Runs the detector on synthetic frames: a textured scene viewed through a "camera" that pans,
// optionally with a square moving independently. Every case runs at 320x240 and 640x480.
// Usage: node test/detector.test.js
'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
const { createDetector } = require('../detector.js');

const SCENE_W = 900;
const SCENE_H = 700;

// The OpenCV.js module object has a `then` method, so it is wrapped to stop promises from
// trying to adopt it (which never settles).
async function loadCv() {
  const cv = require(path.join(__dirname, '..', 'vendor', 'opencv.js'));
  if (!cv.Mat) await new Promise((resolve) => (cv.onRuntimeInitialized = resolve));
  return { cv };
}

// Deterministic blotchy texture so the feature tracker has corners to lock onto.
function makeScene(blobs = 2500) {
  let seed = 12345;
  const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
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
function renderFrame(scene, k, camX, camY, obj, gain) {
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
      if (obj && x >= obj.x && x < obj.x + obj.s && y >= obj.y && y < obj.y + obj.s) {
        const cell = 8 * k;
        v = Math.floor((x - obj.x) / cell) % 2 === Math.floor((y - obj.y) / cell) % 2 ? 250 : 10; // checkerboard
      }
      const i = 4 * (y * W + x);
      out[i] = out[i + 1] = out[i + 2] = v * gain;
      out[i + 3] = 255;
    }
  }
  return out;
}

function run(cv, defaultScene, k, { pan, object, scene = defaultScene, exposureJump = null, frames = 30 }) {
  const det = createDetector(cv);
  const results = [];
  let ms = 0;
  for (let f = 0; f < frames; f++) {
    const camX = 100 + pan.x * f;
    const camY = 100 + pan.y * f + 2 * Math.sin(f / 3); // a little hand shake
    const obj = object ? { x: k * (40 + object.vx * f), y: 90 * k, s: 40 * k } : null;
    const gain = exposureJump && f >= exposureJump.frame ? exposureJump.gain : 1;
    const frame = renderFrame(scene, k, camX, camY, obj, gain);
    const t0 = performance.now();
    const r = det.processFrame(frame, 320 * k, 240 * k);
    ms += performance.now() - t0;
    results.push({ ...r, obj });
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
  const cases = [
    { name: 'still camera, nothing moving', pan: { x: 0, y: 0 }, object: null },
    { name: 'panning camera, nothing moving', pan: { x: 4, y: 1.5 }, object: null },
    { name: 'still camera, object moving', pan: { x: 0, y: 0 }, object: { vx: 6 } },
    { name: 'panning camera, object moving', pan: { x: 4, y: 1.5 }, object: { vx: 6 } },
    // Plain wall: almost nothing for the tracker to lock onto except the moving object itself.
    { name: 'low-texture scene, object moving', pan: { x: 0, y: 0 }, object: { vx: 6 }, scene: makeScene(0) },
    // Auto exposure brightens the whole picture by 20% from one frame to the next.
    { name: 'exposure jump while panning', pan: { x: 4, y: 1.5 }, object: null, exposureJump: { frame: 15, gain: 1.2 } },
  ];

  let failed = 0;
  for (const k of [1, 2]) {
    for (const c of cases) {
      const { results, ms } = run(cv, scene, k, c);
      const lost = results.filter((r) => r.mode !== 'ok').length;
      let falseBoxes = 0;
      let hits = 0;
      let maxExcess = 0; // how far a hit box extends beyond the object's own size, in frame px
      for (const r of results) {
        for (const b of r.boxes) {
          if (r.obj && contains(b, r.obj, 10 * k)) {
            hits++;
            maxExcess = Math.max(maxExcess, b.w - r.obj.s, b.h - r.obj.s);
          } else falseBoxes++;
        }
      }
      const name = `${320 * k}px  ${c.name}`;
      const summary = `notOk=${lost} hits=${hits}/${results.length} false=${falseBoxes}` +
        (c.object ? ` excess=${maxExcess}px` : '') + ` ${ms.toFixed(1)}ms/frame`;
      try {
        assert.equal(lost, 0, 'every frame should track');
        assert.ok(falseBoxes <= 1, 'at most one stray box across the run');
        if (c.object) {
          assert.ok(hits >= results.length - 2, 'moving object boxed in nearly every frame');
          // A tight box spans the object's old and new positions: its size plus one frame of travel
          // relative to the background (its own motion plus the pan, plus up to 2 px of hand shake).
          const travel = c.object.vx + c.pan.x + c.pan.y + 2;
          assert.ok(maxExcess <= (travel + 2) * k, 'box should hug the object');
        }
        console.log(`PASS  ${name}  (${summary})`);
      } catch (e) {
        failed++;
        console.log(`FAIL  ${name}  (${summary})\n      ${e.message}`);
      }
    }
  }
  process.exit(failed ? 1 : 0);
})();
