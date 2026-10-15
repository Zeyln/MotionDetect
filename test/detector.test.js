// Runs the detector on synthetic frames: a textured scene viewed through a "camera" that pans,
// optionally with a square moving independently. Usage: node test/detector.test.js
'use strict';
const assert = require('node:assert/strict');
const path = require('node:path');
const { createDetector } = require('../detector.js');

const W = 320;
const H = 240;
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

// Camera at (camX, camY) in the scene; optional object square drawn in frame coordinates.
function renderFrame(scene, camX, camY, obj) {
  const out = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      // Bilinear sample so sub-pixel camera motion looks like a real camera.
      const sx = x + camX;
      const sy = y + camY;
      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const fx = sx - x0;
      const fy = sy - y0;
      const p = (xx, yy) => scene[yy * SCENE_W + xx];
      let v =
        p(x0, y0) * (1 - fx) * (1 - fy) + p(x0 + 1, y0) * fx * (1 - fy) +
        p(x0, y0 + 1) * (1 - fx) * fy + p(x0 + 1, y0 + 1) * fx * fy;
      if (obj && x >= obj.x && x < obj.x + obj.s && y >= obj.y && y < obj.y + obj.s) {
        v = ((x - obj.x) >> 3) % 2 === ((y - obj.y) >> 3) % 2 ? 250 : 10; // checkerboard object
      }
      const i = 4 * (y * W + x);
      out[i] = out[i + 1] = out[i + 2] = v;
      out[i + 3] = 255;
    }
  }
  return out;
}

function run(cv, defaultScene, { pan, object, scene = defaultScene, frames = 30 }) {
  const det = createDetector(cv);
  const results = [];
  for (let f = 0; f < frames; f++) {
    const camX = 100 + pan.x * f;
    const camY = 100 + pan.y * f + 2 * Math.sin(f / 3); // a little hand shake
    const obj = object ? { x: 40 + object.vx * f, y: 90, s: 40 } : null;
    const r = det.processFrame(renderFrame(scene, camX, camY, obj), W, H);
    results.push({ ...r, obj });
  }
  det.dispose();
  return results.slice(5); // let the detector settle
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
  ];

  let failed = 0;
  for (const c of cases) {
    const results = run(cv, scene, c);
    const lost = results.filter((r) => r.mode !== 'ok').length;
    let falseBoxes = 0;
    let hits = 0;
    for (const r of results) {
      for (const b of r.boxes) {
        if (r.obj && contains(b, r.obj, 10)) hits++;
        else falseBoxes++;
      }
    }
    const avgInliers = Math.round(results.reduce((s, r) => s + r.inliers, 0) / results.length);
    const summary = `frames=${results.length} notOk=${lost} hits=${hits} falseBoxes=${falseBoxes} avgInliers=${avgInliers}`;
    try {
      assert.equal(lost, 0, 'every frame should track');
      assert.ok(falseBoxes <= 1, 'at most one stray box across the run');
      if (c.object) assert.ok(hits >= results.length - 2, 'moving object boxed in nearly every frame');
      console.log(`PASS  ${c.name}  (${summary})`);
    } catch (e) {
      failed++;
      console.log(`FAIL  ${c.name}  (${summary})\n      ${e.message}`);
    }
  }
  process.exit(failed ? 1 : 0);
})();
