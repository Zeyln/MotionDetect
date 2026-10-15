// Checks the heatmap outline tracing on synthetic heat fields. Usage: node test/outline.test.js
'use strict';
const assert = require('node:assert/strict');
const { trace } = require('../outline.js');

// Soft disc: 255 at the centre falling to 0 at 2r.
function field(w, h, discs) {
  const f = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      for (const [cx, cy, r] of discs) {
        const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy);
        f[y * w + x] = Math.max(f[y * w + x], Math.round(255 * Math.max(0, 1 - d / (2 * r))));
      }
  return f;
}

// Every segment end must meet exactly one other segment end: the outline is closed.
function closed(segments) {
  const ends = new Map();
  const key = (x, y) => `${x.toFixed(4)},${y.toFixed(4)}`;
  for (let i = 0; i < segments.length; i += 2) {
    const k = key(segments[i], segments[i + 1]);
    ends.set(k, (ends.get(k) || 0) + 1);
  }
  return [...ends.values()].every((n) => n === 2);
}

const cases = [
  ['one disc: closed outline at the right radius', () => {
    const { segments, segmentAreas, count } = trace(field(64, 48, [[30, 20, 10]]), 64, 48, 128);
    assert.equal(count, 1);
    assert.ok(segments.length > 40, 'has an outline');
    assert.ok(closed(segments), 'outline is closed');
    for (let i = 0; i < segments.length; i += 2) {
      const d = Math.hypot(segments[i] - 30, segments[i + 1] - 20);
      // Level 128 of 255 is at about half of 2r, i.e. r.
      assert.ok(Math.abs(d - 10) < 0.8, `point at distance ${d.toFixed(2)} should be about 10`);
    }
    assert.ok(segmentAreas.every((a) => a === 1));
  }],
  ['two discs: two areas, each outline labelled with its own', () => {
    const { segments, segmentAreas, count } = trace(field(80, 40, [[20, 20, 6], [60, 20, 6]]), 80, 40, 128);
    assert.equal(count, 2);
    assert.ok(closed(segments));
    const leftLabel = segmentAreas[segments.findIndex((v, i) => i % 4 === 0 && v < 40) / 4];
    for (let s = 0; s < segmentAreas.length; s++) {
      const left = segments[4 * s] < 40;
      assert.equal(segmentAreas[s] === leftLabel, left, 'segment labelled with the disc it surrounds');
    }
  }],
  ['area touching the edge still gets a closed outline', () => {
    const { segments, count } = trace(field(40, 30, [[2, 15, 8]]), 40, 30, 128);
    assert.equal(count, 1);
    assert.ok(closed(segments), 'closed along the field edge');
  }],
  ['nothing above the level: no outline', () => {
    const { segments, count } = trace(new Uint8Array(100).fill(20), 10, 10, 128);
    assert.equal(count, 0);
    assert.equal(segments.length, 0);
  }],
];

let failed = 0;
for (const [name, fn] of cases) {
  try {
    fn();
    console.log(`PASS  outline: ${name}`);
  } catch (e) {
    failed++;
    console.log(`FAIL  outline: ${name}\n      ${e.message}`);
  }
}
process.exit(failed ? 1 : 0);
