/*
 * Traces the outline of the heatmap: the edge of the areas where heat is at least a given level.
 *
 * Marching squares with linear interpolation, so the line runs smoothly between heat samples rather
 * than following their square grid. Areas are also labelled (4-connected), so each piece of outline
 * knows which area it belongs to.
 *
 * Works in the page (global `outline`) and in Node (module.exports) for tests.
 */
(function (root) {
  'use strict';

  // Labels 4-connected areas where field >= level. Returns { labels: Int32Array, count }.
  function labelAreas(field, w, h, level) {
    const labels = new Int32Array(w * h);
    const stack = [];
    let count = 0;
    for (let start = 0; start < field.length; start++) {
      if (labels[start] || field[start] < level) continue;
      labels[start] = ++count;
      stack.push(start);
      while (stack.length) {
        const i = stack.pop();
        const x = i % w;
        const y = (i / w) | 0;
        if (x > 0 && !labels[i - 1] && field[i - 1] >= level) (labels[i - 1] = count), stack.push(i - 1);
        if (x < w - 1 && !labels[i + 1] && field[i + 1] >= level) (labels[i + 1] = count), stack.push(i + 1);
        if (y > 0 && !labels[i - w] && field[i - w] >= level) (labels[i - w] = count), stack.push(i - w);
        if (y < h - 1 && !labels[i + w] && field[i + w] >= level) (labels[i + w] = count), stack.push(i + w);
      }
    }
    return { labels, count };
  }

  // For each case (corner bits: top-left 8, top-right 4, bottom-right 2, bottom-left 1), the pairs of
  // cell edges an outline segment joins. Edges: 0 top, 1 right, 2 bottom, 3 left. Saddles (5, 10) are
  // resolved with the cell centre; these entries are for a centre below the level.
  const CASES = [
    [], [[3, 2]], [[2, 1]], [[3, 1]], [[0, 1]], [[0, 1], [3, 2]], [[0, 2]], [[3, 0]],
    [[3, 0]], [[0, 2]], [[3, 0], [2, 1]], [[0, 1]], [[3, 1]], [[2, 1]], [[3, 2]], [],
  ];
  const SADDLE_CENTRE_INSIDE = { 5: [[3, 0], [2, 1]], 10: [[0, 1], [3, 2]] };

  /**
   * Traces where `field` (w x h, e.g. Uint8Array) crosses `level`. Outside the field counts as 0, so
   * areas touching the edge still get a closed outline. Returns { segments, segmentAreas, labels,
   * count }: segments as [x0, y0, x1, y1, ...] in field coordinates (sample i is centred at i + 0.5),
   * and for each segment the label of the area it surrounds (labels as from labelAreas).
   */
  function trace(field, w, h, level) {
    const { labels, count } = labelAreas(field, w, h, level);
    const value = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : field[y * w + x]);
    const label = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : labels[y * w + x]);
    const segments = [];
    const segmentAreas = [];
    for (let cy = -1; cy < h; cy++) {
      for (let cx = -1; cx < w; cx++) {
        const tl = value(cx, cy);
        const tr = value(cx + 1, cy);
        const br = value(cx + 1, cy + 1);
        const bl = value(cx, cy + 1);
        const c = (tl >= level ? 8 : 0) | (tr >= level ? 4 : 0) | (br >= level ? 2 : 0) | (bl >= level ? 1 : 0);
        if (c === 0 || c === 15) continue;
        let pairs = CASES[c];
        if ((c === 5 || c === 10) && (tl + tr + br + bl) / 4 >= level) pairs = SADDLE_CENTRE_INSIDE[c];
        // Where the level crosses each edge, and the label of that edge's inside corner. Heat values
        // are whole numbers, so crossing at level - 0.5 means the same inside/outside split but the line
        // never runs exactly through a sample (which would leave pieces meeting at a point).
        const iso = level - 0.5;
        const at = (a, b) => (iso - a) / (b - a);
        const point = (edge) => {
          switch (edge) {
            case 0: return [cx + at(tl, tr), cy, tl >= level ? label(cx, cy) : label(cx + 1, cy)];
            case 1: return [cx + 1, cy + at(tr, br), tr >= level ? label(cx + 1, cy) : label(cx + 1, cy + 1)];
            case 2: return [cx + at(bl, br), cy + 1, bl >= level ? label(cx, cy + 1) : label(cx + 1, cy + 1)];
            default: return [cx, cy + at(tl, bl), tl >= level ? label(cx, cy) : label(cx, cy + 1)];
          }
        };
        for (const [e1, e2] of pairs) {
          const p = point(e1);
          const q = point(e2);
          segments.push(p[0] + 0.5, p[1] + 0.5, q[0] + 0.5, q[1] + 0.5);
          segmentAreas.push(p[2]);
        }
      }
    }
    return { segments, segmentAreas, labels, count };
  }

  const api = { trace, labelAreas };
  root.outline = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof self !== 'undefined' ? self : globalThis);
