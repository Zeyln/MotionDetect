/*
 * Plain-JavaScript image operations for the detector's per-frame work.
 *
 * OpenCV.js runs these 10-20x slower in Firefox than in Chrome or Safari (its WebAssembly is slow
 * there), while plain JavaScript is fast in every browser. So the detector does its pixel work here
 * and only uses OpenCV for fitting the camera motion to a few hundred tracked points.
 *
 * Images are row-major typed arrays (Uint8Array or Float32Array) of width w and height h.
 * Works in a Web Worker (global `imgproc`) and in Node (module.exports).
 */
(function (root) {
  'use strict';

  // Grayscale with OpenCV's weights (0.299 R + 0.587 G + 0.114 B).
  function rgbaToGray(rgba, out) {
    for (let i = 0, j = 0; i < out.length; i++, j += 4) {
      out[i] = (77 * rgba[j] + 150 * rgba[j + 1] + 29 * rgba[j + 2] + 128) >> 8;
    }
  }

  // 5x5 Gaussian ([1 4 6 4 1] / 16 in each direction), edges clamped. `tmp` is a Float32Array of w*h.
  // Works in place (src === dst) and for any typed arrays; Uint8 output is rounded.
  function gaussian5(src, dst, w, h, tmp) {
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const a = src[row + (x < 2 ? 0 : x - 2)];
        const b = src[row + (x < 1 ? 0 : x - 1)];
        const c = src[row + x];
        const d = src[row + (x + 1 < w ? x + 1 : w - 1)];
        const e = src[row + (x + 2 < w ? x + 2 : w - 1)];
        tmp[row + x] = a + 4 * b + 6 * c + 4 * d + e;
      }
    }
    const round = dst instanceof Float32Array ? 0 : 0.5;
    for (let y = 0; y < h; y++) {
      const r0 = (y < 2 ? 0 : y - 2) * w;
      const r1 = (y < 1 ? 0 : y - 1) * w;
      const r2 = y * w;
      const r3 = (y + 1 < h ? y + 1 : h - 1) * w;
      const r4 = (y + 2 < h ? y + 2 : h - 1) * w;
      for (let x = 0; x < w; x++) {
        dst[r2 + x] = (tmp[r0 + x] + 4 * tmp[r1 + x] + 6 * tmp[r2 + x] + 4 * tmp[r3 + x] + tmp[r4 + x]) / 256 + round;
      }
    }
  }

  // Max over a k x k square (k odd), i.e. a dilation; outside the image is ignored. Works in place.
  // `tmp` is a Uint8Array of w*h. Large k uses van Herk / Gil-Werman (three passes whatever k is);
  // the vertical pass runs over whole rows at a time, which keeps memory access sequential.
  const scratch = new Map(); // per-size buffers for the vertical pass
  function dilate(src, dst, w, h, k, tmp) {
    if (k <= 1) {
      if (dst !== src) dst.set(src);
      return;
    }
    if (k === 3) {
      for (let y = 0; y < h; y++) {
        const row = y * w;
        if (w === 1) {
          tmp[row] = src[row];
          continue;
        }
        tmp[row] = src[row] > src[row + 1] ? src[row] : src[row + 1];
        for (let x = 1; x < w - 1; x++) {
          const a = src[row + x - 1], b = src[row + x], c = src[row + x + 1];
          const m = a > b ? a : b;
          tmp[row + x] = m > c ? m : c;
        }
        const e = row + w - 1;
        tmp[e] = src[e] > src[e - 1] ? src[e] : src[e - 1];
      }
      for (let y = 0; y < h; y++) {
        const row = y * w;
        const up = y > 0 ? row - w : row;
        const down = y < h - 1 ? row + w : row;
        for (let x = 0; x < w; x++) {
          const a = tmp[up + x], b = tmp[row + x], c = tmp[down + x];
          const m = a > b ? a : b;
          dst[row + x] = m > c ? m : c;
        }
      }
      return;
    }
    const r = k >> 1;
    // Rows.
    const mw = w + 2 * r;
    const g = new Uint8Array(mw);
    const hb = new Uint8Array(mw);
    const line = new Uint8Array(mw);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      line.fill(0);
      line.set(src.subarray(row, row + w), r);
      for (let i = 0, p = 0; i < mw; i++, p++) {
        if (p === k) p = 0;
        const v = line[i];
        g[i] = p === 0 || g[i - 1] < v ? v : g[i - 1];
      }
      hb[mw - 1] = line[mw - 1];
      for (let i = mw - 2; i >= 0; i--) {
        const v = line[i];
        hb[i] = (i + 1) % k === 0 || hb[i + 1] < v ? v : hb[i + 1];
      }
      for (let x = 0; x < w; x++) {
        const a = hb[x], b = g[x + 2 * r];
        tmp[row + x] = a > b ? a : b;
      }
    }
    // Columns, a whole row at a time: G and HB hold the prefix / suffix maxima per padded row.
    const mh = h + 2 * r;
    const key = mh * w;
    let buf = scratch.get(key);
    if (!buf) {
      buf = { G: new Uint8Array(mh * w), HB: new Uint8Array(mh * w) };
      scratch.set(key, buf);
    }
    const { G, HB } = buf;
    for (let i = 0, p = 0; i < mh; i++, p++) {
      if (p === k) p = 0;
      const j = i - r;
      const o = i * w;
      if (j < 0 || j >= h) {
        if (p === 0) G.fill(0, o, o + w);
        else G.copyWithin(o, o - w, o);
        continue;
      }
      const s0 = j * w;
      if (p === 0) {
        for (let x = 0; x < w; x++) G[o + x] = tmp[s0 + x];
      } else {
        for (let x = 0; x < w; x++) {
          const v = tmp[s0 + x], q = G[o - w + x];
          G[o + x] = q > v ? q : v;
        }
      }
    }
    for (let i = mh - 1; i >= 0; i--) {
      const j = i - r;
      const o = i * w;
      const blockEnd = (i + 1) % k === 0 || i === mh - 1;
      if (j < 0 || j >= h) {
        if (blockEnd) HB.fill(0, o, o + w);
        else HB.copyWithin(o, o + w, o + 2 * w);
        continue;
      }
      const s0 = j * w;
      if (blockEnd) {
        for (let x = 0; x < w; x++) HB[o + x] = tmp[s0 + x];
      } else {
        for (let x = 0; x < w; x++) {
          const v = tmp[s0 + x], q = HB[o + w + x];
          HB[o + x] = q > v ? q : v;
        }
      }
    }
    for (let y = 0; y < h; y++) {
      const a0 = y * w;
      const b0 = (y + 2 * r) * w;
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const a = HB[a0 + x], b = G[b0 + x];
        dst[row + x] = a > b ? a : b;
      }
    }
  }

  // Min over a k x k square (an erosion); outside the image is ignored. Done as an inverted dilation.
  function erode(src, dst, w, h, k, tmp) {
    for (let i = 0; i < dst.length; i++) dst[i] = 255 - src[i];
    dilate(dst, dst, w, h, k, tmp);
    for (let i = 0; i < dst.length; i++) dst[i] = 255 - dst[i];
  }

  // Mean over a k x k square (k odd), edges clamped. `tmp` is a Float32Array of w*h. Uint8 in/out.
  function boxBlur(src, dst, w, h, k, tmp) {
    const r = k >> 1;
    for (let y = 0; y < h; y++) {
      const row = y * w;
      let sum = 0;
      for (let i = -r; i <= r; i++) sum += src[row + Math.min(w - 1, Math.max(0, i))];
      for (let x = 0; x < w; x++) {
        tmp[row + x] = sum;
        sum += src[row + Math.min(w - 1, x + r + 1)] - src[row + Math.max(0, x - r)];
      }
    }
    const norm = 1 / (k * k);
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let i = -r; i <= r; i++) sum += tmp[Math.min(h - 1, Math.max(0, i)) * w + x];
      for (let y = 0; y < h; y++) {
        dst[y * w + x] = sum * norm + 0.5;
        sum += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
      }
    }
  }

  function invert3(m) {
    const [a, b, c, d, e, f, g, h, i] = m;
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-12) return null;
    const s = 1 / det;
    return [A * s, -(b * i - c * h) * s, (b * f - c * e) * s,
      B * s, (a * i - c * g) * s, -(a * f - c * d) * s,
      C * s, -(a * h - b * g) * s, (a * e - b * d) * s];
  }

  /**
   * Warps `src` by homography `m` (row-major 3x3, mapping src coordinates to dst) with bilinear
   * sampling, like OpenCV's warpPerspective. Pixels that come from outside `src` become 0, and get 0
   * in `validOut` (else 255) when it is given.
   */
  function warpPerspective(src, dst, w, h, m, validOut) {
    const mi = invert3(m);
    if (!mi) {
      dst.fill(0);
      if (validOut) validOut.fill(0);
      return;
    }
    const round = dst instanceof Float32Array ? 0 : 0.5;
    for (let y = 0; y < h; y++) {
      let X = mi[1] * y + mi[2];
      let Y = mi[4] * y + mi[5];
      let Z = mi[7] * y + mi[8];
      const row = y * w;
      for (let x = 0; x < w; x++, X += mi[0], Y += mi[3], Z += mi[6]) {
        const u = X / Z;
        const v = Y / Z;
        if (!(u >= 0 && v >= 0 && u <= w - 1 && v <= h - 1)) {
          dst[row + x] = 0;
          if (validOut) validOut[row + x] = 0;
          continue;
        }
        let x0 = u | 0;
        let y0 = v | 0;
        if (x0 >= w - 1) x0 = w - 2;
        if (y0 >= h - 1) y0 = h - 2;
        const fx = u - x0;
        const fy = v - y0;
        const i = y0 * w + x0;
        const top = src[i] + (src[i + 1] - src[i]) * fx;
        const bottom = src[i + w] + (src[i + w + 1] - src[i + w]) * fx;
        dst[row + x] = top + (bottom - top) * fy + round;
        if (validOut) validOut[row + x] = 255;
      }
    }
  }

  // Averages s x s blocks of a Float32 image into a Uint8 image of ceil(w/s) x ceil(h/s).
  function shrink(src, w, h, s, out, ow, oh) {
    for (let oy = 0; oy < oh; oy++) {
      const y1 = Math.min(h, (oy + 1) * s);
      for (let ox = 0; ox < ow; ox++) {
        const x1 = Math.min(w, (ox + 1) * s);
        let sum = 0;
        let n = 0;
        for (let y = oy * s; y < y1; y++) {
          for (let x = ox * s, i = y * w + x; x < x1; x++, i++) sum += src[i];
          n += x1 - ox * s;
        }
        const v = sum / n;
        out[oy * ow + ox] = v > 255 ? 255 : v + 0.5;
      }
    }
  }

  // ---------- Pyramids and Lucas-Kanade tracking ----------

  /**
   * Builds an image pyramid from a Uint8 frame: level 0 is the frame itself, each further level is
   * blurred and halved. Every level carries its central-difference gradients for tracking.
   * `pyr` (from a previous call at the same size) is reused when given.
   */
  function buildPyramid(gray, w, h, levels, pyr) {
    if (!pyr || pyr.w !== w || pyr.h !== h || pyr.levels.length !== levels) {
      pyr = { w, h, levels: [] };
      let lw = w;
      let lh = h;
      for (let l = 0; l < levels; l++) {
        pyr.levels.push({
          w: lw, h: lh,
          img: new Float32Array(lw * lh), gx: new Float32Array(lw * lh), gy: new Float32Array(lw * lh),
          tmp: new Float32Array(lw * lh), blur: new Float32Array(lw * lh),
        });
        lw = Math.max(1, lw >> 1);
        lh = Math.max(1, lh >> 1);
      }
    }
    const L = pyr.levels;
    for (let i = 0; i < gray.length; i++) L[0].img[i] = gray[i];
    for (let l = 1; l < L.length; l++) {
      const up = L[l - 1];
      const lv = L[l];
      gaussian5(up.img, up.blur, up.w, up.h, up.tmp);
      for (let y = 0; y < lv.h; y++) {
        for (let x = 0; x < lv.w; x++) lv.img[y * lv.w + x] = up.blur[2 * y * up.w + 2 * x];
      }
    }
    for (const lv of L) gradients(lv.img, lv.gx, lv.gy, lv.w, lv.h);
    return pyr;
  }

  function gradients(img, gx, gy, w, h) {
    for (let y = 0; y < h; y++) {
      const up = (y > 0 ? y - 1 : 0) * w;
      const down = (y < h - 1 ? y + 1 : h - 1) * w;
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const l = x > 0 ? x - 1 : 0;
        const r = x < w - 1 ? x + 1 : w - 1;
        gx[row + x] = (img[row + r] - img[row + l]) * 0.5;
        gy[row + x] = (img[down + x] - img[up + x]) * 0.5;
      }
    }
  }

  /**
   * Picks well-spread corners (Shi-Tomasi: smallest eigenvalue of the local gradient structure) on
   * pyramid level 1 for speed, at most one per grid cell, about `maxCorners` in total. Returns
   * Float32Array [x0, y0, x1, y1, ...] in level-0 coordinates.
   */
  function findCorners(pyr, maxCorners) {
    const lvl = pyr.levels.length > 1 ? 1 : 0;
    const { w, h, gx, gy } = pyr.levels[lvl];
    const f = 1 << lvl;
    const cell = Math.max(4, Math.round(Math.sqrt((w * h) / maxCorners)));
    const cols = Math.ceil(w / cell);
    const rows = Math.ceil(h / cell);
    const best = new Float32Array(cols * rows);
    const bestX = new Int32Array(cols * rows);
    const bestY = new Int32Array(cols * rows);
    let top = 0;
    const border = 3;
    for (let y = border; y < h - border; y++) {
      for (let x = border; x < w - border; x++) {
        let a = 0, b = 0, c = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1, i = (y + dy) * w + x - 1; dx <= 1; dx++, i++) {
            const ix = gx[i];
            const iy = gy[i];
            a += ix * ix;
            b += ix * iy;
            c += iy * iy;
          }
        }
        const score = (a + c) / 2 - Math.sqrt(((a - c) / 2) ** 2 + b * b);
        const ci = ((y / cell) | 0) * cols + ((x / cell) | 0);
        if (score > best[ci]) {
          best[ci] = score;
          bestX[ci] = x;
          bestY[ci] = y;
          if (score > top) top = score;
        }
      }
    }
    // Like OpenCV's qualityLevel 0.01, plus a floor so flat noise isn't picked.
    const minScore = Math.max(0.01 * top, 25);
    const out = [];
    for (let i = 0; i < best.length; i++) if (best[i] >= minScore) out.push(bestX[i] * f, bestY[i] * f);
    return new Float32Array(out);
  }

  function sample(img, w, x, y) {
    const x0 = x | 0;
    const y0 = y | 0;
    const fx = x - x0;
    const fy = y - y0;
    const i = y0 * w + x0;
    const top = img[i] + (img[i + 1] - img[i]) * fx;
    const bottom = img[i + w] + (img[i + w + 1] - img[i + w]) * fx;
    return top + (bottom - top) * fy;
  }

  /**
   * Pyramidal Lucas-Kanade (Bouguet): tracks each point in `pts` from pyramid `from` into pyramid
   * `to`. Writes positions into `out` and 1/0 into `status`. Window is (2*radius+1)^2.
   */
  function trackPoints(from, to, pts, out, status, radius = 7, maxIter = 10, eps = 0.03) {
    const n = pts.length / 2;
    const side = 2 * radius + 1;
    const N = side * side;
    const patch = new Float32Array(N);
    const pgx = new Float32Array(N);
    const pgy = new Float32Array(N);
    const levels = from.levels.length;
    for (let p = 0; p < n; p++) {
      let gxs = 0; // guess carried down the pyramid (level coordinates)
      let gys = 0;
      let ok = true;
      for (let l = levels - 1; l >= 0 && ok; l--) {
        const A = from.levels[l];
        const B = to.levels[l];
        const s = 1 / (1 << l);
        const px = pts[2 * p] * s;
        const py = pts[2 * p + 1] * s;
        if (px - radius < 0 || py - radius < 0 || px + radius >= A.w - 1 || py + radius >= A.h - 1) {
          if (l === 0) ok = false;
          gxs *= 2;
          gys *= 2;
          continue; // too close to the border at this level; refine on finer ones
        }
        let gxx = 0, gxy = 0, gyy = 0;
        for (let j = -radius, k = 0; j <= radius; j++) {
          for (let i = -radius; i <= radius; i++, k++) {
            const ix = sample(A.gx, A.w, px + i, py + j);
            const iy = sample(A.gy, A.w, px + i, py + j);
            patch[k] = sample(A.img, A.w, px + i, py + j);
            pgx[k] = ix;
            pgy[k] = iy;
            gxx += ix * ix;
            gxy += ix * iy;
            gyy += iy * iy;
          }
        }
        const det = gxx * gyy - gxy * gxy;
        const minEig = (gxx + gyy) / 2 - Math.sqrt(((gxx - gyy) / 2) ** 2 + gxy * gxy);
        if (det < 1e-6 || minEig < 0.5 * N) {
          ok = false;
          break;
        }
        let vx = 0;
        let vy = 0;
        for (let it = 0; it < maxIter; it++) {
          const qx = px + gxs + vx;
          const qy = py + gys + vy;
          if (qx - radius < 0 || qy - radius < 0 || qx + radius >= B.w - 1 || qy + radius >= B.h - 1) {
            ok = false;
            break;
          }
          let bx = 0, by = 0;
          for (let j = -radius, k = 0; j <= radius; j++) {
            for (let i = -radius; i <= radius; i++, k++) {
              const d = patch[k] - sample(B.img, B.w, qx + i, qy + j);
              bx += d * pgx[k];
              by += d * pgy[k];
            }
          }
          const ex = (gyy * bx - gxy * by) / det;
          const ey = (gxx * by - gxy * bx) / det;
          vx += ex;
          vy += ey;
          if (ex * ex + ey * ey < eps * eps) break;
        }
        gxs += vx;
        gys += vy;
        if (l > 0) {
          gxs *= 2;
          gys *= 2;
        }
      }
      status[p] = ok ? 1 : 0;
      out[2 * p] = pts[2 * p] + gxs;
      out[2 * p + 1] = pts[2 * p + 1] + gys;
    }
  }

  const api = {
    rgbaToGray, gaussian5, dilate, erode, boxBlur, invert3, warpPerspective, shrink,
    buildPyramid, findCorners, trackPoints,
  };
  root.imgproc = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof self !== 'undefined' ? self : globalThis);
