/*
 * Motion detector that compensates for camera movement.
 *
 * Per frame:
 *   1. Track corner features from the previous frame into the current one (pyramidal Lucas-Kanade),
 *      then back again, keeping only points that return to where they started.
 *   2. Fit a homography to those matches with RANSAC (OpenCV). Background points dominate, so this
 *      is the camera's own motion; points on moving objects get rejected as outliers.
 *   3. Warp the previous two frames onto the current one, match their brightness (auto exposure),
 *      and compare each pixel with its 3x3 neighbourhood in each, which tolerates 1 px of
 *      misalignment. Changes since the previous frame outline each blob; a blob is only kept if
 *      some of its pixels also differ from the frame before last. The spot an object just left
 *      matches that frame, so no ghost is left behind, and a glitch in one frame is ignored.
 *   4. Threshold and clean up with morphology. Nearby blobs are grouped using a dilated copy of
 *      the mask, but each box is fitted to the confirmed pixels so it hugs the object.
 *   5. Feed the boxes to a tracker. A target is reported once it has been seen in minHits frames.
 *      When it stops moving it is kept as a "held" box for holdMs, carried along with the camera
 *      motion so it stays on the target while the phone moves.
 *
 * Optionally (opts.heatmap) it also keeps a motion heatmap: confirmed motion pixels of confirmed
 * targets are added as heat, which fades over heatFadeMs and is carried along with the camera motion
 * so it stays where the motion happened. It is returned at a quarter of the frame size.
 *
 * All pixel work is plain JavaScript (imgproc.js), which is fast in every browser; OpenCV.js is only
 * used for the RANSAC homography fit, because its other operations are 10-20x slower in Firefox.
 *
 * Kernel sizes, feature spacing and RANSAC tolerance are tuned for a 320 px frame and scale with
 * the frame size. The noise filter stays 3x3 at every size so small, thin movers survive.
 *
 * Works in a Web Worker (globals `createDetector`, needs `imgproc`) and in Node (module.exports).
 */
(function (root) {
  'use strict';

  const ip = root.imgproc || (typeof require === 'function' ? require('./imgproc.js') : null);

  const DEFAULTS = {
    threshold: 30,        // gray-level difference (0-255) that counts as change
    minArea: 0.0001,      // smallest motion box to report, as a fraction of the frame area
    maxChangedArea: 0.4,  // if more than this fraction changed, motion estimate is bad: report nothing
    minHits: 3,           // frames a target must be seen in before it is reported
    minHoldHits: 5,       // frames a target must be seen in before it is held after stopping
    holdMs: 3000,         // how long a target stays boxed after it stops moving
    matchDistance: 0.06,  // extra distance (fraction of frame size) a target may jump between frames
    maxCoastFrames: 2,    // frames a very fast target that went unseen is followed (hidden) along its path
    maxTracks: 30,
    maxCorners: 200,      // feature points used to estimate camera motion (one per grid cell)
    minFeatures: 12,      // fewer tracked points than this: camera motion can't be trusted
    minSpreadCells: 5,    // camera-motion inliers must cover this many cells of a 4x4 grid
    maxRoundTrip: 1,      // px at 320 px: forward-backward tracking error allowed per point
    heatmap: false,       // also build and return a motion heatmap
    heatFadeMs: 1000,     // how long heat takes to fade away (it halves every quarter of this)
  };

  const odd = (n) => Math.max(3, 2 * Math.floor(n / 2) + 1);
  const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

  // Product of two row-major 3x3 matrices: applying b, then a.
  function multiply(a, b) {
    const r = new Array(9);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        r[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
    return r;
  }

  function createDetector(cv, options) {
    const opts = Object.assign({}, DEFAULTS, options);

    let width = 0;
    let height = 0;
    let started = false;
    let frameCount = 0; // frames in a row with a known camera motion (0 after a reset or loss)
    let lastMotion = null; // camera motion prev2 -> prev, as a row-major 3x3 array
    // { x, y, w, h, vx, vy, hits, lastSeen, held, coast: frames coasted, cx/cy: offset coasted, id }
    let tracks = [];
    let nextId = 1;
    // Frames and masks (Uint8Array unless noted), allocated per frame size.
    let gray, prevGray, prev2Gray, warped, warpedMin, warpedMax, diff, motion, motion2, groups, warpedValid;
    let u8tmp, f32tmp;
    let labels; // Int32Array, connected-component labels of `groups`
    let heat, heatTmp, heatSrc, heatSmall; // heat and heatTmp are Float32Array
    let pyrPrev = null; // pyramid of prevGray
    let pyrCur = null; // pyramid of gray
    let lastHeatTime = null;
    let scale = 1; // frame size relative to the 320 px the defaults are tuned for
    let pyramidLevels, growSize, heatBlurSize;

    function allocate(w, h) {
      width = w;
      height = h;
      const n = w * h;
      scale = Math.max(w, h) / 320;
      pyramidLevels = Math.round(3 + Math.log2(Math.max(1, scale)));
      growSize = odd(7 * scale);
      heatBlurSize = odd(7 * scale);
      [gray, prevGray, prev2Gray, warped, warpedMin, warpedMax, diff, motion, motion2, groups, warpedValid, u8tmp, heatSrc] =
        Array.from({ length: 13 }, () => new Uint8Array(n));
      f32tmp = new Float32Array(n);
      labels = new Int32Array(n);
      heat = new Float32Array(n);
      heatTmp = new Float32Array(n);
      heatSmall = null;
      pyrPrev = pyrCur = null;
      reset();
    }

    function reset() {
      started = false;
      frameCount = 0;
      lastMotion = null;
      tracks = [];
      lastHeatTime = null;
      if (heat) heat.fill(0);
    }

    // Returns { m, features, inliers }: m is the camera motion prev -> current as a row-major 3x3
    // array (IDENTITY when it can't be told apart from a still camera), or null if tracking failed.
    function estimateCameraMotion() {
      const corners = ip.findCorners(pyrPrev, opts.maxCorners);
      const count = corners.length / 2;
      // Too little texture to track (blank wall, darkness). Assume a still camera.
      if (count < opts.minFeatures) return { m: IDENTITY, features: count, inliers: 0 };

      const next = new Float32Array(corners.length);
      const back = new Float32Array(corners.length);
      const status = new Uint8Array(count);
      const backStatus = new Uint8Array(count);
      ip.trackPoints(pyrPrev, pyrCur, corners, next, status);
      ip.trackPoints(pyrCur, pyrPrev, next, back, backStatus);

      const maxRoundTrip = opts.maxRoundTrip * scale;
      const src = [];
      const dst = [];
      for (let i = 0; i < count; i++) {
        if (status[i] !== 1 || backStatus[i] !== 1) continue;
        const dx = back[2 * i] - corners[2 * i];
        const dy = back[2 * i + 1] - corners[2 * i + 1];
        if (dx * dx + dy * dy > maxRoundTrip * maxRoundTrip) continue;
        src.push(corners[2 * i], corners[2 * i + 1]);
        dst.push(next[2 * i], next[2 * i + 1]);
      }
      const tracked = src.length / 2;
      if (tracked < opts.minFeatures) return { m: null, features: tracked, inliers: 0 };

      const srcMat = cv.matFromArray(tracked, 1, cv.CV_32FC2, src);
      const dstMat = cv.matFromArray(tracked, 1, cv.CV_32FC2, dst);
      const inlierMask = new cv.Mat();
      const H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 2 * scale, inlierMask, 2000, 0.995);
      let inliers = 0;
      let cells = 0;
      let m = null;
      if (!H.empty()) {
        ({ inliers, cells } = inlierSpread(inlierMask, dst));
        m = Array.from(H.data64F);
      }
      srcMat.delete();
      dstMat.delete();
      inlierMask.delete();
      H.delete();

      // If the agreeing points are few or bunched together, they are probably on one moving
      // object over a plain background, and "compensating" would erase that object.
      // Assume a still camera instead; the changed-area guard catches it if that's wrong.
      if (!m || inliers < opts.minFeatures || cells < opts.minSpreadCells) {
        return { m: IDENTITY, features: tracked, inliers };
      }
      if (!plausible(m)) return { m: null, features: tracked, inliers };
      return { m, features: tracked, inliers };
    }

    // Counts RANSAC inliers and how many cells of a 4x4 grid over the frame they land in.
    function inlierSpread(inlierMask, dst) {
      const occupied = new Set();
      let inliers = 0;
      for (let i = 0; i < inlierMask.rows; i++) {
        if (!inlierMask.data[i]) continue;
        inliers++;
        const gx = Math.min(3, Math.floor((4 * dst[2 * i]) / width));
        const gy = Math.min(3, Math.floor((4 * dst[2 * i + 1]) / height));
        occupied.add(gy * 4 + gx);
      }
      return { inliers, cells: occupied.size };
    }

    // Reject homographies that would fling the frame corners further than a phone can move in one frame.
    function plausible(m) {
      const limit = 0.35 * Math.max(width, height);
      for (const [x, y] of [[0, 0], [width, 0], [0, height], [width, height]]) {
        const d = m[6] * x + m[7] * y + m[8];
        if (Math.abs(d) < 1e-6) return false;
        const u = (m[0] * x + m[1] * y + m[2]) / d;
        const v = (m[3] * x + m[4] * y + m[5]) / d;
        if (Math.hypot(u - x, v - y) > limit) return false;
      }
      return true;
    }

    /**
     * Connected groups (8-connected) of the dilated mask become candidate boxes. Each box is fitted to
     * the motion pixels of its group; with `confirm`, only confirmed (two-step) pixels count, so a
     * ghost that the grouping step joined onto its object doesn't stretch the box, and groups without
     * confirmed pixels are dropped.
     */
    function findBoxes(confirm) {
      const n = labelGroups();
      const minX = new Int32Array(n + 1).fill(width);
      const minY = new Int32Array(n + 1).fill(height);
      const maxX = new Int32Array(n + 1).fill(-1);
      const maxY = new Int32Array(n + 1).fill(-1);
      const count = new Int32Array(n + 1);
      const px = confirm ? motion2 : motion;
      for (let y = 0, i = 0; y < height; y++) {
        for (let x = 0; x < width; x++, i++) {
          const l = labels[i];
          if (!l || !px[i]) continue;
          count[l]++;
          if (x < minX[l]) minX[l] = x;
          if (x > maxX[l]) maxX[l] = x;
          if (y < minY[l]) minY[l] = y;
          if (y > maxY[l]) maxY[l] = y;
        }
      }
      const minPx = opts.minArea * width * height;
      const boxes = [];
      for (let l = 1; l <= n; l++) {
        if (maxX[l] < 0) continue;
        const r = { x: minX[l], y: minY[l], w: maxX[l] - minX[l] + 1, h: maxY[l] - minY[l] + 1, confirmed: count[l] };
        if (r.w * r.h >= minPx && (!confirm || r.confirmed >= 2)) boxes.push(r);
      }
      return mergeBoxes(boxes, 0);
    }

    // Two-pass connected-component labelling of `groups` into `labels`; returns the label count.
    function labelGroups() {
      const parent = [0];
      const find = (a) => {
        while (parent[a] !== a) a = parent[a] = parent[parent[a]];
        return a;
      };
      const union = (a, b) => {
        a = find(a);
        b = find(b);
        if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
      };
      const w = width;
      for (let y = 0, i = 0; y < height; y++) {
        for (let x = 0; x < w; x++, i++) {
          if (!groups[i]) {
            labels[i] = 0;
            continue;
          }
          const left = x > 0 ? labels[i - 1] : 0;
          const upLeft = x > 0 && y > 0 ? labels[i - w - 1] : 0;
          const up = y > 0 ? labels[i - w] : 0;
          const upRight = x < w - 1 && y > 0 ? labels[i - w + 1] : 0;
          let l = left || upLeft || up || upRight;
          if (!l) {
            l = parent.length;
            parent.push(l);
          } else {
            if (left) union(l, left);
            if (upLeft) union(l, upLeft);
            if (up) union(l, up);
            if (upRight) union(l, upRight);
          }
          labels[i] = l;
        }
      }
      const compact = new Int32Array(parent.length);
      let n = 0;
      for (let l = 1; l < parent.length; l++) {
        const r = find(l);
        if (!compact[r]) compact[r] = ++n;
        compact[l] = compact[r];
      }
      for (let i = 0; i < labels.length; i++) if (labels[i]) labels[i] = compact[labels[i]];
      return n;
    }

    // ---------- Target tracking ----------

    // Moves every track by the camera motion m (row-major 3x3 homography) so boxes stay on their
    // targets while the phone moves. Tracks that leave the frame are dropped.
    function moveTracks(m) {
      for (const t of tracks) {
        const xs = [];
        const ys = [];
        for (const [x, y] of [[t.x, t.y], [t.x + t.w, t.y], [t.x, t.y + t.h], [t.x + t.w, t.y + t.h]]) {
          const d = m[6] * x + m[7] * y + m[8];
          xs.push((m[0] * x + m[1] * y + m[2]) / d);
          ys.push((m[3] * x + m[4] * y + m[5]) / d);
        }
        t.x = Math.min(...xs);
        t.y = Math.min(...ys);
        t.w = Math.max(...xs) - t.x;
        t.h = Math.max(...ys) - t.y;
      }
      tracks = tracks.filter((t) => t.x + t.w > 0 && t.y + t.h > 0 && t.x < width && t.y < height);
    }

    // Matches this frame's boxes to existing targets by predicted position, nearest pairs first.
    function updateTracks(detections, now) {
      const reach = opts.matchDistance * Math.max(width, height);
      detections = absorbIntoHeld(detections);
      const pairs = [];
      tracks.forEach((t, ti) => {
        const px = t.x + t.w / 2 + t.vx;
        const py = t.y + t.h / 2 + t.vy;
        detections.forEach((d, di) => {
          const dist = Math.hypot(d.x + d.w / 2 - px, d.y + d.h / 2 - py);
          const gate = (Math.hypot(t.w, t.h) + Math.hypot(d.w, d.h)) / 2 + reach;
          if (dist < gate) pairs.push({ ti, di, dist });
        });
      });
      pairs.sort((a, b) => a.dist - b.dist);

      const matchedTracks = new Set();
      const matchedDetections = new Set();
      for (const { ti, di } of pairs) {
        if (matchedTracks.has(ti) || matchedDetections.has(di)) continue;
        matchedTracks.add(ti);
        matchedDetections.add(di);
        const t = tracks[ti];
        const d = detections[di];
        // Per-frame movement since it was last seen (undoing any coasting along the prediction).
        const steps = t.coast + 1;
        t.vx = 0.5 * t.vx + 0.5 * ((d.x + d.w / 2 - (t.x + t.w / 2) + t.cx) / steps);
        t.vy = 0.5 * t.vy + 0.5 * ((d.y + d.h / 2 - (t.y + t.h / 2) + t.cy) / steps);
        Object.assign(t, { x: d.x, y: d.y, w: d.w, h: d.h, hits: t.hits + 1, lastSeen: now, held: false });
        t.coast = t.cx = t.cy = 0;
      }

      const slowSpeed = Math.max(1, scale); // px per frame
      tracks = tracks.filter((t, ti) => {
        if (matchedTracks.has(ti)) return true;
        if (t.held) return true;
        // Unseen this frame. A target that was moving against the background may just have been
        // missed (it blended in, or the frame was blurred). Follow its slowing path for a couple of
        // frames (see reportTracks for when it is shown) so the same target picks it up again instead
        // of leaving a held box behind and starting a new one. If it isn't found, it is held where it
        // was last seen, as if it had stopped there.
        const speed = Math.hypot(t.vx, t.vy);
        if (t.coast < opts.maxCoastFrames && speed > Math.max(slowSpeed, 0.25 * Math.max(t.w, t.h))) {
          t.vx *= 0.7;
          t.vy *= 0.7;
          t.x += t.vx;
          t.y += t.vy;
          t.cx += t.vx;
          t.cy += t.vy;
          t.coast++;
          return t.hits >= opts.minHits;
        }
        // Its path took it to the edge of the frame (within one box size): it has left, so there is
        // nothing to hold.
        if (t.coast && (t.x < t.w || t.y < t.h || t.x + 2 * t.w > width || t.y + 2 * t.h > height)) return false;
        // It stopped (or wasn't found again): hold it where it was last seen.
        t.x -= t.cx;
        t.y -= t.cy;
        t.coast = t.cx = t.cy = 0;
        t.vx = t.vy = 0;
        t.held = true;
        // Only hold targets followed for a while; short-lived fragments and flickers just vanish.
        return t.hits >= opts.minHoldHits;
      });
      detections.forEach((d, di) => {
        if (!matchedDetections.has(di)) {
          tracks.push({ ...d, vx: 0, vy: 0, hits: 1, lastSeen: now, held: false, coast: 0, cx: 0, cy: 0, id: nextId++ });
        }
      });
      mergeOverlappingTracks(now);
    }

    // A stopped target still flickers with small differences on its edges, because no camera-motion
    // estimate lines frames up perfectly. Detections well inside a held box, and clearly smaller than
    // it, are swallowed so they neither shrink the box nor spawn new targets; the hold timer keeps
    // running. A detection the size of the target, or reaching outside the box (the target moving
    // off again), still matches normally.
    function absorbIntoHeld(detections) {
      const margin = 0.02 * Math.max(width, height);
      const held = tracks.filter((t) => t.held);
      if (!held.length) return detections;
      return detections.filter((d) => !held.some((t) =>
        d.w * d.h < 0.7 * t.w * t.h &&
        d.x >= t.x - margin && d.y >= t.y - margin &&
        d.x + d.w <= t.x + t.w + margin && d.y + d.h <= t.y + t.h + margin));
    }

    // A target can break into fragments and rejoin. Overlapping tracks collapse into the one seen
    // most often; if both moved this frame their boxes are combined.
    function mergeOverlappingTracks(now) {
      tracks.sort((a, b) => b.hits - a.hits);
      const kept = [];
      for (const t of tracks) {
        const owner = kept.find((k) => overlaps(k, t, 0));
        if (!owner) {
          kept.push(t);
        } else if (owner.lastSeen === now && t.lastSeen === now) {
          const x = Math.min(owner.x, t.x);
          const y = Math.min(owner.y, t.y);
          owner.w = Math.max(owner.x + owner.w, t.x + t.w) - x;
          owner.h = Math.max(owner.y + owner.h, t.y + t.h) - y;
          owner.x = x;
          owner.y = y;
        }
      }
      tracks = kept;
    }

    function expireTracks(now) {
      tracks = tracks.filter((t) => now - t.lastSeen <= opts.holdMs);
      if (tracks.length > opts.maxTracks) {
        tracks.sort((a, b) => b.lastSeen - a.lastSeen);
        tracks.length = opts.maxTracks;
      }
    }

    function reportTracks(now) {
      return tracks
        // A coasting target is shown at its predicted position only while that is less than its own
        // size away from where it was last seen (so the box still overlaps it, e.g. a bird you are
        // following with the camera). Further than that is too much of a guess: nothing is shown
        // until it is seen again or held.
        .filter((t) => t.hits >= opts.minHits && (!t.coast || Math.hypot(t.cx, t.cy) < Math.max(t.w, t.h)))
        .map((t) => {
          const x = Math.max(0, Math.round(t.x));
          const y = Math.max(0, Math.round(t.y));
          return {
            x,
            y,
            w: Math.min(width, Math.round(t.x + t.w)) - x,
            h: Math.min(height, Math.round(t.y + t.h)) - y,
            held: t.held,
            id: t.id,
          };
        });
    }

    /**
     * Process one RGBA frame captured at time `now` (ms). Returns { boxes, heat, mode, features, inliers }.
     * Boxes are in frame pixels; `held` marks targets that have stopped moving.
     * mode: 'init' (first frame), 'ok', 'lost' (camera moved too fast to track), 'noisy' (too much changed).
     */
    function processFrame(data, w, h, now = Date.now()) {
      if (w !== width || h !== height) allocate(w, h);
      ip.rgbaToGray(data, gray);
      ip.gaussian5(gray, gray, width, height, f32tmp);
      pyrCur = ip.buildPyramid(gray, width, height, pyramidLevels, pyrSpare);

      if (!started) {
        started = true;
        advance(null);
        return { boxes: [], heat: null, mode: 'init', features: 0, inliers: 0 };
      }

      const { m: m1, features, inliers } = estimateCameraMotion();
      expireTracks(now);
      if (!m1) {
        // Can't tell where the camera went; leave held boxes where they are until it settles.
        advance(null);
        return { boxes: reportTracks(now), heat: updateHeat(null, null, false, now), mode: 'lost', features, inliers };
      }
      if (m1 !== IDENTITY) moveTracks(m1);

      const n = width * height;
      const thr = opts.threshold;
      // What changed since the previous frame gives each blob's full extent...
      changedSince(prevGray, m1, warpedValid, diff);
      for (let i = 0; i < n; i++) motion[i] = diff[i] > thr && warpedValid[i] ? 255 : 0;
      ip.erode(motion, motion, width, height, 3, u8tmp); // opening: drop specks smaller than 3x3
      ip.dilate(motion, motion, width, height, 3, u8tmp);
      ip.dilate(motion, groups, width, height, 2 * growSize - 1, u8tmp);
      // ...and pixels that also changed since the frame before (camera motion composed over both
      // steps) confirm it. A spot an object has just left, or a glitch in the previous frame, matches
      // the frame before last, so its blob has no confirmed pixels and is dropped.
      const twoStep = frameCount >= 1 && lastMotion !== null;
      if (twoStep) {
        // Its result is only used where the first comparison found motion, so it reuses that one's
        // valid-pixel mask instead of computing its own.
        changedSince(prev2Gray, multiply(m1, lastMotion), null, diff);
        for (let i = 0; i < n; i++) motion2[i] = diff[i] > thr && motion[i] ? 255 : 0;
      }

      advance(m1);

      let changed = 0;
      for (let i = 0; i < n; i++) if (groups[i]) changed++;
      if (changed > opts.maxChangedArea * n) {
        return { boxes: reportTracks(now), heat: updateHeat(m1, null, false, now), mode: 'noisy', features, inliers };
      }
      updateTracks(findBoxes(twoStep), now);
      // Heat comes only from confirmed targets seen this frame, so it gets the tracker's noise
      // filtering: blips that never reach minHits sightings add no heat.
      const seen = tracks.filter((t) => t.hits >= opts.minHits && t.lastSeen === now);
      const heatMap = updateHeat(m1, seen, twoStep, now);
      return { boxes: reportTracks(now), heat: heatMap, mode: 'ok', features, inliers };
    }

    /**
     * Fades the heatmap, moves it with camera motion `m` (if known), and adds the motion pixels inside
     * `targets` (boxes of confirmed targets seen this frame; two-step pixels when `twoStep`). Returns it
     * at a quarter of the frame size as { w, h, data: Uint8Array }, or null if the heatmap is off.
     */
    function updateHeat(m, targets, twoStep, now) {
      if (!opts.heatmap) return null;
      const dt = lastHeatTime === null ? 0 : now - lastHeatTime;
      lastHeatTime = now;
      const n = width * height;

      if (m && m !== IDENTITY) {
        ip.warpPerspective(heat, heatTmp, width, height, m, null);
        [heat, heatTmp] = [heatTmp, heat];
      }
      const decay = opts.heatFadeMs > 0 ? Math.pow(0.5, dt / (opts.heatFadeMs / 4)) : 0;
      for (let i = 0; i < n; i++) heat[i] *= decay;

      if (targets && targets.length) {
        const source = twoStep ? motion2 : motion;
        heatSrc.fill(0);
        for (const t of targets) {
          const x0 = Math.max(0, Math.round(t.x));
          const y0 = Math.max(0, Math.round(t.y));
          const x1 = Math.min(width, Math.round(t.x + t.w));
          const y1 = Math.min(height, Math.round(t.y + t.h));
          for (let y = y0; y < y1; y++) {
            for (let x = x0, i = y * width + x0; x < x1; x++, i++) heatSrc[i] = source[i];
          }
        }
        // Fill in sparse edge pixels and soften the blob into a glow.
        ip.dilate(heatSrc, heatSrc, width, height, growSize, u8tmp);
        ip.boxBlur(heatSrc, heatSrc, width, height, heatBlurSize, f32tmp);
        for (let i = 0; i < n; i++) if (heatSrc[i] > heat[i]) heat[i] = heatSrc[i];
      }

      // A fresh array every time: it is handed (transferred) to the page.
      const sw = Math.ceil(width / 4);
      const sh = Math.ceil(height / 4);
      const small = new Uint8Array(sw * sh);
      ip.shrink(heat, width, height, 4, small, sw, sh);
      return { w: sw, h: sh, data: small };
    }

    // Shifts the frame history: current -> prev -> prev2. `m` is the camera motion prev -> current,
    // or null if it is unknown (first frame, lost tracking), which restarts the history.
    let pyrSpare = null;
    function advance(m) {
      if (m) {
        prev2Gray.set(prevGray);
        lastMotion = m;
        frameCount++;
      } else {
        lastMotion = null;
        frameCount = 0;
      }
      prevGray.set(gray);
      // The current pyramid becomes the previous one; the old previous one is reused next frame.
      pyrSpare = pyrPrev;
      pyrPrev = pyrCur;
      pyrCur = null;
    }

    /**
     * Writes into `out` how much each pixel of the current frame differs from `ref` once `ref` is
     * warped by camera motion `m` and brightness-matched. If `validOut` is given it receives the pixels
     * that have a comparison; otherwise the existing `warpedValid` is used for brightness matching.
     * A pixel only counts as changed if it falls outside the brightness range of its 3x3 neighbourhood
     * in the warped frame. This forgives up to 1 px of misalignment, which otherwise flickers on every
     * sharp edge; real movers shift more than that against their surroundings. Kept at 3x3 at every
     * size so small, slow movers still register.
     */
    function changedSince(ref, m, validOut, out) {
      ip.warpPerspective(ref, warped, width, height, m, validOut);
      // Pixels near where the reference frame ran out are unreliable too.
      if (validOut) ip.erode(validOut, validOut, width, height, growSize, u8tmp);
      const gain = brightnessGain(validOut || warpedValid);
      ip.erode(warped, warpedMin, width, height, 3, u8tmp);
      ip.dilate(warped, warpedMax, width, height, 3, u8tmp);
      for (let i = 0; i < out.length; i++) {
        const g = gray[i];
        const lo = warpedMin[i] * gain;
        const hi = warpedMax[i] * gain;
        const d = g > hi ? g - hi : lo > g ? lo - g : 0;
        out[i] = d > 255 ? 255 : d;
      }
    }

    // Phones adjust exposure as the scene changes, brightening or darkening the whole frame.
    // Returns the factor that matches the warped reference frame's average brightness to the current one.
    function brightnessGain(mask) {
      let before = 0;
      let current = 0;
      let count = 0;
      for (let i = 0; i < mask.length; i++) {
        if (!mask[i]) continue;
        before += warped[i];
        current += gray[i];
        count++;
      }
      if (!count || before < count) return 1;
      const gain = current / before;
      return Math.abs(gain - 1) > 0.01 ? gain : 1;
    }

    function configure(next) {
      Object.assign(opts, next);
    }

    function dispose() {
      // Everything is plain typed arrays now; nothing to free.
    }

    return { processFrame, configure, reset, dispose, options: opts };
  }

  function overlaps(a, b, gap) {
    return a.x - gap < b.x + b.w && b.x - gap < a.x + a.w && a.y - gap < b.y + b.h && b.y - gap < a.y + a.h;
  }

  // Repeatedly merge boxes that overlap or sit within `gap` pixels of each other.
  function mergeBoxes(boxes, gap) {
    let merged = true;
    while (merged) {
      merged = false;
      for (let i = 0; i < boxes.length && !merged; i++) {
        for (let j = i + 1; j < boxes.length; j++) {
          const a = boxes[i];
          const b = boxes[j];
          if (!overlaps(a, b, gap)) continue;
          const x = Math.min(a.x, b.x);
          const y = Math.min(a.y, b.y);
          boxes[i] = { x, y, w: Math.max(a.x + a.w, b.x + b.w) - x, h: Math.max(a.y + a.h, b.y + b.h) - y };
          boxes.splice(j, 1);
          merged = true;
          break;
        }
      }
    }
    return boxes;
  }

  root.createDetector = createDetector;
  if (typeof module === 'object' && module.exports) module.exports = { createDetector, DEFAULTS };
})(typeof self !== 'undefined' ? self : globalThis);
