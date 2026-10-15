/*
 * Motion detector that compensates for camera movement, built on OpenCV.js.
 *
 * Per frame:
 *   1. Track corner features from the previous frame into the current one (Lucas-Kanade), then
 *      back again, keeping only points that return to where they started.
 *   2. Fit a homography to those matches with RANSAC. Background points dominate, so this
 *      is the camera's own motion; points on moving objects get rejected as outliers.
 *   3. Warp the previous frame onto the current one, match its brightness (auto exposure), and
 *      compare each pixel with its 3x3 neighbourhood there, which tolerates 1 px of misalignment.
 *      Whatever still differs moved independently of the camera.
 *   4. Threshold and clean up with morphology. Nearby blobs are grouped using a dilated copy of
 *      the mask, but each box is fitted to the undilated pixels so it hugs the object.
 *   5. Feed the boxes to a tracker. A target is reported once it has been seen in minHits frames.
 *      When it stops moving it is kept as a "held" box for holdMs, carried along with the camera
 *      motion so it stays on the target while the phone moves.
 *
 * Kernel sizes, feature spacing and RANSAC tolerance are tuned for a 320 px frame and scale with
 * the frame size. The noise filter stays 3x3 at every size so small, thin movers survive.
 *
 * Works in a Web Worker (global `createDetector`) and in Node (module.exports) for tests.
 */
(function (root) {
  'use strict';

  const DEFAULTS = {
    threshold: 30,        // gray-level difference (0-255) that counts as change
    minArea: 0.0001,      // smallest motion box to report, as a fraction of the frame area
    maxChangedArea: 0.4,  // if more than this fraction changed, motion estimate is bad: report nothing
    minHits: 2,           // frames a target must be seen in before it is reported
    holdMs: 3000,         // how long a target stays boxed after it stops moving
    matchDistance: 0.06,  // extra distance (fraction of frame size) a target may jump between frames
    maxTracks: 30,
    maxCorners: 300,      // at 320 px; scales with frame size up to 400
    minFeatures: 12,      // fewer tracked points than this: camera motion can't be trusted
    minSpreadCells: 5,    // camera-motion inliers must cover this many cells of a 4x4 grid
    maxRoundTrip: 1,      // px at 320 px: forward-backward tracking error allowed per point
  };

  const odd = (n) => Math.max(3, 2 * Math.floor(n / 2) + 1);

  function createDetector(cv, options) {
    const opts = Object.assign({}, DEFAULTS, options);

    let width = 0;
    let height = 0;
    let hasPrev = false;
    let tracks = []; // { x, y, w, h, vx, vy, hits, lastSeen, held, id }
    let nextId = 1;
    let rgba, gray, prevGray, warped, warpedMin, warpedMax, above, motion, groups, valid, warpedValid;
    let kernelOpen, kernelGrow;
    let scale = 1; // frame size relative to the 320 px the defaults are tuned for
    let pyramidLevels;

    const empty = new cv.Mat();
    const identity = cv.Mat.eye(3, 3, cv.CV_64F);
    const winSize = new cv.Size(21, 21);
    const blurSize = new cv.Size(5, 5);
    const criteria = new cv.TermCriteria(cv.TermCriteria_EPS | cv.TermCriteria_COUNT, 20, 0.03);

    function freeFrames() {
      const mats = [rgba, gray, prevGray, warped, warpedMin, warpedMax, above, motion, groups, valid, warpedValid];
      for (const m of [...mats, kernelOpen, kernelGrow]) {
        if (m) m.delete();
      }
    }

    function allocate(w, h) {
      freeFrames();
      width = w;
      height = h;
      scale = Math.max(w, h) / 320;
      pyramidLevels = Math.round(3 + Math.log2(Math.max(1, scale)));
      // Rectangular kernels: several times faster than elliptical ones in OpenCV.js, and the shape
      // doesn't matter since boxes are fitted to the exact motion pixels afterwards.
      kernelOpen = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
      kernelGrow = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(odd(7 * scale), odd(7 * scale)));
      rgba = new cv.Mat(h, w, cv.CV_8UC4);
      gray = new cv.Mat(h, w, cv.CV_8UC1);
      prevGray = new cv.Mat(h, w, cv.CV_8UC1);
      warped = new cv.Mat(h, w, cv.CV_8UC1);
      motion = new cv.Mat(h, w, cv.CV_8UC1);
      groups = new cv.Mat(h, w, cv.CV_8UC1);
      valid = new cv.Mat(h, w, cv.CV_8UC1, new cv.Scalar(255));
      warpedValid = new cv.Mat(h, w, cv.CV_8UC1);
      warpedMin = new cv.Mat(h, w, cv.CV_8UC1);
      warpedMax = new cv.Mat(h, w, cv.CV_8UC1);
      above = new cv.Mat(h, w, cv.CV_8UC1);
      reset();
    }

    function reset() {
      hasPrev = false;
      tracks = [];
    }

    // Returns { H, features, inliers } where H maps prevGray -> gray, or H = null if tracking failed.
    // The caller must delete H unless it is the shared identity matrix.
    function estimateCameraMotion() {
      const corners = new cv.Mat();
      const next = new cv.Mat();
      const back = new cv.Mat();
      const status = new cv.Mat();
      const backStatus = new cv.Mat();
      const err = new cv.Mat();
      try {
        const maxCorners = Math.min(400, Math.round(opts.maxCorners * scale));
        const spacing = Math.max(5, Math.round(7 * scale));
        cv.goodFeaturesToTrack(prevGray, corners, maxCorners, 0.01, spacing, empty, 7, false, 0.04);
        // Too little texture to track (blank wall, darkness). Assume a still camera.
        if (corners.rows < opts.minFeatures) return { H: identity, features: corners.rows, inliers: 0 };

        cv.calcOpticalFlowPyrLK(prevGray, gray, corners, next, status, err, winSize, pyramidLevels, criteria);
        cv.calcOpticalFlowPyrLK(gray, prevGray, next, back, backStatus, err, winSize, pyramidLevels, criteria);

        const maxRoundTrip = opts.maxRoundTrip * scale;
        const src = [];
        const dst = [];
        for (let i = 0; i < status.rows; i++) {
          if (status.data[i] !== 1 || backStatus.data[i] !== 1) continue;
          const dx = back.data32F[2 * i] - corners.data32F[2 * i];
          const dy = back.data32F[2 * i + 1] - corners.data32F[2 * i + 1];
          if (dx * dx + dy * dy > maxRoundTrip * maxRoundTrip) continue;
          src.push(corners.data32F[2 * i], corners.data32F[2 * i + 1]);
          dst.push(next.data32F[2 * i], next.data32F[2 * i + 1]);
        }
        const tracked = src.length / 2;
        if (tracked < opts.minFeatures) return { H: null, features: tracked, inliers: 0 };

        const srcMat = cv.matFromArray(tracked, 1, cv.CV_32FC2, src);
        const dstMat = cv.matFromArray(tracked, 1, cv.CV_32FC2, dst);
        const inlierMask = new cv.Mat();
        const H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 2 * scale, inlierMask, 2000, 0.995);
        let inliers = 0;
        let cells = 0;
        if (!H.empty()) ({ inliers, cells } = inlierSpread(inlierMask, dst));
        srcMat.delete();
        dstMat.delete();
        inlierMask.delete();

        // If the agreeing points are few or bunched together, they are probably on one moving
        // object over a plain background, and "compensating" would erase that object.
        // Assume a still camera instead; the changed-area guard catches it if that's wrong.
        if (H.empty() || inliers < opts.minFeatures || cells < opts.minSpreadCells) {
          H.delete();
          return { H: identity, features: tracked, inliers };
        }
        if (!plausible(H)) {
          H.delete();
          return { H: null, features: tracked, inliers };
        }
        return { H, features: tracked, inliers };
      } finally {
        corners.delete();
        next.delete();
        back.delete();
        status.delete();
        backStatus.delete();
        err.delete();
      }
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
    function plausible(H) {
      const m = H.data64F;
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

    // Groups come from the dilated mask; each box is then shrunk to the motion pixels inside it.
    function findBoxes() {
      const contours = new cv.MatVector();
      const hierarchy = new cv.Mat();
      cv.findContours(groups, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      const minPx = opts.minArea * width * height;
      const boxes = [];
      for (let i = 0; i < contours.size(); i++) {
        const c = contours.get(i);
        const g = cv.boundingRect(c);
        c.delete();
        const r = motionBounds(g);
        if (r && r.w * r.h >= minPx) boxes.push(r);
      }
      contours.delete();
      hierarchy.delete();
      return mergeBoxes(boxes, 0);
    }

    // Bounding box of the motion pixels inside rect g, or null if there are none.
    // (This OpenCV.js build has no findNonZero.)
    function motionBounds(g) {
      const px = motion.data;
      let minX = Infinity, minY = Infinity, maxX = -1, maxY = -1;
      for (let y = g.y; y < g.y + g.height; y++) {
        const row = y * width;
        for (let x = g.x; x < g.x + g.width; x++) {
          if (!px[row + x]) continue;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          maxY = y;
        }
      }
      return maxX < 0 ? null : { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
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
        t.vx = 0.5 * t.vx + 0.5 * (d.x + d.w / 2 - (t.x + t.w / 2));
        t.vy = 0.5 * t.vy + 0.5 * (d.y + d.h / 2 - (t.y + t.h / 2));
        Object.assign(t, { x: d.x, y: d.y, w: d.w, h: d.h, hits: t.hits + 1, lastSeen: now, held: false });
      }

      tracks = tracks.filter((t, ti) => {
        if (matchedTracks.has(ti)) return true;
        // It stopped: hold it where it was last seen.
        t.vx = t.vy = 0;
        t.held = true;
        return t.hits >= opts.minHits; // an unconfirmed one-frame blip was just noise
      });
      detections.forEach((d, di) => {
        if (!matchedDetections.has(di)) tracks.push({ ...d, vx: 0, vy: 0, hits: 1, lastSeen: now, held: false, id: nextId++ });
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
        .filter((t) => t.hits >= opts.minHits)
        .map((t) => {
          const x = Math.max(0, Math.round(t.x));
          const y = Math.max(0, Math.round(t.y));
          return {
            x,
            y,
            w: Math.min(width, Math.round(t.x + t.w)) - x,
            h: Math.min(height, Math.round(t.y + t.h)) - y,
            held: t.lastSeen !== now,
            id: t.id,
          };
        });
    }

    /**
     * Process one RGBA frame captured at time `now` (ms). Returns { boxes, mode, features, inliers }.
     * Boxes are in frame pixels; `held` marks targets that have stopped moving.
     * mode: 'init' (first frame), 'ok', 'lost' (camera moved too fast to track), 'noisy' (too much changed).
     */
    function processFrame(data, w, h, now = Date.now()) {
      if (w !== width || h !== height) allocate(w, h);
      rgba.data.set(data);
      cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
      cv.GaussianBlur(gray, gray, blurSize, 0); // 5x5 at any size; larger kernels are far slower here

      if (!hasPrev) {
        gray.copyTo(prevGray);
        hasPrev = true;
        return { boxes: [], mode: 'init', features: 0, inliers: 0 };
      }

      const { H, features, inliers } = estimateCameraMotion();
      expireTracks(now);
      if (!H) {
        // Can't tell where the camera went; leave held boxes where they are until it settles.
        gray.copyTo(prevGray);
        return { boxes: reportTracks(now), mode: 'lost', features, inliers };
      }
      if (H !== identity) moveTracks(H.data64F.slice());

      const size = new cv.Size(width, height);
      const black = new cv.Scalar(0);
      cv.warpPerspective(prevGray, warped, H, size, cv.INTER_LINEAR, cv.BORDER_CONSTANT, black);
      // Pixels warped in from outside the previous frame have no valid comparison.
      cv.warpPerspective(valid, warpedValid, H, size, cv.INTER_NEAREST, cv.BORDER_CONSTANT, black);
      cv.erode(warpedValid, warpedValid, kernelGrow);
      if (H !== identity) H.delete();

      matchBrightness();

      // A pixel only counts as changed if it falls outside the brightness range of its 3x3
      // neighbourhood in the aligned previous frame. This forgives up to 1 px of misalignment, which
      // otherwise flickers on every sharp edge; real movers shift more than that against their
      // surroundings. Kept at 3x3 at every size so small, slow movers still register.
      cv.erode(warped, warpedMin, kernelOpen);
      cv.dilate(warped, warpedMax, kernelOpen);
      cv.subtract(gray, warpedMax, above); // saturates at 0: only where brighter than any neighbour
      cv.subtract(warpedMin, gray, motion); // only where darker than any neighbour
      cv.add(motion, above, motion);
      cv.threshold(motion, motion, opts.threshold, 255, cv.THRESH_BINARY);
      cv.bitwise_and(motion, warpedValid, motion);
      cv.morphologyEx(motion, motion, cv.MORPH_OPEN, kernelOpen);
      cv.dilate(motion, groups, kernelGrow, new cv.Point(-1, -1), 2);

      gray.copyTo(prevGray);

      if (cv.countNonZero(groups) > opts.maxChangedArea * width * height) {
        return { boxes: reportTracks(now), mode: 'noisy', features, inliers };
      }
      updateTracks(findBoxes(), now);
      return { boxes: reportTracks(now), mode: 'ok', features, inliers };
    }

    // Phones adjust exposure as the scene changes, brightening or darkening the whole frame.
    // Scale the warped previous frame so its average brightness matches the current one.
    function matchBrightness() {
      const before = cv.mean(warped, warpedValid)[0];
      const now = cv.mean(gray, warpedValid)[0];
      if (before < 1) return;
      const gain = now / before;
      if (Math.abs(gain - 1) > 0.01) warped.convertTo(warped, -1, gain, 0);
    }

    function configure(next) {
      Object.assign(opts, next);
    }

    function dispose() {
      freeFrames();
      for (const m of [empty, identity]) m.delete();
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
