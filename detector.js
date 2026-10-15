/*
 * Motion detector that compensates for camera movement, built on OpenCV.js.
 *
 * Per frame:
 *   1. Track corner features from the previous frame into the current one (Lucas-Kanade).
 *   2. Fit a homography to those matches with RANSAC. Background points dominate, so this
 *      is the camera's own motion; points on moving objects get rejected as outliers.
 *   3. Warp the previous frame onto the current one and take the absolute difference.
 *      Whatever still differs moved independently of the camera.
 *   4. Threshold, clean up with morphology, and return bounding boxes of the blobs.
 *   5. Only report boxes that were also seen in the previous frame(s), to suppress flicker.
 *
 * Works in a Web Worker (global `createDetector`) and in Node (module.exports) for tests.
 */
(function (root) {
  'use strict';

  const DEFAULTS = {
    threshold: 30,        // gray-level difference (0-255) that counts as change
    minArea: 0.002,       // smallest blob to report, as a fraction of the frame area
    maxChangedArea: 0.4,  // if more than this fraction changed, motion estimate is bad: report nothing
    minHits: 2,           // consecutive frames a box must appear in before it is reported
    maxCorners: 300,
    minFeatures: 12,      // fewer tracked points than this: camera motion can't be trusted
    minSpreadCells: 5,    // camera-motion inliers must cover this many cells of a 4x4 grid
    blur: 5,
  };

  function createDetector(cv, options) {
    const opts = Object.assign({}, DEFAULTS, options);

    let width = 0;
    let height = 0;
    let hasPrev = false;
    let prevBoxes = [];
    let rgba, gray, prevGray, warped, diff, valid, warpedValid;

    const empty = new cv.Mat();
    const identity = cv.Mat.eye(3, 3, cv.CV_64F);
    const kernelOpen = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
    const kernelGrow = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(7, 7));
    const winSize = new cv.Size(21, 21);
    const criteria = new cv.TermCriteria(cv.TermCriteria_EPS | cv.TermCriteria_COUNT, 20, 0.03);

    function freeFrames() {
      for (const m of [rgba, gray, prevGray, warped, diff, valid, warpedValid]) if (m) m.delete();
    }

    function allocate(w, h) {
      freeFrames();
      width = w;
      height = h;
      rgba = new cv.Mat(h, w, cv.CV_8UC4);
      gray = new cv.Mat(h, w, cv.CV_8UC1);
      prevGray = new cv.Mat(h, w, cv.CV_8UC1);
      warped = new cv.Mat(h, w, cv.CV_8UC1);
      diff = new cv.Mat(h, w, cv.CV_8UC1);
      valid = new cv.Mat(h, w, cv.CV_8UC1, new cv.Scalar(255));
      warpedValid = new cv.Mat(h, w, cv.CV_8UC1);
      reset();
    }

    function reset() {
      hasPrev = false;
      prevBoxes = [];
    }

    // Returns { H, features, inliers } where H maps prevGray -> gray, or H = null if tracking failed.
    // The caller must delete H unless it is the shared identity matrix.
    function estimateCameraMotion() {
      const corners = new cv.Mat();
      const next = new cv.Mat();
      const status = new cv.Mat();
      const err = new cv.Mat();
      try {
        cv.goodFeaturesToTrack(prevGray, corners, opts.maxCorners, 0.01, 7, empty, 7, false, 0.04);
        // Too little texture to track (blank wall, darkness). Assume a still camera.
        if (corners.rows < opts.minFeatures) return { H: identity, features: corners.rows, inliers: 0 };

        cv.calcOpticalFlowPyrLK(prevGray, gray, corners, next, status, err, winSize, 3, criteria);

        const src = [];
        const dst = [];
        for (let i = 0; i < status.rows; i++) {
          if (status.data[i] !== 1) continue;
          src.push(corners.data32F[2 * i], corners.data32F[2 * i + 1]);
          dst.push(next.data32F[2 * i], next.data32F[2 * i + 1]);
        }
        const tracked = src.length / 2;
        if (tracked < opts.minFeatures) return { H: null, features: tracked, inliers: 0 };

        const srcMat = cv.matFromArray(tracked, 1, cv.CV_32FC2, src);
        const dstMat = cv.matFromArray(tracked, 1, cv.CV_32FC2, dst);
        const inlierMask = new cv.Mat();
        const H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 3, inlierMask);
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
        status.delete();
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

    function findBoxes() {
      const contours = new cv.MatVector();
      const hierarchy = new cv.Mat();
      cv.findContours(diff, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      const minPx = opts.minArea * width * height;
      const boxes = [];
      for (let i = 0; i < contours.size(); i++) {
        const c = contours.get(i);
        const r = cv.boundingRect(c);
        c.delete();
        if (r.width * r.height >= minPx) boxes.push({ x: r.x, y: r.y, w: r.width, h: r.height });
      }
      contours.delete();
      hierarchy.delete();
      return mergeBoxes(boxes, Math.round(0.02 * Math.max(width, height)));
    }

    // Each box inherits the hit count of the previous boxes it overlaps, plus one.
    function confirm(boxes) {
      for (const b of boxes) {
        let hits = 0;
        for (const p of prevBoxes) if (overlaps(b, p, 0)) hits = Math.max(hits, p.hits);
        b.hits = hits + 1;
      }
      prevBoxes = boxes;
      return boxes.filter((b) => b.hits >= opts.minHits);
    }

    /**
     * Process one RGBA frame. Returns { boxes, mode, features, inliers }, with boxes in frame pixels.
     * mode: 'init' (first frame), 'ok', 'lost' (camera moved too fast to track), 'noisy' (too much changed).
     */
    function processFrame(data, w, h) {
      if (w !== width || h !== height) allocate(w, h);
      rgba.data.set(data);
      cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
      if (opts.blur > 1) cv.GaussianBlur(gray, gray, new cv.Size(opts.blur, opts.blur), 0);

      if (!hasPrev) {
        gray.copyTo(prevGray);
        hasPrev = true;
        return { boxes: [], mode: 'init', features: 0, inliers: 0 };
      }

      const { H, features, inliers } = estimateCameraMotion();
      if (!H) {
        gray.copyTo(prevGray);
        prevBoxes = [];
        return { boxes: [], mode: 'lost', features, inliers };
      }

      const size = new cv.Size(width, height);
      const black = new cv.Scalar(0);
      cv.warpPerspective(prevGray, warped, H, size, cv.INTER_LINEAR, cv.BORDER_CONSTANT, black);
      // Pixels warped in from outside the previous frame have no valid comparison.
      cv.warpPerspective(valid, warpedValid, H, size, cv.INTER_NEAREST, cv.BORDER_CONSTANT, black);
      cv.erode(warpedValid, warpedValid, kernelGrow);
      if (H !== identity) H.delete();

      cv.absdiff(gray, warped, diff);
      cv.threshold(diff, diff, opts.threshold, 255, cv.THRESH_BINARY);
      cv.bitwise_and(diff, warpedValid, diff);
      cv.morphologyEx(diff, diff, cv.MORPH_OPEN, kernelOpen);
      cv.dilate(diff, diff, kernelGrow, new cv.Point(-1, -1), 2);

      gray.copyTo(prevGray);

      if (cv.countNonZero(diff) > opts.maxChangedArea * width * height) {
        prevBoxes = [];
        return { boxes: [], mode: 'noisy', features, inliers };
      }
      return { boxes: confirm(findBoxes()), mode: 'ok', features, inliers };
    }

    function configure(next) {
      Object.assign(opts, next);
    }

    function dispose() {
      freeFrames();
      for (const m of [empty, identity, kernelOpen, kernelGrow]) m.delete();
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
