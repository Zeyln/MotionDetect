/*
 * Extract mode: "time difference" effects that compare the live picture with itself a moment ago.
 *
 * Everything is canvas blending, done by the graphics chip; no detector or OpenCV is involved. The
 * last frames are kept in a ring of canvases at 720 px, each effect is composed on one working
 * canvas of that size, and the result is scaled once onto the screen.
 *
 * Exposes `createExtractor()` (global, for the page).
 */
(function (root) {
  'use strict';

  const MAX_DELAY = 30; // frames of history kept (about 1 s at 30 fps)
  const LONG_SIDE = 720;

  /**
   * The effects, in the order they appear. `strength` names what the second slider does for that
   * effect (null when it has none), `initial` is its starting value (1..5).
   */
  const EFFECTS = [
    { id: 'overlay', name: 'Inverted overlay', strength: 'Contrast', initial: 2, about: 'Still areas turn grey; movement shows as light and dark edges.' },
    { id: 'difference', name: 'Difference', strength: 'Gain', initial: 2, about: 'Still areas turn black; anything that moved glows in its own colours.' },
    { id: 'exclusion', name: 'Exclusion', strength: null, about: 'Still areas take on strange washed-out colours; movers flash in inverted ones.' },
    { id: 'split', name: 'Colour split', strength: null, about: 'Red, green and blue come from different moments: movers get rainbow fringes.' },
    { id: 'thermal', name: 'Thermal', strength: 'Gain', initial: 3, about: 'How much each spot changed, in heat colours over a dimmed picture.' },
    { id: 'trails', name: 'Trails', strength: 'Trail length', initial: 3, about: 'Movement leaves glowing light trails, like a long exposure.' },
    { id: 'echo', name: 'Echo', strength: 'Copies', initial: 3, about: 'Fading copies of whatever moves, like a strobe photo.' },
  ];

  // Heat colours for Thermal: black, purple, red, orange, yellow, white.
  const THERMAL = (() => {
    const stops = [[0, 0, 0, 0], [0.15, 60, 0, 90], [0.4, 200, 0, 60], [0.65, 255, 120, 0], [0.85, 255, 230, 40], [1, 255, 255, 255]];
    const lut = new Uint8ClampedArray(256 * 3);
    for (let v = 0; v < 256; v++) {
      const t = v / 255;
      let i = 1;
      while (stops[i][0] < t) i++;
      const [t0, r0, g0, b0] = stops[i - 1];
      const [t1, r1, g1, b1] = stops[i];
      const f = (t - t0) / (t1 - t0);
      lut[3 * v] = r0 + (r1 - r0) * f;
      lut[3 * v + 1] = g0 + (g1 - g0) * f;
      lut[3 * v + 2] = b0 + (b1 - b0) * f;
    }
    return lut;
  })();

  function canvas(w, h, opts) {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    return { c, g: c.getContext('2d', opts) };
  }

  function createExtractor() {
    let w = 0;
    let h = 0;
    let ring = [];
    let head = -1;
    let filled = 0;
    let comp, work, trail, small; // working canvases at ring size (small: for Thermal)
    let thermalImage = null;

    function resize(vw, vh) {
      const s = LONG_SIDE / Math.max(vw, vh);
      w = Math.round(vw * s);
      h = Math.round(vh * s);
      ring = [];
      head = -1;
      filled = 0;
      comp = canvas(w, h);
      work = canvas(w, h);
      trail = canvas(w, h);
      const sw = Math.max(1, Math.round(w / 3));
      const sh = Math.max(1, Math.round(h / 3));
      small = canvas(sw, sh, { willReadFrequently: true });
      thermalImage = small.g.createImageData(sw, sh);
    }

    // Stores the video's current frame as the newest in the history.
    function push(video) {
      const { videoWidth: vw, videoHeight: vh } = video;
      if (!vw) return;
      const s = LONG_SIDE / Math.max(vw, vh);
      if (Math.round(vw * s) !== w || Math.round(vh * s) !== h) resize(vw, vh);
      head = (head + 1) % (MAX_DELAY + 1);
      if (!ring[head]) ring[head] = canvas(w, h);
      ring[head].g.drawImage(video, 0, 0, w, h);
      filled = Math.min(filled + 1, MAX_DELAY + 1);
    }

    function reset() {
      filled = 0;
      head = -1;
      if (trail) trail.g.clearRect(0, 0, w, h);
    }

    // The frame `k` frames ago (or the oldest one kept, early on).
    function past(k) {
      const back = Math.min(k, filled - 1);
      return ring[(head - back + (MAX_DELAY + 1) * 2) % (MAX_DELAY + 1)].c;
    }

    function blend(g, mode, alpha = 1) {
      g.globalCompositeOperation = mode;
      g.globalAlpha = alpha;
    }

    // Multiplies the composite's brightness by 2^passes (adding it onto itself).
    function boost(passes) {
      for (let i = 0; i < passes; i++) {
        blend(work.g, 'copy');
        work.g.drawImage(comp.c, 0, 0);
        blend(comp.g, 'lighter');
        comp.g.drawImage(work.c, 0, 0);
      }
    }

    // `src` inverted, into `work`.
    function inverted(src) {
      blend(work.g, 'copy');
      work.g.drawImage(src, 0, 0);
      blend(work.g, 'difference');
      work.g.fillStyle = '#fff';
      work.g.fillRect(0, 0, w, h);
      return work.c;
    }

    const draw = {
      // The motion-extraction trick: a half-transparent inverted copy of the past frame over the
      // live one. Still areas cancel to grey; Contrast then pushes everything away from grey.
      overlay(delay, strength) {
        const g = comp.g;
        blend(g, 'copy');
        g.drawImage(past(0), 0, 0);
        blend(g, 'source-over', 0.5);
        g.drawImage(inverted(past(delay)), 0, 0);
        for (let i = 1; i < strength; i++) {
          blend(work.g, 'copy');
          work.g.drawImage(comp.c, 0, 0);
          blend(g, 'overlay'); // overlaying a grey-centred image on itself steepens its contrast
          g.drawImage(work.c, 0, 0);
        }
      },
      difference(delay, strength) {
        blend(comp.g, 'copy');
        comp.g.drawImage(past(0), 0, 0);
        blend(comp.g, 'difference');
        comp.g.drawImage(past(delay), 0, 0);
        boost(strength - 1);
      },
      // Exclusion blend (a + b - 2ab) of now and then: where they match it gives 2a(1 - a), a washed-out
      // inversion; where something moved the colours flip.
      exclusion(delay) {
        blend(comp.g, 'copy');
        comp.g.drawImage(past(0), 0, 0);
        blend(comp.g, 'exclusion');
        comp.g.drawImage(past(delay), 0, 0);
      },
      // Each colour channel from a different moment: now, `delay` ago and twice that.
      split(delay) {
        const g = comp.g;
        blend(g, 'copy');
        g.fillStyle = '#000';
        g.fillRect(0, 0, w, h);
        for (const [k, color] of [[0, '#f00'], [delay, '#0f0'], [Math.min(MAX_DELAY, 2 * delay), '#00f']]) {
          blend(work.g, 'copy');
          work.g.drawImage(past(k), 0, 0);
          blend(work.g, 'multiply');
          work.g.fillStyle = color;
          work.g.fillRect(0, 0, w, h);
          blend(g, 'lighter');
          g.drawImage(work.c, 0, 0);
        }
      },
      // The amount of change, read back at a third of the size and coloured, over a dimmed picture.
      thermal(delay, strength) {
        const s = small;
        blend(s.g, 'copy');
        s.g.drawImage(past(0), 0, 0, s.c.width, s.c.height);
        blend(s.g, 'difference');
        s.g.drawImage(past(delay), 0, 0, s.c.width, s.c.height);
        const src = s.g.getImageData(0, 0, s.c.width, s.c.height).data;
        const out = thermalImage.data;
        const gain = 2 * strength;
        for (let i = 0; i < src.length; i += 4) {
          const v = Math.min(255, ((src[i] + src[i + 1] + src[i + 2]) / 3) * gain) | 0;
          out[i] = THERMAL[3 * v];
          out[i + 1] = THERMAL[3 * v + 1];
          out[i + 2] = THERMAL[3 * v + 2];
          out[i + 3] = 255;
        }
        s.g.putImageData(thermalImage, 0, 0);
        const g = comp.g;
        blend(g, 'copy');
        g.drawImage(past(0), 0, 0);
        blend(g, 'source-over', 0.7);
        g.fillStyle = '#000';
        g.fillRect(0, 0, w, h);
        blend(g, 'screen');
        g.imageSmoothingEnabled = true;
        g.drawImage(s.c, 0, 0, w, h);
      },
      // Frame-to-frame change is added onto a slowly fading layer, which is screened over the picture.
      trails(delay, strength) {
        const t = trail.g;
        blend(t, 'source-over', [0.3, 0.15, 0.08, 0.04, 0.02][strength - 1]);
        t.fillStyle = '#000';
        t.fillRect(0, 0, w, h);
        blend(work.g, 'copy');
        work.g.drawImage(past(0), 0, 0);
        blend(work.g, 'difference');
        work.g.drawImage(past(Math.min(delay, 3)), 0, 0);
        blend(t, 'lighter');
        t.drawImage(work.c, 0, 0);
        t.drawImage(work.c, 0, 0); // twice: brighter trails
        const g = comp.g;
        blend(g, 'copy');
        g.drawImage(past(0), 0, 0);
        blend(g, 'source-over', 0.45);
        g.fillStyle = '#000';
        g.fillRect(0, 0, w, h);
        blend(g, 'screen');
        g.drawImage(trail.c, 0, 0);
      },
      // Older frames on top of the live one at falling opacity: still things don't change, movers
      // leave a row of fading copies.
      echo(delay, strength) {
        const g = comp.g;
        blend(g, 'copy');
        g.drawImage(past(0), 0, 0);
        for (let i = strength; i >= 1; i--) {
          blend(g, 'source-over', 0.55 / i);
          g.drawImage(past(Math.min(MAX_DELAY, i * delay)), 0, 0);
        }
      },
    };

    /**
     * Draws effect `id` over the whole of `ctx` (W x H), cropped to fill it like the live video.
     * `delay` is in frames (1..30); `strength` is 1..5.
     */
    function render(ctx, W, H, id, delay, strength) {
      if (!filled) return false;
      (draw[id] || draw.overlay)(Math.max(1, delay), Math.max(1, Math.min(5, strength)));
      blend(comp.g, 'source-over');
      const fit = Math.max(W / w, H / h);
      const dw = w * fit;
      const dh = h * fit;
      ctx.globalCompositeOperation = 'source-over';
      ctx.globalAlpha = 1;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(comp.c, (W - dw) / 2, (H - dh) / 2, dw, dh);
      return true;
    }

    return { push, render, reset };
  }

  root.createExtractor = createExtractor;
  root.EXTRACT_EFFECTS = EFFECTS;
  root.EXTRACT_MAX_DELAY = MAX_DELAY;
})(typeof self !== 'undefined' ? self : globalThis);
