'use strict';

const $ = (id) => document.getElementById(id);
const video = $('cam');
const overlay = $('overlay');
const overlayCtx = overlay.getContext('2d');
const stage = $('stage');
const ui = $('ui');
const hud = $('hud');
const statusEl = $('status');
const startBtn = $('start');
const message = $('message');

// Full-resolution copy of the frame being analysed, shown on screen with its heat when the video
// is synced; the detector gets a downscaled copy of the same frame.
const frame = document.createElement('canvas');
const frameCtx = frame.getContext('2d');
const grab = document.createElement('canvas');
const grabCtx = grab.getContext('2d', { willReadFrequently: true });
let frameReady = false; // `frame` holds a captured picture (reset when the camera starts)

// Heatmap from the detector (a quarter of the analysed frame size), coloured through a lookup table:
// transparent for no heat, then red, orange and yellow as it gets hotter.
const heatCanvas = document.createElement('canvas');
const heatCtx = heatCanvas.getContext('2d');
let heatImage = null;
let heat = null;
const HEAT_COLORS = (() => {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let v = 0; v < 256; v++) {
    const t = v / 255;
    lut[4 * v] = 255;
    lut[4 * v + 1] = 255 * Math.min(1, Math.max(0, (t - 0.3) / 0.7));
    lut[4 * v + 2] = 255 * Math.min(1, Math.max(0, (t - 0.85) / 0.15)) * 0.5;
    lut[4 * v + 3] = t < 0.05 ? 0 : 255 * Math.min(0.75, 0.15 + 0.8 * t);
  }
  return lut;
})();

const DETAILS = [
  { value: 480, name: 'Fast' },
  { value: 640, name: 'Normal' },
  { value: 800, name: 'High' },
  { value: 960, name: 'Max' },
];

const SETTINGS_KEY = 'motiondetect.settings';
// Bump when a default changes in a way that should override what's saved on the phone.
const SETTINGS_VERSION = 2;
const settings = Object.assign(
  {
    mode: 'detect', // 'detect' (motion detector) or 'extract' (time-difference effects)
    sensitivity: 6,
    minSize: 3,
    display: 'heatmap', // 'heatmap', 'outline' or 'both'
    heatFade: 1, // seconds
    detail: 800,
    syncVideo: true,
    effect: 'overlay',
    delay: 6, // frames
    strengths: {}, // per effect, 1..5
    showStats: false,
    facing: 'environment',
  },
  loadSettings(),
  { version: SETTINGS_VERSION }
);

let worker = null;
let workerReady = false;
let inFlight = false;
let stream = null;
let running = false;
let wakeLock = null;
let boxes = [];
let boxesFrame = { w: 1, h: 1 }; // size of the frame the heat was computed on
let lastMode = 'init';
let lastFrameTime = null; // capture time of the last camera frame used, to skip repeats
const stats = { frames: 0, since: performance.now(), fps: 0, ms: 0, inliers: 0 };
const extractor = createExtractor();
const camera = { frames: 0, since: performance.now(), fps: 30 }; // camera frame rate, for the Delay label

// ---------- Settings ----------

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {};
    // Version 1 stored `detail` as frame width; it is now the long side, with new options.
    if (saved.version !== SETTINGS_VERSION) delete saved.detail;
    // Rectangular boxes were replaced by outlines of the heat.
    if (saved.display === 'boxes') saved.display = 'outline';
    return saved;
  } catch {
    return {};
  }
}

function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    // Private mode or storage disabled: settings just won't persist.
  }
}

// Slider positions (1-10) mapped to detector options.
function detectorOptions() {
  return {
    threshold: Math.round(60 - 5 * settings.sensitivity), // 55 (least sensitive) .. 10 (most)
    // Area of the motion box as a fraction of the frame: 3 is about 6x6 px at 800 px; 1 lets
    // through specks of a couple of pixels, 10 only things covering several percent of the view.
    minArea: 0.0001 * Math.pow(2.4, settings.minSize - 3),
    heatmap: true, // both displays are drawn from the heat
    heatFadeMs: settings.heatFade * 1000,
  };
}

const effectInfo = (id) => EXTRACT_EFFECTS.find((e) => e.id === id) || EXTRACT_EFFECTS[0];
const strengthOf = (id) => settings.strengths[id] || effectInfo(id).initial || 3;

// Shows the settings on the controls and applies them.
function applySettings() {
  const detect = settings.mode === 'detect';
  setChoice($('mode'), settings.mode);
  $('detect-controls').hidden = !detect;
  $('extract-controls').hidden = detect;

  setChoice($('view'), settings.display);
  $('sensitivity').value = settings.sensitivity;
  $('sensitivity-out').textContent = settings.sensitivity;
  $('min-size').value = settings.minSize;
  $('min-size-out').textContent = settings.minSize;
  $('heat-fade').value = settings.heatFade;
  $('heat-fade-out').textContent = settings.heatFade.toFixed(1) + ' s';
  $('detail').textContent = 'Detail: ' + (DETAILS.find((d) => d.value === settings.detail) || DETAILS[2]).name;
  $('sync').setAttribute('aria-pressed', String(settings.syncVideo));

  const effect = effectInfo(settings.effect);
  setChoice($('effects'), effect.id);
  $('effect-about').textContent = effect.about;
  $('delay').value = settings.delay;
  $('delay-out').textContent = Math.round((settings.delay * 1000) / camera.fps) + ' ms';
  $('strength-row').hidden = !effect.strength;
  $('strength-label').textContent = effect.strength || '';
  $('strength').value = strengthOf(effect.id);
  $('strength-out').textContent = strengthOf(effect.id);

  $('stats-btn').setAttribute('aria-pressed', String(settings.showStats));
  hud.hidden = !settings.showStats || !running;
  // The canvas shows the picture itself in Extract mode, and in Detect mode when synced.
  stage.classList.toggle('canvas-only', !detect || settings.syncVideo);
  if (worker) worker.postMessage({ type: 'config', options: detectorOptions() });
}

function changed() {
  saveSettings();
  applySettings();
  draw();
}

// Segmented controls and chips: buttons with data-value, exactly one checked.
function setChoice(group, value) {
  for (const b of group.querySelectorAll('button')) {
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', String(b.dataset.value === value));
  }
}
function onChoice(group, handler) {
  group.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (b && b.dataset.value) handler(b.dataset.value);
  });
}

// Effect chips are built from the list in effects.js.
for (const e of EXTRACT_EFFECTS) {
  const b = document.createElement('button');
  b.type = 'button';
  b.dataset.value = e.id;
  b.textContent = e.name;
  $('effects').append(b);
}

onChoice($('mode'), (mode) => {
  settings.mode = mode;
  changed();
  enterMode();
});
onChoice($('view'), (v) => {
  settings.display = v;
  changed();
});
onChoice($('effects'), (id) => {
  settings.effect = id;
  extractor.reset(); // Trails would otherwise carry over
  changed();
});
const sliders = {
  sensitivity: (v) => (settings.sensitivity = v),
  'min-size': (v) => (settings.minSize = v),
  'heat-fade': (v) => (settings.heatFade = v),
  delay: (v) => (settings.delay = v),
  strength: (v) => (settings.strengths[settings.effect] = v),
};
for (const [id, set] of Object.entries(sliders)) {
  $(id).addEventListener('input', (e) => {
    set(Number(e.target.value));
    changed();
  });
}
$('detail').addEventListener('click', () => {
  const i = DETAILS.findIndex((d) => d.value === settings.detail);
  settings.detail = DETAILS[(i + 1) % DETAILS.length].value;
  changed();
});
$('sync').addEventListener('click', () => {
  settings.syncVideo = !settings.syncVideo;
  changed();
});
$('stats-btn').addEventListener('click', () => {
  settings.showStats = !settings.showStats;
  changed();
});

// Tapping the picture hides or shows all controls.
stage.addEventListener('click', () => {
  if (!running) return;
  $('panel').hidden = !$('panel').hidden;
  $('hint').hidden = true;
});

// ---------- Detector worker (Detect mode only) ----------

function startWorker() {
  if (worker) return;
  workerReady = false;
  showStatus('Loading detector…');
  worker = new Worker('worker.js');
  worker.postMessage({ type: 'config', options: detectorOptions() });
  worker.onmessage = ({ data: msg }) => {
    if (msg.type === 'ready') {
      workerReady = true;
      showStatus(null);
    } else if (msg.type === 'result') {
      inFlight = false;
      boxes = msg.boxes;
      heat = msg.heat;
      lastMode = msg.mode;
      stats.ms = msg.ms;
      stats.inliers = msg.inliers;
      countFrame();
      draw();
    } else if (msg.type === 'error') {
      inFlight = false;
      console.error('Detector error:', msg.message);
      draw(); // keep the synced picture moving even if one frame failed
    }
  };
  worker.onerror = (e) => showStatus('Could not load the detector: ' + (e.message || 'unknown error'));
}

// Extract mode runs without the detector: unload it entirely.
function stopWorker() {
  if (!worker) return;
  worker.terminate();
  worker = null;
  workerReady = false;
  inFlight = false;
  boxes = [];
  heat = null;
  showStatus(null);
}

function showStatus(text) {
  statusEl.textContent = text || '';
  statusEl.hidden = !text;
}

function enterMode() {
  if (settings.mode === 'detect') {
    startWorker();
    frameReady = false;
  } else {
    stopWorker();
    extractor.reset();
  }
  stats.frames = 0;
  stats.since = performance.now();
  draw();
}

// ---------- Camera ----------

function showStartError(text) {
  message.textContent = text;
  $('start-screen').hidden = false;
  ui.hidden = true;
}

async function startCamera() {
  message.textContent = '';
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    showStartError('Camera access needs HTTPS. Open this page over an https:// address.');
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: settings.facing, width: { ideal: 1280 }, height: { ideal: 720 } },
    });
  } catch (e) {
    showStartError(
      e.name === 'NotAllowedError'
        ? 'Camera permission was denied. Allow camera access for this site in Settings, then reload.'
        : 'Could not open the camera: ' + e.message
    );
    return;
  }
  video.srcObject = stream;
  await video.play().catch(() => {});
  stage.classList.toggle('mirrored', settings.facing === 'user');
  running = true;
  boxes = [];
  heat = null;
  frameReady = false;
  lastFrameTime = null;
  extractor.reset();
  if (worker) worker.postMessage({ type: 'reset' });
  $('start-screen').hidden = true;
  ui.hidden = false;
  applySettings();
  requestWakeLock();
  scheduleFrame();
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
}

async function requestWakeLock() {
  try {
    wakeLock = await navigator.wakeLock?.request('screen');
  } catch {
    // Not supported or refused; the screen may dim during use.
  }
}

startBtn.addEventListener('click', async () => {
  await startCamera();
  if (!running) return;
  enterMode();
  // First time only: tell people they can hide the controls.
  try {
    if (!localStorage.getItem('motiondetect.hinted')) {
      $('hint').hidden = false;
      setTimeout(() => ($('hint').hidden = true), 3500);
      localStorage.setItem('motiondetect.hinted', '1');
    }
  } catch {
    // Storage unavailable: skip the hint.
  }
});

$('flip').addEventListener('click', async () => {
  settings.facing = settings.facing === 'user' ? 'environment' : 'user';
  saveSettings();
  stopCamera();
  await startCamera();
});

// iOS stops the camera when the app is backgrounded; restart it on return.
document.addEventListener('visibilitychange', async () => {
  if (document.hidden) {
    if (running) {
      stopCamera();
      running = 'paused';
    }
  } else if (running === 'paused') {
    await startCamera();
  }
});

// ---------- Frame loop ----------

function scheduleFrame() {
  if (running !== true) return;
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame(onFrame);
}

function onFrame(now, meta) {
  if (running !== true) return;
  // The callback can fire on every screen refresh while the camera delivers fewer new pictures
  // (an iPhone drops to ~20 fps in dim light), handing over the same picture several times. A repeat
  // compared with itself shows no motion at all, so only frames with a new capture time are used.
  const t = meta ? meta.captureTime ?? meta.mediaTime : undefined;
  if (t !== undefined && t === lastFrameTime) {
    scheduleFrame();
    return;
  }
  lastFrameTime = t ?? null;
  countCameraFrame();
  if (settings.mode === 'extract') {
    extractor.push(video);
    countFrame();
    drawExtract();
  } else if (workerReady && !inFlight && video.videoWidth) {
    // Drop frames while the detector is busy so the heat never lags behind the picture.
    sendFrame();
  }
  scheduleFrame();
}

function sendFrame() {
  // `detail` is the long side, so portrait and landscape process the same number of pixels.
  const { videoWidth: vw, videoHeight: vh } = video;
  const w = vw >= vh ? settings.detail : Math.round((settings.detail * vw) / vh);
  const h = vw >= vh ? Math.round((settings.detail * vh) / vw) : settings.detail;
  if (grab.width !== w || grab.height !== h) {
    grab.width = w;
    grab.height = h;
  }
  if (settings.syncVideo) {
    // Snapshot the frame once, then downscale from the snapshot, so the picture shown later is
    // exactly the one the heat comes from.
    if (frame.width !== vw || frame.height !== vh) {
      frame.width = vw;
      frame.height = vh;
    }
    frameCtx.drawImage(video, 0, 0, vw, vh);
    frameReady = true;
    grabCtx.drawImage(frame, 0, 0, w, h);
  } else {
    grabCtx.drawImage(video, 0, 0, w, h);
  }
  const { data } = grabCtx.getImageData(0, 0, w, h);
  boxesFrame = { w, h };
  inFlight = true;
  worker.postMessage({ type: 'frame', buffer: data.buffer, width: w, height: h }, [data.buffer]);
}

function countFrame() {
  stats.frames++;
  const now = performance.now();
  if (now - stats.since >= 1000) {
    stats.fps = (stats.frames * 1000) / (now - stats.since);
    stats.frames = 0;
    stats.since = now;
  }
}

// The camera's own frame rate, so the Delay slider can say how long its frames are.
function countCameraFrame() {
  camera.frames++;
  const now = performance.now();
  if (now - camera.since >= 2000) {
    camera.fps = Math.max(1, (camera.frames * 1000) / (now - camera.since));
    camera.frames = 0;
    camera.since = now;
    if (settings.mode === 'extract') $('delay-out').textContent = Math.round((settings.delay * 1000) / camera.fps) + ' ms';
  }
}

// ---------- Drawing ----------

function resizeOverlay() {
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.round(overlay.clientWidth * dpr);
  overlay.height = Math.round(overlay.clientHeight * dpr);
  draw();
}
window.addEventListener('resize', resizeOverlay);

function updateHud(extra) {
  if (settings.showStats && running) hud.textContent = `${stats.fps.toFixed(0)} fps${extra}`;
}

function drawExtract() {
  const W = overlay.width;
  const H = overlay.height;
  if (!extractor.render(overlayCtx, W, H, settings.effect, settings.delay, strengthOf(settings.effect))) {
    overlayCtx.clearRect(0, 0, W, H);
  }
  updateHud('');
}

function draw() {
  if (settings.mode === 'extract') {
    if (running) drawExtract();
    return;
  }
  const W = overlay.width;
  const H = overlay.height;
  overlayCtx.clearRect(0, 0, W, H);

  // Synced: show the analysed frame itself (cropped to fill the screen like the live video), so
  // the heat lines up with the picture. It is one frame behind live and updates at the detector's
  // frame rate.
  if (settings.syncVideo && running && frameReady) {
    const fit = Math.max(W / frame.width, H / frame.height);
    const fw = frame.width * fit;
    const fh = frame.height * fit;
    overlayCtx.drawImage(frame, (W - fw) / 2, (H - fh) / 2, fw, fh);
  }

  const note = lastMode === 'lost' ? ' · moving too fast' : lastMode === 'noisy' ? ' · unstable' : '';
  updateHud(` · ${stats.ms.toFixed(0)} ms · ${stats.inliers} pts · ${boxes.filter((b) => !b.held).length} moving${note}`);
  if (!video.videoWidth) return;

  // The video is drawn with object-fit: cover, so map frame coordinates the same way.
  const scale = Math.max(W / boxesFrame.w, H / boxesFrame.h);
  const offX = (W - boxesFrame.w * scale) / 2;
  const offY = (H - boxesFrame.h * scale) / 2;
  const dpr = window.devicePixelRatio || 1;

  if (!heat) return;
  const fw = boxesFrame.w * scale;
  const fh = boxesFrame.h * scale;
  if (settings.display !== 'outline') drawHeat(offX, offY, fw, fh);
  if (settings.display !== 'heatmap') drawOutlines(offX, offY, fw, fh, dpr);
}

// Heat level (of 255) the outline follows: about where the heatmap's glow becomes clearly visible.
const OUTLINE_LEVEL = 40;

// Outlines the areas the heatmap covers. Areas with motion right now get a solid red line; areas
// that are only fading (the target stopped or left) get a fainter one until their heat is gone.
function drawOutlines(x, y, w, h, dpr) {
  const { segments, segmentAreas, labels, count } = outline.trace(heat.data, heat.w, heat.h, OUTLINE_LEVEL);
  if (!segments.length) return;
  // An area is "moving" if a target was seen in it this frame.
  const moving = new Uint8Array(count + 1);
  const s = heat.w / boxesFrame.w;
  for (const b of boxes) {
    if (b.held) continue;
    const x0 = Math.max(0, Math.floor(b.x * s));
    const y0 = Math.max(0, Math.floor(b.y * s));
    const x1 = Math.min(heat.w, Math.ceil((b.x + b.w) * s));
    const y1 = Math.min(heat.h, Math.ceil((b.y + b.h) * s));
    for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) moving[labels[yy * heat.w + xx]] = 1;
  }
  const sx = w / heat.w;
  const sy = h / heat.h;
  const solid = new Path2D();
  const faint = new Path2D();
  for (let i = 0; i < segmentAreas.length; i++) {
    const path = moving[segmentAreas[i]] ? solid : faint;
    path.moveTo(x + segments[4 * i] * sx, y + segments[4 * i + 1] * sy);
    path.lineTo(x + segments[4 * i + 2] * sx, y + segments[4 * i + 3] * sy);
  }
  overlayCtx.lineWidth = 3 * dpr;
  overlayCtx.lineCap = 'round';
  overlayCtx.lineJoin = 'round';
  overlayCtx.strokeStyle = 'rgba(255, 32, 32, 0.45)';
  overlayCtx.stroke(faint);
  overlayCtx.strokeStyle = '#ff2020';
  overlayCtx.stroke(solid);
}

// Colours the heatmap and stretches it over the picture; smoothing while scaling it up gives the
// soft glow.
function drawHeat(x, y, w, h) {
  if (heatCanvas.width !== heat.w || heatCanvas.height !== heat.h) {
    heatCanvas.width = heat.w;
    heatCanvas.height = heat.h;
    heatImage = heatCtx.createImageData(heat.w, heat.h);
  }
  const px = heatImage.data;
  for (let i = 0; i < heat.data.length; i++) {
    const c = 4 * heat.data[i];
    px[4 * i] = HEAT_COLORS[c];
    px[4 * i + 1] = HEAT_COLORS[c + 1];
    px[4 * i + 2] = HEAT_COLORS[c + 2];
    px[4 * i + 3] = HEAT_COLORS[c + 3];
  }
  heatCtx.putImageData(heatImage, 0, 0);
  overlayCtx.imageSmoothingEnabled = true;
  overlayCtx.drawImage(heatCanvas, x, y, w, h);
}

// ---------- Boot ----------

applySettings();
resizeOverlay();
// Start loading the detector right away if the app opens in Detect mode.
if (settings.mode === 'detect') startWorker();

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch((e) => console.warn('Service worker not registered:', e));
}
