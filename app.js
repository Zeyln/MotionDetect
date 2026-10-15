'use strict';

const $ = (id) => document.getElementById(id);
const video = $('cam');
const overlay = $('overlay');
const overlayCtx = overlay.getContext('2d');
const stage = $('stage');
const hud = $('hud');
const startBtn = $('start');
const message = $('message');

// Full-resolution copy of the frame being analysed, shown on screen with its boxes when the video
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

const SETTINGS_KEY = 'motiondetect.settings';
// Bump when a default changes in a way that should override what's saved on the phone.
const SETTINGS_VERSION = 2;
const settings = Object.assign(
  {
    sensitivity: 6,
    minSize: 3,
    display: 'heatmap', // 'heatmap', 'outline' or 'both'
    heatFade: 1, // seconds
    detail: 800,
    syncVideo: true,
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
let boxesFrame = { w: 1, h: 1 }; // size of the frame the boxes were computed on
let lastMode = 'init';
const stats = { frames: 0, since: performance.now(), fps: 0, ms: 0, inliers: 0 };

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

function applySettings() {
  $('sensitivity').value = settings.sensitivity;
  $('sensitivity-out').textContent = settings.sensitivity;
  $('min-size').value = settings.minSize;
  $('min-size-out').textContent = settings.minSize;
  $('display').value = settings.display;
  $('heat-fade').value = settings.heatFade;
  $('heat-fade-out').textContent = settings.heatFade + ' s';
  $('detail').value = String(settings.detail);
  $('sync-video').checked = settings.syncVideo;
  $('show-stats').checked = settings.showStats;
  hud.hidden = !settings.showStats || !running;
  // Synced: the live <video> is hidden and the analysed frames are drawn on the canvas instead.
  stage.classList.toggle('synced', settings.syncVideo);
  if (worker) worker.postMessage({ type: 'config', options: detectorOptions() });
}

$('settings').addEventListener('input', () => {
  settings.sensitivity = Number($('sensitivity').value);
  settings.minSize = Number($('min-size').value);
  settings.display = $('display').value;
  settings.heatFade = Number($('heat-fade').value);
  settings.detail = Number($('detail').value);
  settings.showStats = $('show-stats').checked;
  settings.syncVideo = $('sync-video').checked;
  saveSettings();
  applySettings();
  draw();
});

$('settings-btn').addEventListener('click', () => {
  const panel = $('settings');
  panel.hidden = !panel.hidden;
  $('settings-btn').setAttribute('aria-expanded', String(!panel.hidden));
});

// ---------- Detector worker ----------

function startWorker() {
  worker = new Worker('worker.js');
  worker.postMessage({ type: 'config', options: detectorOptions() });
  worker.onmessage = ({ data: msg }) => {
    if (msg.type === 'ready') {
      workerReady = true;
      startBtn.disabled = false;
      startBtn.textContent = 'Start camera';
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
  worker.onerror = (e) => {
    message.textContent = 'Could not load the detector: ' + (e.message || 'unknown error');
    startBtn.textContent = 'Detector failed';
  };
}

// ---------- Camera ----------

function showStartError(text) {
  message.textContent = text;
  $('start-screen').hidden = false;
  $('controls').hidden = true;
  hud.hidden = true;
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
  worker.postMessage({ type: 'reset' });
  $('start-screen').hidden = true;
  $('controls').hidden = false;
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

startBtn.addEventListener('click', startCamera);

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

function onFrame() {
  if (running !== true) return;
  // Drop frames while the detector is busy so boxes never lag behind the picture.
  if (workerReady && !inFlight && video.videoWidth) sendFrame();
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
    // exactly the one the boxes come from.
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

// ---------- Drawing ----------

function resizeOverlay() {
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.round(overlay.clientWidth * dpr);
  overlay.height = Math.round(overlay.clientHeight * dpr);
  draw();
}
window.addEventListener('resize', resizeOverlay);

function draw() {
  const W = overlay.width;
  const H = overlay.height;
  overlayCtx.clearRect(0, 0, W, H);

  // Synced: show the analysed frame itself (cropped to fill the screen like the live video), so
  // the boxes line up with the picture. It is one frame behind live and updates at the detector's
  // frame rate.
  if (settings.syncVideo && running && frameReady) {
    const fit = Math.max(W / frame.width, H / frame.height);
    const fw = frame.width * fit;
    const fh = frame.height * fit;
    overlayCtx.drawImage(frame, (W - fw) / 2, (H - fh) / 2, fw, fh);
  }

  if (settings.showStats && running) {
    const note = lastMode === 'lost' ? ' · moving too fast' : lastMode === 'noisy' ? ' · unstable' : '';
    hud.textContent =
      `${stats.fps.toFixed(0)} fps · ${stats.ms.toFixed(0)} ms · ${stats.inliers} pts · ` +
      `${boxes.filter((b) => !b.held).length} moving${note}`;
  }
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
startWorker();

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch((e) => console.warn('Service worker not registered:', e));
}
