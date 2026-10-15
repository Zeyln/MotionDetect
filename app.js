'use strict';

const $ = (id) => document.getElementById(id);
const video = $('cam');
const overlay = $('overlay');
const overlayCtx = overlay.getContext('2d');
const stage = $('stage');
const hud = $('hud');
const startBtn = $('start');
const message = $('message');

// Downscaled copy of each camera frame that gets sent to the detector.
const grab = document.createElement('canvas');
const grabCtx = grab.getContext('2d', { willReadFrequently: true });

const SETTINGS_KEY = 'motiondetect.settings';
// Bump when a default changes in a way that should override what's saved on the phone.
const SETTINGS_VERSION = 2;
const settings = Object.assign(
  { sensitivity: 6, minSize: 3, detail: 800, showStats: false, facing: 'environment' },
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
    minArea: 0.0005 * Math.pow(1.8, settings.minSize - 1), // 0.05% .. ~10% of the frame
  };
}

function applySettings() {
  $('sensitivity').value = settings.sensitivity;
  $('sensitivity-out').textContent = settings.sensitivity;
  $('min-size').value = settings.minSize;
  $('min-size-out').textContent = settings.minSize;
  $('detail').value = String(settings.detail);
  $('show-stats').checked = settings.showStats;
  hud.hidden = !settings.showStats || !running;
  if (worker) worker.postMessage({ type: 'config', options: detectorOptions() });
}

$('settings').addEventListener('input', () => {
  settings.sensitivity = Number($('sensitivity').value);
  settings.minSize = Number($('min-size').value);
  settings.detail = Number($('detail').value);
  settings.showStats = $('show-stats').checked;
  saveSettings();
  applySettings();
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
      lastMode = msg.mode;
      stats.ms = msg.ms;
      stats.inliers = msg.inliers;
      countFrame();
      draw();
    } else if (msg.type === 'error') {
      inFlight = false;
      console.error('Detector error:', msg.message);
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
  grabCtx.drawImage(video, 0, 0, w, h);
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

  if (settings.showStats && running) {
    const note = lastMode === 'lost' ? ' · moving too fast' : lastMode === 'noisy' ? ' · unstable' : '';
    hud.textContent =
      `${stats.fps.toFixed(0)} fps · ${stats.ms.toFixed(0)} ms · ${stats.inliers} pts · ${boxes.length} boxes${note}`;
  }
  if (!boxes.length || !video.videoWidth) return;

  // The video is drawn with object-fit: cover, so map frame coordinates the same way.
  const scale = Math.max(W / boxesFrame.w, H / boxesFrame.h);
  const offX = (W - boxesFrame.w * scale) / 2;
  const offY = (H - boxesFrame.h * scale) / 2;
  const dpr = window.devicePixelRatio || 1;

  overlayCtx.strokeStyle = '#ff2020';
  overlayCtx.lineWidth = 3 * dpr;
  overlayCtx.lineJoin = 'miter';
  for (const b of boxes) {
    overlayCtx.strokeRect(offX + b.x * scale, offY + b.y * scale, b.w * scale, b.h * scale);
  }
}

// ---------- Boot ----------

applySettings();
resizeOverlay();
startWorker();

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').catch((e) => console.warn('Service worker not registered:', e));
}
