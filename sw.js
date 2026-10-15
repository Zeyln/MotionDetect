// Offline support. App files are network-first so edits show up on reload;
// the 10 MB OpenCV.js file is cache-first since it never changes.
const CACHE = 'motiondetect-v3';
const APP_FILES = [
  './',
  'index.html',
  'styles.css',
  'outline.js',
  'app.js',
  'worker.js',
  'imgproc.js',
  'detector.js',
  'manifest.webmanifest',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'icons/apple-touch-icon.png',
];
const VENDOR = 'vendor/opencv.js';

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll([...APP_FILES, VENDOR])).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET' || new URL(request.url).origin !== location.origin) return;

  if (request.url.endsWith(VENDOR)) {
    event.respondWith(caches.match(request).then((hit) => hit || fetch(request)));
    return;
  }
  event.respondWith(
    fetch(request)
      .then((response) => {
        if (response.ok) {
          const copy = response.clone();
          caches.open(CACHE).then((cache) => cache.put(request, copy));
        }
        return response;
      })
      .catch(() => caches.match(request))
  );
});
