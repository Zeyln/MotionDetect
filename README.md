# MotionDetect

A phone web app (PWA) that shows the camera and draws red boxes around anything moving in the shot. It runs whenever the camera view is open, without recording, and it compensates for the phone itself moving.

## How detection works

`detector.js` runs in a Web Worker using OpenCV.js (`vendor/opencv.js`, 4.10, about 10 MB). Each frame goes through these steps:

1. The frame is downscaled so its long side is 480–960 px (set under **Detail**) and converted to grayscale.
2. **Camera motion.** Up to 400 corner features are tracked from the previous frame with Lucas–Kanade optical flow, then tracked back again. Points that don't return to where they started are discarded. A RANSAC homography is fitted to the rest. Background points agree with each other, so the fit describes how the camera moved. Points on moving objects are rejected as outliers.
3. The previous frame is warped onto the current one and brightness-matched to it, which cancels the phone's automatic exposure changes. Then the two are subtracted. Anything still different moved by itself.
4. The difference is thresholded (**Sensitivity**) and cleaned with morphology. Nearby blobs are grouped using a padded copy of the mask, but each box is fitted to the exact changed pixels, so it hugs the object. Blobs below **Minimum size** are dropped.
5. A box is shown only if it overlaps a box from the previous frame, which suppresses one-frame flicker.

All pixel-based tuning (kernel sizes, feature spacing, tracking tolerances) is set for a 320 px frame and scales with the processing resolution, so changing **Detail** doesn't change how sensitive the detector is.

Safeguards:

- **Low-texture scenes.** If the points that agree on the camera motion are few or bunched in one area (a plain wall), they are probably on the moving object. The detector then assumes a still camera, so it doesn't erase that object.
- **Fast movement.** If tracking fails because the phone moved too fast, or more than 40% of the frame changed, no boxes are drawn for that frame.

## Run it

```sh
npm start          # serves on http://localhost:8080
npm test           # synthetic-frame tests for the detector (Node, no browser)
```

On a computer with a webcam, `http://localhost:8080` works directly in the browser.

## Test on an iPhone

Safari only allows camera access over **HTTPS**, so the phone needs an `https://` address. Either of these works:

**A. Quick tunnel (no account).** Start `npm start`, then in a second terminal:

```sh
winget install --id Cloudflare.cloudflared   # once
cloudflared tunnel --url http://localhost:8080
```

Open the printed `https://….trycloudflare.com` address in Safari on the phone. Anyone with the link can reach the app while the tunnel runs. Corporate networks sometimes block tunnels; if so, use option B.

**B. GitHub Pages.** Push this folder to a GitHub repository and enable Pages (Settings → Pages → deploy from branch). The app uses relative paths, so it works from a sub-path.

On the phone, tap **Start camera** and allow camera access. To install it like an app, use **Share → Add to Home Screen**; it then opens full screen without the Safari bars. The first load downloads OpenCV.js (about 10 MB). After that, the service worker caches it and the app works offline.

## Files

| File | Role |
| --- | --- |
| `index.html`, `styles.css` | Full-screen camera view, controls, settings sheet |
| `app.js` | Camera, frame capture, worker messaging, box drawing, settings |
| `worker.js` | Loads OpenCV.js and runs the detector off the main thread |
| `detector.js` | The motion-compensated detection pipeline |
| `sw.js`, `manifest.webmanifest`, `icons/` | PWA install and offline support |
| `serve.js` | Dependency-free local static server |
| `test/detector.test.js` | Tests at 320 and 640 px: still/panning camera × object/no object, a plain-wall scene, a sudden exposure change, and box tightness |

## Tuning

- **Sensitivity** lowers the brightness difference that counts as motion. Raise it to catch subtle movement; lower it if lighting flicker causes false boxes.
- **Minimum size** ignores small blobs such as leaves or noise.
- **Detail** sets the processing resolution (long side). Higher resolution catches smaller or more distant movement but uses more battery. The default is High (800 px). If **Show stats** reports under about 20 fps, step down a level.
- **Show stats** shows fps, per-frame time, the number of background points used for camera tracking, and the box count.
