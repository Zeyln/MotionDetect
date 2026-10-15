# MotionDetect

A phone web app (PWA) that shows the camera and draws red boxes around anything moving in the shot. It runs whenever the camera view is open, without recording, and it compensates for the phone itself moving.

## How detection works

`detector.js` runs in a Web Worker using OpenCV.js (`vendor/opencv.js`, 4.10, about 10 MB). Each frame goes through these steps:

1. The frame is downscaled so its long side is 480–960 px (set under **Detail**) and converted to grayscale.
2. **Camera motion.** Up to 400 corner features are tracked from the previous frame with Lucas–Kanade optical flow, then tracked back again. Points that don't return to where they started are discarded. A RANSAC homography is fitted to the rest. Background points agree with each other, so the fit describes how the camera moved. Points on moving objects are rejected as outliers.
3. The previous frame is warped onto the current one and brightness-matched to it, which cancels the phone's automatic exposure changes. A pixel counts as changed only if it falls outside the brightness range of its 3×3 neighbourhood in that aligned frame. This forgives up to 1 px of misalignment, which would otherwise flicker on every sharp edge. Anything still different moved by itself.
4. **Three-frame confirmation.** The frame before last is aligned too, using the camera motion over both steps. A blob is kept only if some of its pixels also differ from that frame, and its box is fitted to those pixels. The spot an object has just left looks the same as two frames ago, so it doesn't leave a ghost box behind, and a glitch in a single frame (a reflection, a compression artifact) is ignored.
5. The difference is thresholded (**Sensitivity**) and cleaned with morphology. Nearby blobs are grouped using a padded copy of the mask, but each box is fitted to the exact changed pixels, so it hugs the object. Blobs below **Minimum size** are dropped.
6. **Target tracking.** Each box is matched to a known target by its predicted position (last position plus velocity). A target is shown once it has been seen in three frames, which filters out short flickers. If a moving target goes unseen for a moment (it blends into the background, or the frame is blurred), the tracker follows its predicted, slowing path for up to 2 frames, so the same target picks it up again instead of a new one. The predicted box is only drawn while it is less than the target's own size from where it was last seen (it still overlaps the target); beyond that nothing is drawn until the target is seen again or held. A target whose path leads out of the frame is dropped rather than held at the edge.
7. **Held targets.** When a target stops moving, its box stays, drawn dashed, for the **Keep boxes after stopping** time. Only targets that were followed for at least 5 frames are held, so short-lived fragments and flickers just disappear. Each frame the box is moved by the camera's motion, so it stays on the target while you pan. If the target moves off again, the box goes solid and follows it. Small flickers inside a held box don't shrink it or create new targets.

Kernel sizes, feature spacing and tracking tolerances are set for a 320 px frame and scale with the processing resolution. The noise filter and the misalignment allowance stay at 3×3 px at every resolution, so small and thin movers (a distant bird, a mouse cursor) aren't erased.

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
| `test/detector.test.js` | Tests at 480 and 640 px with sensor noise: still/panning camera × object/no object, a plain-wall scene, a sudden exposure change, a tiny object, a fast small object (no trail of boxes), a fast object that stops dead (no box running ahead), one-frame glitches, targets that stop (held boxes, also while panning), hold expiry, and box tightness |

## Tuning

- **Sensitivity** lowers the brightness difference that counts as motion. Raise it to catch subtle movement; lower it if lighting flicker causes false boxes.
- **Minimum size** ignores small blobs such as leaves or noise. At the default (3), anything about 6×6 px in the processed frame counts; lower it to catch even smaller specks.
- **Keep boxes after stopping** sets how long a box stays on a target after it stops moving (0–10 s, default 3 s; 0 turns holding off). Held boxes are dashed.
- **Detail** sets the processing resolution (long side). Higher resolution catches smaller or more distant movement but uses more battery. The default is High (800 px). If **Show stats** reports under about 20 fps, step down a level.
- **Sync video with boxes** (on by default) shows each analysed frame together with its boxes, so they line up exactly with the picture. The picture then runs one frame behind the live camera and updates at the detector's frame rate (the fps in **Show stats**). Turn it off for the smoother live picture, with boxes trailing fast movers slightly.
- **Show stats** shows fps, per-frame time, the number of background points used for camera tracking, and how many targets are moving and held.
