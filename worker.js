// Runs the detector off the main thread so the camera preview stays smooth.
// Messages in:  {type:'config', options} | {type:'reset'} | {type:'frame', buffer, width, height}
// Messages out: {type:'ready'} | {type:'error', message} | {type:'result', boxes, heat, mode, features, inliers, ms}
'use strict';

importScripts('vendor/opencv.js', 'imgproc.js', 'detector.js');

let detector = null;
let pendingOptions = {};

// The OpenCV.js module has a `then` method, so it is never passed to resolve() directly.
function whenCvReady() {
  return new Promise((resolve) => {
    if (self.cv.Mat) resolve();
    else self.cv.onRuntimeInitialized = () => resolve();
  });
}

whenCvReady()
  .then(() => {
    detector = createDetector(self.cv, pendingOptions);
    postMessage({ type: 'ready' });
  })
  .catch((e) => postMessage({ type: 'error', message: String(e) }));

onmessage = ({ data: msg }) => {
  if (msg.type === 'config') {
    if (detector) detector.configure(msg.options);
    else Object.assign(pendingOptions, msg.options);
  } else if (msg.type === 'reset') {
    if (detector) detector.reset();
  } else if (msg.type === 'frame' && detector) {
    const t0 = performance.now();
    try {
      const r = detector.processFrame(new Uint8ClampedArray(msg.buffer), msg.width, msg.height, t0);
      postMessage({ type: 'result', ...r, ms: performance.now() - t0 }, r.heat ? [r.heat.data.buffer] : []);
    } catch (e) {
      // OpenCV.js throws numeric pointers for C++ exceptions.
      const message =
        typeof e === 'number' && self.cv.exceptionFromPtr ? self.cv.exceptionFromPtr(e).msg : String(e);
      postMessage({ type: 'error', message });
    }
  }
};
