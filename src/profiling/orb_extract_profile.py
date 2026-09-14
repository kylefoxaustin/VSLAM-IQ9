#!/usr/bin/env python3
# Measure ORB feature-extraction cost at ORB-SLAM3 settings on EuRoC frames,
# to size the offloadable front-end vs total tracking time. Run on the board.
import cv2, glob, numpy as np, time, sys
frames = sorted(glob.glob("/opt/orbslam/data/MH_01/mav0/cam0/data/*.png"))[500:600]
imgs = [cv2.imread(f, 0) for f in frames]
orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, fastThreshold=20, edgeThreshold=19)
for im in imgs[:10]: orb.detectAndCompute(im, None)          # warm
def med(fn):
    t = []
    for im in imgs:
        s = time.perf_counter(); fn(im); t.append((time.perf_counter()-s)*1e3)
    return float(np.median(t))
ext = med(lambda im: orb.detectAndCompute(im, None))
det = med(lambda im: orb.detect(im, None))
print(f"ORB detectAndCompute (extract) median = {ext:.3f} ms")
print(f"  detect-only (pyramid+FAST+orient) = {det:.3f} ms | descriptor ~ {ext-det:.3f} ms")
