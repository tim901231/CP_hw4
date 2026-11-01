"""Find where in a focus sweep each part of the scene is sharpest.

Usage: python capture/analyze_range.py stack/range
Prints per-shot sharpness for a 3x3 grid of image regions, which shot each
region peaks at, and saves a contact sheet (range_sheet.jpg) in that folder.
"""
import sys
import glob
import os
import numpy as np
import cv2

folder = sys.argv[1] if len(sys.argv) > 1 else "stack/range"
paths = sorted(p for p in glob.glob(os.path.join(folder, "*"))
               if p.lower().endswith((".jpg", ".jpeg", ".png")))
if not paths:
    sys.exit(f"no JPEGs in {folder} (for NEF-only capture, set the camera to NEF+JPEG for this sweep)")

rows, cols = 3, 3
sharp = []
thumbs = []
for p in paths:
    im = cv2.imread(p)
    g = cv2.cvtColor(cv2.resize(im, (1500, 1000)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1), cv2.CV_32F)
    H, W = g.shape
    sharp.append([[lap[r*H//rows:(r+1)*H//rows, c*W//cols:(c+1)*W//cols].var()
                   for c in range(cols)] for r in range(rows)])
    t = cv2.resize(im, (300, 200))
    cv2.putText(t, os.path.basename(p), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    thumbs.append(t)
sharp = np.array(sharp)                     # (shots, rows, cols)

print("brightness of first shot (0-255):", cv2.imread(paths[0], 0).mean().round(1))
print("\nsharpness per shot (rows = shots, columns = regions top-left .. bottom-right):")
for p, s in zip(paths, sharp.reshape(len(paths), -1)):
    print(f"{os.path.basename(p):>16}", " ".join(f"{v:7.1f}" for v in s))
peak = sharp.reshape(len(paths), -1).argmax(0)
print("\nshot index where each region is sharpest:", peak.reshape(rows, cols).tolist())
print(f"focus range used by the scene: shots {peak.min()} .. {peak.max()} of 0 .. {len(paths) - 1}")

while len(thumbs) % 4:
    thumbs.append(np.zeros_like(thumbs[0]))
sheet = np.concatenate([np.concatenate(thumbs[i:i+4], 1) for i in range(0, len(thumbs), 4)], 0)
cv2.imwrite(os.path.join(folder, "range_sheet.jpg"), sheet)
print("contact sheet:", os.path.join(folder, "range_sheet.jpg"))
