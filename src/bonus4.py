"""Bonus 4: confocal stereo + depth from focus on my Nikon D3500 focal-aperture stack.

Run from the repository root:
    python src/bonus4.py                               # process data/own/nikon_af -> results/bonus4/
    python src/bonus4.py prepare NEF_DIR OUT_DIR [--width W]
        decode the NEFs with dcraw to linear 16-bit PNGs (camera WB, sRGB primaries),
        named A<f-number>F<focus index>.png; the report used --width 1504, the committed
        copies in data/own/nikon_af are --width 752
    python src/bonus4.py run --stack DIR --out DIR     # process another prepared stack

All pixel-sized parameters are given for the 1504-pixel-wide stack of the report and are
scaled with the width of the input.
"""
import argparse
import glob
import os
import re
import subprocess

import numpy as np
import cv2
from utils import lRGB2XYZ
from main import gamma_encode, save_img

STACK_DIR = "data/own/nikon_af"
OUT_DIR = "results/bonus4"
APERTURES = ["4", "5.6", "8", "11"]           # wide -> narrow
N_FOCUS = 8                                    # F0 = far (8.41 m) .. F7 = near (0.25 m)
FOCUS_DIST_MM = [8410, 8410, 890, 500, 380, 320, 250, 250]   # EXIF FocusDistance (coarse)
FOCAL_MM = 24.0
REF_WIDTH = 1504                               # width the pixel parameters below are tuned for
REF_FOCUS = 0                                  # most magnified frame (this lens breathes: near
                                               # focus is wider), so others scale up into it
CROP = 0.03                                    # trim warp borders on every side
EQUALIZE_SIGMA = 40                            # smoothness of the per-image gain maps
PRESMOOTH = 1.5                                # blur before warping (confocal stereo only)
DFF_SIGMA_1, DFF_SIGMA_2 = 2, 10
CONFOCAL_DOWNSCALE = 4                         # the captures are ~7 stops under, so average
                                               # pixels down before measuring variance
CONFOCAL_WINDOW = 11
FILL_SIGMA = 25


def prepare(nef_dir, out_dir, width):
    # dcraw: -4 linear 16-bit, -T TIFF, -w camera white balance, -h half size, -o 1 sRGB
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(nef_dir, "*.NEF"))):
        info = subprocess.run(["dcraw", "-i", "-v", path], capture_output=True, text=True).stdout
        f_number = float(re.search(r"Aperture: f/([\d.]+)", info).group(1))
        focus = int(re.search(r"F(\d+)\.NEF$", path).group(1))     # focus index from the file name
        tiff = subprocess.run(["dcraw", "-c", "-4", "-T", "-w", "-h", "-o", "1", path],
                              capture_output=True, check=True).stdout
        im = cv2.imdecode(np.frombuffer(tiff, np.uint8), cv2.IMREAD_UNCHANGED)
        im = cv2.resize(im, (width, round(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)
        name = f"A{f_number:g}F{focus}.png"
        cv2.imwrite(os.path.join(out_dir, name), im)
        print(os.path.basename(path), "->", name)


def load(stack_dir, a, f):
    im = cv2.imread(os.path.join(stack_dir, f"A{a}F{f}.png"), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"no image for f/{a} focus {f} in {stack_dir}")
    return im[..., ::-1].astype(np.float32) / 65535


def to_u8(lin, gain):
    return np.round(np.clip(gamma_encode(np.clip(lin * gain, 0, 1)), 0, 1) * 255).astype(np.uint8)


def lens_scale(dist_mm):
    # Gaussian lens formula: image distance s_i = 1 / (1/f - 1/s_o); image size grows with s_i
    s_i = 1 / (1 / FOCAL_MM - 1 / np.asarray(dist_mm, np.float64))
    return s_i[REF_FOCUS] / s_i                # scale that maps each focus setting onto REF_FOCUS


def estimate_similarity(src, dst):
    # scale + rotation + translation from src to dst using ORB matches and RANSAC
    orb = cv2.ORB_create(5000)
    k1, d1 = orb.detectAndCompute(src, None)
    k2, d2 = orb.detectAndCompute(dst, None)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    p1 = np.float32([k1[m.queryIdx].pt for m in matches])
    p2 = np.float32([k2[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    return M, int(inliers.sum()), len(matches)


def estimate_alignment(stack):
    # one global similarity per focus setting, estimated on the deepest-DoF (f/11) images
    # and applied to every aperture at that focus setting. Neighbouring focus settings look
    # alike, so match each to the previous one and chain the transforms back to REF_FOCUS
    gray = lambda im: cv2.cvtColor(to_u8(im, 0.5 / np.median(im)), cv2.COLOR_RGB2GRAY)
    predicted = lens_scale(FOCUS_DIST_MM)
    transforms = []
    to_ref = np.eye(3)
    for f in range(stack.shape[1]):
        if f != REF_FOCUS:
            M, n_in, n_all = estimate_similarity(gray(stack[-1, f]), gray(stack[-1, f - 1]))
            to_ref = to_ref @ np.vstack([M, [0, 0, 1]])
        else:
            n_in = n_all = 0
        M = to_ref[:2].copy()
        scale = np.hypot(M[0, 0], M[1, 0])
        print(f"focus {f}: scale {scale:.4f} (lens formula {predicted[f]:.4f}), "
              f"shift ({M[0, 2]:+.1f}, {M[1, 2]:+.1f}) px, {n_in}/{n_all} inliers")
        transforms.append(M)
    return transforms


def warp(stack, transforms, presmooth=0):
    # bilinear resampling at a non-integer scale averages noise more where samples fall
    # between pixels, which leaves a periodic pattern in the noise level that differs per
    # focus setting; a small blur before warping makes the noise level uniform again
    A, F, H, W, _ = stack.shape
    out = np.empty_like(stack)
    for a in range(A):
        for f, M in enumerate(transforms):
            im = cv2.GaussianBlur(stack[a, f], (0, 0), presmooth) if presmooth else stack[a, f]
            out[a, f] = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
    cy, cx = int(H * CROP), int(W * CROP)
    return out[:, :, cy:H - cy, cx:W - cx]


def equalize(stack, sigma=40, target=0.18):
    # exposure + vignetting differ per aperture and the room light drifted between shots,
    # so divide every image by a smooth per-pixel, per-channel gain relative to the widest
    # aperture at the same focus setting (heavy blur keeps defocus out of the gain), then
    # scale the whole stack so the f/4 median sits at mid-grey
    blur = lambda im: cv2.GaussianBlur(im, (0, 0), sigma)
    out = stack.copy()
    for a in range(1, stack.shape[0]):
        gains = []
        for f in range(stack.shape[1]):
            gain = blur(stack[a, f]) / (blur(stack[0, f]) + 1e-6)
            out[a, f] = stack[a, f] / gain
            gains.append(np.median(gain))
        print(f"f/{APERTURES[a]}: gain vs f/{APERTURES[0]} per focus "
              + " ".join(f"{g:.3f}" for g in gains)
              + f" (N^2 predicts {(float(APERTURES[0]) / float(APERTURES[a])) ** 2:.3f})")
    return out * (target / np.median(out[0]))


def confocal_depth(stack, window=15):
    # confocal constancy: at the in-focus setting every aperture sees the same colour,
    # so the variance across apertures (summed over RGB) is minimal there; aggregate it
    # over a small window (the paper matches aperture-focus images, not single pixels)
    var = np.var(stack, axis=0).sum(axis=-1)                       # (F, H, W)
    var = np.stack([cv2.blur(v, (window, window)) for v in var])
    depth = np.argmin(var, axis=0)
    # textureless pixels look the same at every focus setting, so their cost curve is flat
    # and the argmin is noise; confidence = how far the minimum dips below the mean cost
    conf = (var.mean(0) - var.min(0)) / (var.mean(0) + 1e-12)
    return depth, conf


def fill_low_confidence(depth, conf, sigma=25, power=4):
    # confidence-weighted smoothing: confident pixels keep their depth, flat regions take
    # the depth of nearby textured pixels
    w = conf ** power
    blur = lambda im: cv2.GaussianBlur(im.astype(np.float32), (0, 0), sigma)
    filled = blur(w * depth) / (blur(w) + 1e-12)
    alpha = np.clip(conf / np.percentile(conf, 90), 0, 1)
    return alpha * depth + (1 - alpha) * filled


def depth_from_focus(imgs, sigma_1=2, sigma_2=10, eps=1e-6):
    # same pipeline as merge() in main.py, on linear images with N_FOCUS slices
    ws = []
    for im in imgs:
        I_lum = lRGB2XYZ(im)[:, :, 1]
        I_high = I_lum - cv2.GaussianBlur(I_lum, (0, 0), sigma_1)
        ws.append(cv2.GaussianBlur(I_high ** 2, (0, 0), sigma_2) + eps)
    ws = np.stack(ws)
    d = np.arange(len(imgs))[:, None, None]
    depth = (d * ws).sum(0) / ws.sum(0)
    aif = (imgs * ws[..., None]).sum(0) / ws.sum(0)[..., None]
    return aif, depth


def save_depth(depth, path):
    # focus index 0 = far -> dark, N_FOCUS - 1 = near -> bright
    norm = np.clip(depth / (N_FOCUS - 1), 0, 1)
    cv2.imwrite(path, cv2.applyColorMap(np.round(norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS))
    save_img(norm, path.replace(".png", "_gray.png"))


def collage(stack, gain, path, scale=0.25):
    rows = [np.concatenate([cv2.resize(to_u8(im, gain), None, fx=scale, fy=scale) for im in row], 1)
            for row in stack]
    save_img(np.concatenate(rows, 0) / 255, path)


def run(stack_dir, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    stack = np.stack([np.stack([load(stack_dir, a, f) for f in range(N_FOCUS)]) for a in APERTURES])
    print("stack", stack.shape)                   # (apertures, focus, H, W, 3)
    px = stack.shape[3] / REF_WIDTH               # scale of the pixel-sized parameters

    collage(stack, 0.18 / np.median(stack[0]), f"{out_dir}/stack_raw.png")

    transforms = estimate_alignment(stack)
    eq = lambda s: equalize(s, sigma=EQUALIZE_SIGMA * px)
    sharp = eq(warp(stack, transforms))                                # for depth from focus / AIF
    smooth = eq(warp(stack, transforms, presmooth=PRESMOOTH * px))     # for confocal stereo
    stack_raw, stack = stack, sharp
    collage(stack, 1.0, f"{out_dir}/stack_aligned_equalized.png")

    # alignment check at f/11: far focus vs near focus, before and after alignment
    before = eq(warp(stack_raw, [np.eye(3)[:2]] * N_FOCUS))
    diff = lambda s: np.abs(s[-1, 0] - s[-1, -1]).mean(-1)
    save_img(np.clip(np.concatenate([diff(before), diff(stack)], 1) * 4, 0, 1),
             f"{out_dir}/align_residual_before_after.png")
    del before

    ds = max(1, round(CONFOCAL_DOWNSCALE * px))
    small = np.stack([np.stack([cv2.resize(im, None, fx=1 / ds, fy=1 / ds, interpolation=cv2.INTER_AREA)
                                for im in row]) for row in smooth])
    depth_cs, conf = confocal_depth(small, window=CONFOCAL_WINDOW)
    filled = fill_low_confidence(depth_cs, conf, sigma=FILL_SIGMA * px / ds)
    up = lambda im, interp: cv2.resize(im.astype(np.float32), smooth.shape[3:1:-1], interpolation=interp)
    save_depth(up(depth_cs, cv2.INTER_NEAREST), f"{out_dir}/confocal_depth.png")
    save_img(np.clip(up(conf, cv2.INTER_LINEAR) / np.percentile(conf, 99), 0, 1),
             f"{out_dir}/confocal_confidence.png")
    save_depth(up(filled, cv2.INTER_LINEAR), f"{out_dir}/confocal_depth_filled.png")

    aif, depth_dff = depth_from_focus(stack[0], DFF_SIGMA_1 * px, DFF_SIGMA_2 * px)  # widest aperture
    save_img(to_u8(aif, 1.0) / 255, f"{out_dir}/all_in_focus.png")
    save_depth(depth_dff, f"{out_dir}/dff_depth.png")
    print("saved to", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("prepare", help="decode NEFs to linear 16-bit PNGs")
    p.add_argument("nef_dir")
    p.add_argument("out_dir")
    p.add_argument("--width", type=int, default=REF_WIDTH)
    r = sub.add_parser("run", help="process a prepared stack (default command)")
    r.add_argument("--stack", default=STACK_DIR)
    r.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.nef_dir, args.out_dir, args.width)
    else:
        run(getattr(args, "stack", STACK_DIR), getattr(args, "out", OUT_DIR))
