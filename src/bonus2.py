"""Bonus 2: depth from the Problem 1 focal stack by blur estimation and model fitting,
adapted from Suwajanakorn et al., "Depth from Focus with Your Mobile Phone" (CVPR 2015).

Run from the repository root, after `python src/main.py p1-refocus`:
python src/bonus2.py                 # known d, square PSF -> results/bonus2/refocus/
python src/bonus2.py --model paper   # thin-lens Eq. 10, disk PSF -> results/bonus2/thin_lens/

Refocus mode returns normalized refocus coordinates, not metric depth.
Both modes omit MRF stitching, bokeh correction and spatial depth refinement.
"""
from pathlib import Path
import os
import warnings

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace
from scipy.optimize import least_squares, minimize_scalar
from scipy.sparse import coo_matrix


def all_in_focus(imgs):
    """Select the sharpest frame per pixel, keeping its RGB values."""
    gray = imgs @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    sharpness = np.stack([
        gaussian_filter(np.abs(gaussian_laplace(im, 1)), 2) for im in gray
    ])
    index = sharpness.argmax(axis=0)
    y, x = np.indices(index.shape)
    return imgs[index, y, x]


def disk_psf(radius):
    """Antialiased disk so fractional radius steps change the kernel."""
    if radius == 0:
        return np.ones((1, 1), dtype=np.float32)
    k = int(np.ceil(radius + 0.5))
    offsets = (np.arange(8) + 0.5) / 8 - 0.5
    x = np.arange(-k, k + 1)[:, None] + offsets
    disk = (x[:, None, :, None]**2 + x[None, :, None, :]**2 <= radius**2)
    kernel = disk.mean(axis=(2, 3))
    return (kernel / kernel.sum()).astype(np.float32)


def square_psf(radius):
    """Uniform square with pixel-area antialiasing; radius is half-width."""
    if radius == 0:
        return np.ones((1, 1), np.float32)
    k = int(np.ceil(radius + .5))
    x = np.arange(-k, k + 1)
    weights = np.maximum(0, np.minimum(x + .5, radius) - np.maximum(x - .5, -radius))
    weights = weights / weights.sum()
    return np.outer(weights, weights).astype(np.float32)


def estimate_blur(imgs, aif, radii=None, sigma=15, psf="disk"):
    """Eqs. 7-9: RGB absolute difference, spatial averaging, radius/confidence.

    Radii are in the input image's pixels. No alignment scale correction is
    needed for this already aligned stack. Confidence is not a probability.
    sigma=15 matches Section 8. Filters use reflected image boundaries.
    """
    if psf not in ("disk", "square"):
        raise ValueError("psf must be disk or square")
    make_kernel = disk_psf if psf == "disk" else square_psf
    if radii is None:
        radii = np.arange(0, 6.5 + 0.125, 0.25)
    radii = np.asarray(radii)
    shape = imgs.shape[:3]
    best = np.full(shape, np.inf)
    total = np.zeros(shape)
    B = np.zeros(shape)
    for radius in radii:
        blurred = cv2.filter2D(aif, -1, make_kernel(radius))
        for i, img in enumerate(imgs):
            cost = gaussian_filter(np.abs(img - blurred).mean(axis=-1), sigma)
            better = cost < best[i]
            B[i][better] = radius
            best[i] = np.minimum(best[i], cost)
            total[i] += cost
    gap = np.maximum(total / len(radii) - best, 0)
    C = gap**2                          # Eq. 9, alpha = 2; no hard rejection.
    # One global scale preserves the minimizer of Eq. 10.
    if C.max() > 0:
        C /= C.max()
    return B, C


def predict_blur(A, F, f, depth):
    """Eq. 4: A in effective pixel units; F, f and depth in common units."""
    f = np.asarray(f, dtype=np.float64)[:, None]
    s = np.asarray(depth, dtype=np.float64)[None, :]
    return A * np.abs(f - s) / s * F / (f - F)


def fit_eq10(B, C, near_focus=10.0, far_focus=32.0, max_nfev=1000):
    """Fit relative depth with fixed endpoint focus distances and ordered f.

    Input frame order defines the relative depth direction. The scene is
    assumed to lie within the focus range. Only pixels whose confidence is
    zero in EVERY frame return NaN: Eq. 10 cannot constrain their depth.
    """
    B, C = np.asarray(B, float), np.asarray(C, float)
    if B.ndim != 3 or B.shape != C.shape or B.shape[0] < 3:
        raise ValueError("B and C must have shape (N,H,W), N >= 3")
    if not (np.isfinite(B).all() and np.isfinite(C).all()) or np.any(B < 0) or np.any(C < 0):
        raise ValueError("B and C must be finite and nonnegative")
    if not 0 < near_focus < far_focus:
        raise ValueError("Require 0 < near_focus < far_focus")
    N, H, W = B.shape
    valid = np.any(C > 0, axis=0)  # No confidence cutoff or minimum frame count.
    if not valid.any():
        raise ValueError("All confidence values are zero; Eq. 10 has no depth information")
    Bf, Cf = B[:, valid], C[:, valid]
    P = Bf.shape[1]
    f_init = np.linspace(near_focus, far_focus, N)
    depth_init = f_init[np.argmin(np.where(Cf > 0, Bf, np.inf), axis=0)]
    F_init = near_focus * 0.2
    unit = predict_blur(1, F_init, f_init, depth_init)
    A_init = np.sum(Cf**2 * unit * Bf) / max(np.sum(Cf**2 * unit**2), 1e-12)

    # [A, F, N-2 log-gap parameters, P depths]. Positive normalized gaps
    # guarantee ordered focus distances, without an extra penalty term.
    def unpack(x):
        gaps = np.exp(np.r_[x[2:N], 0.0])
        f = near_focus + (far_focus - near_focus) * np.r_[0, np.cumsum(gaps) / gaps.sum()]
        return x[0], x[1], f, x[N:]

    def residual(x):
        A, F, f, s = unpack(x)
        return ((predict_blur(A, F, f, s) - Bf) * Cf).ravel()

    # All global parameters and only the corresponding pixel depth.
    rows = np.arange(N * P)
    jac_sparsity = coo_matrix((
        np.ones(N * P * (N + 1)),
        (np.r_[np.repeat(rows, N), rows],
         np.r_[np.tile(np.arange(N), N * P), N + np.tile(np.arange(P), N)])
    ), shape=(N * P, N + P)).tocsr()
    x0 = np.r_[np.clip(A_init, 0.01, 999), F_init, np.zeros(N - 2), depth_init]
    lower = np.r_[1e-4, near_focus * 1e-4, np.full(N-2, -8), np.full(P, near_focus)]
    upper = np.r_[1000, near_focus * 0.99, np.full(N-2, 8), np.full(P, far_focus)]
    result = least_squares(residual, x0, bounds=(lower, upper),
                           jac_sparsity=jac_sparsity, method="trf",
                           x_scale="jac", max_nfev=max_nfev)
    if not result.success:
        warnings.warn("Eq. 10 did not converge: " + result.message)
    A, F, f, s = unpack(result.x)
    depth = np.full((H, W), np.nan)
    depth[valid] = s
    return {"depth": depth, "aperture": A, "focal_length": F,
            "focus_distances": f, "loss": np.sum(result.fun**2),
            "success": result.success, "solver": result}


def solve_pixel_depths(B, C, positions, slopes, bounds):
    """Global per-pixel minimum of sum C²*(k_i*|t-t_i|-B_i)².

    Between consecutive focus settings the model is linear in t. Solve each
    interval exactly, then keep the lowest cost. No confidence threshold.
    """
    n, h, w = B.shape
    b, c = B.reshape(n, -1), C.reshape(n, -1)
    positions = np.asarray(positions)
    slopes = np.broadcast_to(slopes, (n,))
    breaks = np.unique(np.r_[bounds[0], positions, bounds[1]])
    breaks = breaks[(breaks >= bounds[0]) & (breaks <= bounds[1])]
    answer = np.full(h * w, np.nan)
    for start in range(0, h * w, 16384):
        stop = min(start + 16384, h * w)
        values, weights = b[:, start:stop], c[:, start:stop]**2
        wk = weights * slopes[:, None]**2
        denominator = wk.sum(axis=0)
        best = np.full(stop - start, np.inf)
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            sign = np.where((lo + hi) / 2 >= positions, 1., -1.)
            target = positions[:, None] + sign[:, None] * values / slopes[:, None]
            t = np.clip((wk * target).sum(axis=0) / np.maximum(denominator, 1e-300), lo, hi)
            error = (weights * (slopes[:, None] * abs(t - positions[:, None]) - values)**2).sum(axis=0)
            improve = (error < best) & (denominator > 0)
            best[improve] = error[improve]
            answer[start:stop][improve] = t[improve]
    return answer.reshape(h, w)


def calibration_sample(B, C, count):
    """Small reproducible interior sample for global parameters; not a depth mask."""
    b, c = B[:, ::4, ::4], C[:, ::4, ::4]
    y, x = np.mgrid[:b.shape[1], :b.shape[2]] * 4
    interior = (y >= 64) & (y < B.shape[1]-64) & (x >= 64) & (x < B.shape[2]-64)
    if not interior.any():
        interior = np.ones(b.shape[1:], dtype=bool)
    indices = np.flatnonzero(interior.ravel())
    indices = np.random.default_rng(12).choice(indices, min(count, len(indices)), replace=False)
    return b.reshape(len(B), -1)[:, indices, None], c.reshape(len(C), -1)[:, indices, None]


def fit_refocus(B, C, d):
    """Known light-field refocus settings: radius_i = k * |d_i - d_pixel|.

    d_pixel is a refocus/disparity coordinate, NOT physical distance.
    Fits one global scale on a small sample, then solves every output pixel.
    """
    b, c = calibration_sample(B, C, 800)
    if not np.any(c > 0):
        raise ValueError("Calibration sample has no blur information")
    bounds = (float(np.min(d)), float(np.max(d)))
    def objective(k):
        t = solve_pixel_depths(b, c, d, k, bounds)
        return np.nansum((c * (k * abs(t[None] - d[:, None, None]) - b))**2)
    solver = minimize_scalar(objective, bounds=(3, 20), method="bounded",
                             options={"xatol": 1e-4})
    focus = solve_pixel_depths(B, C, d, solver.x, bounds)
    return {"refocus_d": focus, "depth": (bounds[1] - focus) / (bounds[1] - bounds[0]),
            "scale": float(solver.x), "success": bool(solver.success), "solver": solver}


def main():
    import argparse
    import json
    import time
    from main import read_focus_stack, save_img

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["refocus", "paper"], default="refocus")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    start = time.perf_counter()
    imgs = read_focus_stack()[::-1].copy()  # d_15 near -> d_0 far; keep full resolution.
    d = 0.1 * np.arange(len(imgs)-1, -1, -1)
    aif = all_in_focus(imgs)
    # Selected by two frame-holdout comparisons; see experiment report.
    psf, radius_max, sigma = ("square", 12, 2) if args.model == "refocus" else ("disk", 16, 3)
    print(f"Matching blur: {psf} PSF, radius 0-{radius_max}, sigma={sigma}...", flush=True)
    B, C = estimate_blur(imgs, aif, radii=np.arange(0, radius_max + .125, .25), sigma=sigma, psf=psf)
    print("Fitting global parameters, then every depth pixel...", flush=True)
    if args.model == "refocus":
        output = fit_refocus(B, C, d)
        predicted = output["scale"] * abs(output["refocus_d"][None] - d[:, None, None])
        preview = output["depth"]
        units = "normalized refocus coordinate; 0 near, 1 far; NOT metric depth"
        parameters = {"scale": output["scale"], "d_settings": d.tolist()}
    else:
        b, c = calibration_sample(B, C, 400)
        output = fit_eq10(b, c, max_nfev=2000)
        A, F, f = output["aperture"], output["focal_length"], output["focus_distances"]
        q = solve_pixel_depths(B, C, 1/f, A/(1/F-1/f), (1/32, 1/10))
        output["depth"] = 1/q
        predicted = predict_blur(A, F, f, output["depth"].ravel()).reshape(B.shape)
        preview = (output["depth"] - 10) / 22
        units = "relative thin-lens depth in arbitrary 10-32 coordinates; NOT metric"
        parameters = {"aperture": A, "focal_length": F, "focus_distances": f.tolist()}
    valid = np.isfinite(output["depth"])
    error = predicted[:, valid] - B[:, valid]
    confidence = C[:, valid]
    weighted_rmse = float(np.sqrt(np.sum((confidence * error)**2) / np.sum(confidence**2)))
    rmse = float(np.sqrt(np.mean(error[confidence > 0]**2)))
    folder = Path("results/bonus2/refocus" if args.model == "refocus" else "results/bonus2/thin_lens")
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "depth.npy", output["depth"])
    if args.model == "refocus":
        np.save(folder / "refocus_d.npy", output["refocus_d"])
    np.savez_compressed(folder / "measurements.npz", B=B, C=C, model=args.model,
                        depth_units=units, success=output["success"], **parameters)
    save_img(aif, str(folder / "all_in_focus.png"))
    save_img(np.nan_to_num(np.clip(preview, 0, 1)), str(folder / "depth.png"))
    support = np.sqrt(np.sum(C*C, axis=0))
    save_img(support / max(support.max(), 1e-12), str(folder / "confidence.png"))
    report = dict(model=args.model, units=units, psf=psf, radius_max=radius_max, sigma=sigma,
                  shape=list(output["depth"].shape), success=output["success"],
                  solver_message=str(output["solver"].message), parameters=parameters,
                  weighted_blur_rmse_px=weighted_rmse, unweighted_blur_rmse_px=rmse,
                  depth_coverage=float(valid.mean()), seconds=time.perf_counter()-start)
    (folder / "fit_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("Saved to", folder.resolve())


if __name__ == "__main__":
    main()
