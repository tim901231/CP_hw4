"""Generate the report figures for HW4.

Run from the repository root, after `python src/main.py all`, `python src/bonus2.py`,
`python src/bonus2.py --model paper` and `python src/bonus4.py` have produced their outputs:
    python src/report_figures.py

Outputs PNG figures to report/figures/.
"""
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(__file__))
from main import (D_VALUES, APERTURES, MOSAIC_FULL, TEMPLATES, SEARCH_R,
                  P3_DIR, read_focus_stack, read_af_stack, linear_imgs, gamma_encode, merge, ncc,
                  read_video, LinearFrames, shifts_path)
from utils import lRGB2XYZ

FIG_DIR = "report/figures"

# Part 1
REFOCUS_SHOW = [0, 3, 6, 9, 12, 15]         # stack indices shown as refocusings
SWEEP_SIGMA_1 = [1, 2, 4]
SWEEP_SIGMA_2 = [2, 6, 12]
AFI_PIXELS = {"Front white pawn": (290, 505), "Black piece (back)": (60, 230),
              "Middle white pawn": (150, 372)}          # (row, col)
AFI_COLORS = ["#ffcc33", "#33ccff", "#ff6680"]

# Part 3
TEMPLATE_LABELS = {"near": "Near figure", "middle": "Middle figure", "back": "Back figure"}
TEMPLATE_COLORS = ["#ffcc33", "#33ccff", "#ff6680"]
NCC_DEMO_FRAME = 60

# Part 4
NIKON_OUT = "results/bonus4"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.titlepad": 3,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name))
    plt.close(fig)
    print("saved", name)


def show(ax, img, title=None, cmap=None, vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="lanczos" if cmap is None else "nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)


def grid(images, titles, ncols, name, panel_w=3.0, cmap=None, vmin=None, vmax=None, suptitle=None):
    nrows = int(np.ceil(len(images) / ncols))
    h, w = images[0].shape[:2]
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_w * ncols * h / w / ncols * nrows + 0.3 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, im, t in zip(axes.ravel(), images, titles):
        ax.axis("on")
        show(ax, im, t, cmap, vmin, vmax)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout(pad=0.3)
    save(fig, name)


def rgb(path):
    return cv2.imread(path)[..., ::-1]


# ------------------------------------------------------------------ Part 1 ----
def part1():
    # mosaic of sub-aperture views (full file is ~11200 x 6400; read reduced)
    mosaic = cv2.imread(MOSAIC_FULL, cv2.IMREAD_REDUCED_COLOR_4)[..., ::-1]
    H, W = mosaic.shape[:2]
    vh, vw = H // 16, W // 16
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw=dict(width_ratios=[W / H, 3 * vw / (3 * vh)]))
    show(axes[0], mosaic, "All 16 x 16 sub-aperture views (rows: u, columns: v)")
    axes[0].add_patch(Rectangle((6 * vw, 6 * vh), 3 * vw, 3 * vh, fill=False, ec="#ffcc33", lw=1.2))
    centre = cv2.imread(MOSAIC_FULL)[..., ::-1]
    cH, cW = centre.shape[0] // 16, centre.shape[1] // 16
    show(axes[1], centre[6 * cH:9 * cH, 6 * cW:9 * cW], "Zoom: views u, v = 6..8")
    fig.tight_layout(pad=0.3)
    save(fig, "p1_mosaic.png")
    del centre

    # refocusing
    stack = read_focus_stack()
    grid([stack[i] for i in REFOCUS_SHOW], [f"d = {D_VALUES[i]:.1f}" for i in REFOCUS_SHOW], 3,
         "p1_refocus.png", panel_w=3.3)

    # depth from focus: sigma sweep
    aifs, depths, titles = [], [], []
    for s2 in SWEEP_SIGMA_2:
        for s1 in SWEEP_SIGMA_1:
            aif, depth = merge(stack, sigma_1=s1, sigma_2=s2)
            aifs.append(np.clip(aif, 0, 1)); depths.append(depth)
            titles.append(rf"$\sigma_1={s1}$, $\sigma_2={s2}$")
    grid(depths, titles, 3, "p1_dff_sweep_depth.png", panel_w=3.3, cmap="gray", vmin=0, vmax=1)
    grid(aifs, titles, 3, "p1_dff_sweep_aif.png", panel_w=3.3)

    aif, depth = merge(stack)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.0))
    show(axes[0], np.clip(aif, 0, 1), "All-in-focus image")
    show(axes[1], depth, "Depth map (brighter: smaller d)", cmap="gray", vmin=0, vmax=1)
    fig.tight_layout(pad=0.3)
    save(fig, "p1_dff_final.png")

    # focal-aperture stack collage (every other focus setting)
    af = read_af_stack()                                       # (8 apertures, 16 d, H, W, 3), sRGB
    cols = list(range(0, 16, 2))
    h, w = af.shape[2:4]
    thumbs = [[cv2.resize(af[a, d], (w // 3, h // 3), interpolation=cv2.INTER_AREA) for d in cols]
              for a in range(len(APERTURES))]
    collage = np.concatenate([np.concatenate(r, 1) for r in thumbs], 0)
    fig, ax = plt.subplots(figsize=(10, 10 * collage.shape[0] / collage.shape[1]))
    ax.imshow(np.clip(collage, 0, 1))
    th, tw = h // 3, w // 3
    ax.set_xticks([tw * (i + 0.5) for i in range(len(cols))])
    ax.set_xticklabels([f"d={D_VALUES[d]:.1f}" for d in cols])
    ax.set_yticks([th * (i + 0.5) for i in range(len(APERTURES))])
    ax.set_yticklabels([f"a={a} ({4 * a * a} views)" for a in APERTURES])
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.3)
    save(fig, "p1_af_collage.png")

    # AFIs and confocal depth
    lin = linear_imgs(af)
    var = np.var(lin, axis=0).sum(axis=-1)                    # (16, H, W)
    best = np.argmin(var, axis=0)
    fig = plt.figure(figsize=(10, 4.6))
    gs = fig.add_gridspec(len(AFI_PIXELS), 3, width_ratios=[2.2, 1.6, 1.3])
    ax = fig.add_subplot(gs[:, 0])
    show(ax, np.clip(aif, 0, 1), "Selected pixels")
    for (name, (r, c)), col in zip(AFI_PIXELS.items(), AFI_COLORS):
        ax.plot(c, r, "o", mfc="none", mec=col, ms=9, mew=1.6)
    for k, ((name, (r, c)), col) in enumerate(zip(AFI_PIXELS.items(), AFI_COLORS)):
        a1 = fig.add_subplot(gs[k, 1])
        a1.imshow(np.clip(af[:, :, r, c], 0, 1), interpolation="nearest", aspect="auto")
        a1.axvline(best[r, c], color="w", lw=1, ls="--")
        a1.set_yticks([0, 7]); a1.set_yticklabels(["a=1", "a=8"])
        a1.set_xticks([0, 15]); a1.set_xticklabels(["d=0", "d=1.5"])
        a1.set_title(f"AFI: {name}", color="k")
        for s in a1.spines.values():
            s.set_edgecolor(col); s.set_linewidth(2)
        a2 = fig.add_subplot(gs[k, 2])
        a2.plot(D_VALUES, var[:, r, c], color="k", lw=1)
        a2.axvline(D_VALUES[best[r, c]], color=col, lw=1.5, ls="--")
        a2.set_title("variance across apertures"); a2.set_xlabel("d", labelpad=0)
        a2.tick_params(labelsize=7)
    fig.tight_layout(pad=0.4)
    save(fig, "p1_afi.png")

    conf_depth = -best.astype(np.float32)
    conf_depth = (conf_depth - conf_depth.min()) / (conf_depth.max() - conf_depth.min())
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.1))
    show(axes[0], depth, "Depth from focus", cmap="gray", vmin=0, vmax=1)
    show(axes[1], conf_depth, "Confocal stereo", cmap="gray", vmin=0, vmax=1)
    show(axes[2], cv2.medianBlur((conf_depth * 255).astype(np.uint8), 5), "Confocal stereo, 5x5 median", cmap="gray")
    fig.tight_layout(pad=0.3)
    save(fig, "p1_confocal_vs_dff.png")
    return stack, aif, depth


# ------------------------------------------------- Bonus 2 (better blending) ----
def bonus2(aif, depth):
    ref = rgb("results/bonus2/refocus/all_in_focus.png") / 255
    pap = rgb("results/bonus2/thin_lens/all_in_focus.png") / 255
    imgs = [np.clip(aif, 0, 1), ref,
            depth, cv2.imread("results/bonus2/refocus/depth.png", 0) / 255,
            cv2.imread("results/bonus2/thin_lens/depth.png", 0) / 255, np.sqrt(cv2.imread("results/bonus2/refocus/confidence.png", 0) / 255)]
    titles = ["Part 1 all-in-focus (weighted)", "Bonus all-in-focus (sharpest frame)",
              "Part 1 depth", "Bonus depth: refocus model", "Bonus depth: thin-lens model",
              "Blur-estimate confidence (square root)"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 9.0))
    for ax, im, t in zip(axes.ravel(), imgs, titles):
        show(ax, im, t, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=1)
    fig.tight_layout(pad=0.3)
    save(fig, "b2_compare.png")

    # zoom on the front pieces
    y0, y1, x0, x1 = 170, 330, 440, 640
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    show(axes[0], np.clip(aif, 0, 1)[y0:y1, x0:x1], "Part 1 (weighted)")
    show(axes[1], ref[y0:y1, x0:x1], "Bonus (sharpest frame)")
    show(axes[2], pap[y0:y1, x0:x1], "Bonus, thin-lens run")
    fig.tight_layout(pad=0.3)
    save(fig, "b2_zoom.png")


# ------------------------------------------------------------------ Part 3 ----
def part3():
    frames = read_video()
    lin = LinearFrames(frames)
    gray = {i: lRGB2XYZ(lin[i])[:, :, 1] for i in (0, NCC_DEMO_FRAME)}

    idx = np.linspace(0, len(frames) - 1, 5).astype(int)
    grid([frames[i] for i in idx], [f"frame {i}" for i in idx], 5, "p3_frames.png", panel_w=2.0)

    results, all_shifts = {}, {}
    for key in TEMPLATES:
        z = np.load(shifts_path(key))
        all_shifts[TEMPLATE_LABELS[key]] = (z["shifts"], z["peaks"])
        results[TEMPLATE_LABELS[key]] = rgb(f"{P3_DIR}/refocus_{key}.png")

    # NCC demo on one frame for the first template
    y1, x1, y2, x2 = TEMPLATES["near"]
    shifts, _ = all_shifts[TEMPLATE_LABELS["near"]]
    g = gray[0][y1:y2, x1:x2]
    th, tw = g.shape
    py, px = (y1, x1) if NCC_DEMO_FRAME == 0 else (y1 + shifts[NCC_DEMO_FRAME - 1][0], x1 + shifts[NCC_DEMO_FRAME - 1][1])
    top, left = max(py - SEARCH_R, 0), max(px - SEARCH_R, 0)
    win = gray[NCC_DEMO_FRAME][top:py + th + SEARCH_R, left:px + tw + SEARCH_R]
    h = ncc(win, g)
    # a wide map over a large neighbourhood shows why the search must stay local
    R = 250
    T0, L0 = max(py - R, 0), max(px - R, 0)
    wide = ncc(gray[NCC_DEMO_FRAME][T0:py + th + R, L0:px + tw + R], g)
    disp = lambda im: np.clip(gamma_encode(im), 0, 1)
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.3), gridspec_kw=dict(width_ratios=[1080 / 1920, 0.6, 1, 1]))
    show(axes[0], frames[0], "Frame 0 and template")
    axes[0].add_patch(Rectangle((x1, y1), tw, th, fill=False, ec=TEMPLATE_COLORS[0], lw=1.2))
    show(axes[1], disp(g), "Template g", cmap="gray")
    show(axes[2], wide, f"NCC over a wide area (frame {NCC_DEMO_FRAME})", cmap="viridis", vmin=-1, vmax=1)
    axes[2].add_patch(Rectangle((left - L0, top - T0), h.shape[1], h.shape[0], fill=False, ec="w", lw=1, ls="--"))
    wy, wx = np.unravel_index(np.argmax(h), h.shape)
    axes[2].plot(left - L0 + wx, top - T0 + wy, "r+", ms=10)
    show(axes[3], h, f"NCC in the search window (peak {h.max():.3f})", cmap="viridis", vmin=-1, vmax=1)
    axes[3].plot(wx, wy, "r+", ms=10)
    fig.tight_layout(pad=0.3)
    save(fig, "p3_ncc.png")

    # template trajectories
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.6))
    for (name, (s, p)), col in zip(all_shifts.items(), TEMPLATE_COLORS):
        axes[0].plot(s[:, 1], s[:, 0], "-", color=col, lw=1.2, label=name)
        axes[1].plot(p, color=col, lw=1.2, label=name)
    axes[0].invert_yaxis(); axes[0].set_aspect("equal")
    axes[0].set_xlabel("shift x (px)"); axes[0].set_ylabel("shift y (px)")
    axes[0].set_title("Template position relative to frame 0")
    axes[1].set_xlabel("frame"); axes[1].set_ylabel("peak NCC"); axes[1].set_title("Peak NCC per frame")
    axes[1].legend(fontsize=7)
    fig.tight_layout(pad=0.3)
    save(fig, "p3_shifts.png")

    # refocusing results
    fig, axes = plt.subplots(1, 4, figsize=(10, 4.6))
    show(axes[0], frames[0], "Frame 0 and templates")
    for (y1, x1, y2, x2), col in zip(TEMPLATES.values(), TEMPLATE_COLORS):
        axes[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=col, lw=1.2))
    for ax, ((name, img), (y1, x1, y2, x2), col) in zip(axes[1:], zip(results.items(), TEMPLATES.values(),
                                                                      TEMPLATE_COLORS)):
        show(ax, img, f"Focused on: {name}")
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=col, lw=1.0))
    fig.tight_layout(pad=0.3)
    save(fig, "p3_refocus.png")

    # zoom on the three figures in each result
    y0, y1_, x0, x1_ = 600, 1200, 100, 1080
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.3))
    for ax, ((name, img), col) in zip(axes, zip(results.items(), TEMPLATE_COLORS)):
        show(ax, img[y0:y1_, x0:x1_], f"Focused on: {name}")
    fig.tight_layout(pad=0.3)
    save(fig, "p3_refocus_zoom.png")


# ------------------------------------------------------------------ Part 4 ----
def part4():
    raw = rgb(f"{NIKON_OUT}/stack_raw.png")
    h, w = raw.shape[0] // 4, raw.shape[1] // 8
    fig, ax = plt.subplots(figsize=(10, 10 * raw.shape[0] / raw.shape[1]))
    ax.imshow(raw)
    ax.set_xticks([w * (i + 0.5) for i in range(8)])
    ax.set_xticklabels([f"F{i}\n{d}" for i, d in enumerate(["8.41 m", "8.41 m", "0.89 m", "0.50 m",
                                                             "0.38 m", "0.32 m", "0.25 m", "0.25 m"])])
    ax.set_yticks([h * (i + 0.5) for i in range(4)])
    ax.set_yticklabels(["f/4", "f/5.6", "f/8", "f/11"])
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.3)
    save(fig, "p4_stack.png")

    res = rgb(f"{NIKON_OUT}/align_residual_before_after.png")
    half = res.shape[1] // 2
    grid([res[:, :half], res[:, half:]], ["Before alignment", "After alignment"], 2, "p4_alignment.png", panel_w=5)

    imgs = [rgb(f"{NIKON_OUT}/all_in_focus.png"), rgb(f"{NIKON_OUT}/dff_depth.png"),
            rgb(f"{NIKON_OUT}/confocal_depth.png"), rgb(f"{NIKON_OUT}/confocal_confidence.png"),
            rgb(f"{NIKON_OUT}/confocal_depth_filled.png")]
    titles = ["All-in-focus (f/4 stack)", "Depth from focus (f/4 stack)", "Confocal stereo (argmin)",
              "Confocal confidence", "Confocal stereo, confidence-filled"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 10.2))
    for ax, im, t in zip(axes.ravel(), imgs, titles):
        show(ax, im, t)
    # colour bar legend for the depth maps
    axes[2, 1].axis("off")
    ax = axes[2, 1].inset_axes([0.05, 0.05, 0.06, 0.78])
    ax.imshow(np.linspace(1, 0, 256)[:, None], cmap="viridis", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([0, 255]); ax.set_yticklabels(["near (F7, 0.25 m)", "far (F0, 8.41 m)"])
    ax.yaxis.tick_right(); ax.set_title("Depth colour scale", loc="left", pad=14)
    fig.tight_layout(pad=0.4)
    save(fig, "p4_results.png")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    stack, aif, depth = part1()
    bonus2(aif, depth)
    part4()
    part3()
