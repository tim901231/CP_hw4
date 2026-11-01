"""HW4 Problems 1 and 3: lightfield refocusing, depth from focus, confocal stereo, and
refocusing an unstructured lightfield by template matching.

Run from the repository root:
    python src/main.py <command>        (python src/main.py -h lists the commands)
"""
import argparse
import os

import numpy as np
import cv2
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import correlate2d
from utils import lRGB2XYZ

# Problem 1
LIGHTFIELD = "data/chessboard_lightfield.png"
LENSLET = 16                                   # each lenslet covers 16 x 16 pixels
D_VALUES = 0.1 * np.arange(16)                 # focal stack refocus depths
APERTURES = range(1, 9)                        # square apertures |u|, |v| <= a  (4 a^2 views)
DFF_SIGMA_1, DFF_SIGMA_2 = 2, 6
P1_DIR = "results/problem1"
FSTACK_DIR = f"{P1_DIR}/focal_stack"
AFSTACK_DIR = "output/af_stack"                # 128 images, not committed
MOSAIC_FULL = "output/mosaic.jpg"              # full resolution (~90 MB), not committed

# Problem 3
VIDEO = "data/own/lightfield_video.mp4"
TEMPLATES = {"near": (760, 725, 900, 870),     # (y1, x1, y2, x2) in frame 0
             "middle": (700, 425, 860, 600),
             "back": (860, 180, 980, 300)}
SEARCH_R = 30                                  # search radius around the previous match
P3_DIR = "results/problem3"
SHIFTS_DIR = "output/p3_shifts"


# ------------------------------------------------------------------ helpers ----
def read_img(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_img(img, path):
    # img in [0, 1]; the format follows the file extension
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path, quality=95)


def linear_imgs(srgb):
    return np.where(srgb <= 0.0404482, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def gamma_encode(imgs):
    return np.where(imgs <= 0.0031308, 12.92 * imgs, 1.055 * imgs ** (1 / 2.4) - 0.055)


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


# --------------------------------------------------------------- Problem 1 ----
def read_lightfield(path=LIGHTFIELD):
    # (s, u, t, v, c) blocks of the lenslet image -> L(u, v, s, t, c)
    img = read_img(path)
    img = img.reshape(img.shape[0] // LENSLET, LENSLET, img.shape[1] // LENSLET, LENSLET, 3)
    return np.transpose(img, (1, 3, 0, 2, 4))


def mosaic(L):
    # rows of views: increasing u; columns: increasing v
    n, _, h, w, c = L.shape
    return np.transpose(L, (0, 2, 1, 3, 4)).reshape(n * h, n * w, c)


def refocus(L, d, a=None):
    # Eq. (2): shift every sub-aperture view by (d u, d v) and average over the aperture
    # |u|, |v| <= a (a=None: the full aperture); dividing by the number of views keeps
    # the exposure the same for every aperture
    n, _, h, w, _ = L.shape
    max_uv = (n - 1) / 2
    u = np.arange(n) - max_uv
    v = max_uv - np.arange(n)                  # this lightfield's v axis is mirrored
    grid = (np.arange(h), np.arange(w))
    out = np.zeros((h, w, 3))
    count = 0
    for i in range(n):
        for j in range(n):
            if a is not None and (abs(u[i]) > a or abs(v[j]) > a):
                continue
            X, Y = np.meshgrid(np.arange(h) + d * u[i], np.arange(w) + d * v[j], indexing="ij")
            interp = RegularGridInterpolator(grid, L[i, j], method="linear",
                                             bounds_error=False, fill_value=None)
            out += interp((X, Y))
            count += 1
    return out / count


def fstack_path(i):
    return f"{FSTACK_DIR}/d_{i:02d}.png"


def afstack_path(a, i):
    return f"{AFSTACK_DIR}/a_{a}_d_{i:02d}.png"


def read_focus_stack():
    return np.stack([read_img(fstack_path(i)) for i in range(len(D_VALUES))])


def read_af_stack():
    # (apertures, d, H, W, 3)
    return np.stack([np.stack([read_img(afstack_path(a, i)) for i in range(len(D_VALUES))])
                     for a in APERTURES])


def merge(imgs, sigma_1=DFF_SIGMA_1, sigma_2=DFF_SIGMA_2, eps=1e-5):
    # Eqs. (3)-(8): sharpness weights from the luminance of the linear images
    imgs = linear_imgs(imgs)
    ws = []
    for img in imgs:
        I_lum = lRGB2XYZ(img)[:, :, 1]
        I_high = I_lum - cv2.GaussianBlur(I_lum, (0, 0), sigma_1)
        w = cv2.GaussianBlur(I_high ** 2, (0, 0), sigma_2)
        ws.append(w + eps)                     # textureless pixels fall back to the stack average
    ws = np.stack(ws)

    d = -D_VALUES[:, None, None]
    depth_img = normalize(np.sum(d * ws, axis=0) / np.sum(ws, axis=0))
    final_img = np.sum(imgs * ws[..., None], axis=0) / np.sum(ws, axis=0)[..., None]
    return gamma_encode(final_img), depth_img


def confocal_depth(af_stack):
    # confocal constancy: at the in-focus d, every aperture sees the same colour, so the
    # variance across apertures (summed over RGB) is smallest
    var = np.var(linear_imgs(af_stack), axis=0).sum(axis=-1)
    return np.argmin(var, axis=0)


def p1_views():
    L = read_lightfield()
    m = mosaic(L)
    save_img(m, MOSAIC_FULL)
    save_img(cv2.resize(m, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), f"{P1_DIR}/mosaic_half.jpg")


def p1_refocus():
    L = read_lightfield()
    for i, d in enumerate(D_VALUES):
        save_img(refocus(L, d), fstack_path(i))
        print(f"d = {d:.1f}")


def p1_dff():
    aif, depth = merge(read_focus_stack())
    save_img(aif, f"{P1_DIR}/all_in_focus.png")
    save_img(depth, f"{P1_DIR}/depth.png")


def p1_confocal():
    L = read_lightfield()
    for a in APERTURES:
        for i, d in enumerate(D_VALUES):
            save_img(refocus(L, d, a), afstack_path(a, i))
        print(f"aperture a = {a}")
    best = confocal_depth(read_af_stack())
    save_img(normalize(-best.astype(np.float32)), f"{P1_DIR}/confocal_depth.png")   # same scale as depth.png


# --------------------------------------------------------------- Problem 3 ----
def ncc(I, g):
    # Eq. (9) with only correlations and pixel-wise ops; output indexed by the
    # template's top-left corner ('valid')
    gz = g - g.mean()
    n = g.size
    box = np.ones_like(g)
    num = correlate2d(I, gz, mode="valid")              # sum g~ (I - I_bar) = sum g~ I since sum g~ = 0
    sum_I = correlate2d(I, box, mode="valid")
    sum_I2 = correlate2d(I ** 2, box, mode="valid")
    var_I = np.maximum(sum_I2 - sum_I ** 2 / n, 0)     # sum (I - I_bar)^2 per window
    return num / np.sqrt((gz ** 2).sum() * var_I + 1e-12)


def track_template(gray, y1, x1, y2, x2, r=SEARCH_R):
    # search each frame only within +-r of the previous frame's match
    g = gray[0][y1:y2, x1:x2]
    th, tw = g.shape
    H, W = gray[0].shape
    y, x = y1, x1
    shifts, peaks = [], []
    for I in gray:
        top, left = max(y - r, 0), max(x - r, 0)
        bottom, right = min(y + th + r, H), min(x + tw + r, W)
        h = ncc(I[top:bottom, left:right], g)
        row, col = np.unravel_index(np.argmax(h), h.shape)
        y, x = top + row, left + col
        shifts.append((y - y1, x - x1))
        peaks.append(h.max())
    return np.array(shifts), np.array(peaks)


def shift_and_merge(frames, shifts):
    # move each frame by -(dy, dx) so the template lands where it was in frame 0;
    # uncovered pixels are zero-padded and excluded from that pixel's average
    H, W = frames[0].shape[:2]
    acc = np.zeros(frames[0].shape, np.float64)
    count = np.zeros((H, W, 1), np.float64)
    for f, (dy, dx) in zip(frames, shifts):
        out = np.zeros_like(f)
        valid = np.zeros((H, W, 1))
        # destination rows/cols receive source rows/cols offset by (dy, dx)
        ys, yd = (slice(dy, H), slice(0, H - dy)) if dy >= 0 else (slice(0, H + dy), slice(-dy, H))
        xs, xd = (slice(dx, W), slice(0, W - dx)) if dx >= 0 else (slice(0, W + dx), slice(-dx, W))
        out[yd, xd] = f[ys, xs]
        valid[yd, xd] = 1
        acc += out
        count += valid
    return acc / np.maximum(count, 1)


class LinearFrames:
    """Video frames linearised on access, so the float video is never all in memory."""
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return linear_imgs(self.frames[i].astype(np.float32) / 255)


def read_video(path=VIDEO):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def shifts_path(name):
    return f"{SHIFTS_DIR}/{name}.npz"


def p3():
    frames = LinearFrames(read_video())
    gray = [lRGB2XYZ(frames[i])[:, :, 1] for i in range(len(frames))]
    print("video:", len(frames), "frames of", gray[0].shape)
    os.makedirs(SHIFTS_DIR, exist_ok=True)
    for name, box in TEMPLATES.items():
        shifts, peaks = track_template(gray, *box)
        print(f"{name}: dy {shifts[:, 0].min()}..{shifts[:, 0].max()}, "
              f"dx {shifts[:, 1].min()}..{shifts[:, 1].max()}, min peak NCC {peaks.min():.3f}")
        np.savez(shifts_path(name), shifts=shifts, peaks=peaks)
        save_img(gamma_encode(shift_and_merge(frames, shifts)), f"{P3_DIR}/refocus_{name}.png")


COMMANDS = {
    "p1-views": (p1_views, "Problem 1: sub-aperture mosaic"),
    "p1-refocus": (p1_refocus, "Problem 1: focal stack d = 0, 0.1, ..., 1.5"),
    "p1-dff": (p1_dff, "Problem 1: all-in-focus image and depth from focus (needs p1-refocus)"),
    "p1-confocal": (p1_confocal, "Problem 1: focal-aperture stack and confocal stereo"),
    "p3": (p3, "Problem 3: template-matching refocusing of my video"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=[*COMMANDS, "p1", "all"],
                        help="; ".join(f"{k}: {v[1]}" for k, v in COMMANDS.items())
                        + "; p1: all Problem 1 commands; all: everything")
    args = parser.parse_args()
    if args.command == "all":
        names = list(COMMANDS)
    elif args.command == "p1":
        names = [k for k in COMMANDS if k.startswith("p1")]
    else:
        names = [args.command]
    for name in names:
        print(f"== {name}")
        COMMANDS[name][0]()
