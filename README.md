# CP HW4: Lightfields and Focal Stacks

15-663 Computational Photography, Homework 4. Lightfield refocusing, depth from
focus and confocal stereo on the chessboard lightfield, refocusing my own
unstructured lightfield by template matching, and two bonus problems (better
blending and depth from focus, and my own focal-aperture stack captured with a
Nikon D3500).

The report is [`report.pdf`](report.pdf) (LaTeX source and figures in `report/`; after rebuilding with `latexmk -pdf report.tex` inside `report/`, copy `report/report.pdf` to the repository root).

## Setup

Tested with Python 3.8 and:

```
numpy 1.24  scipy 1.10  opencv-python 5.0  Pillow 10.4  matplotlib 3.7
```

```bash
pip install numpy scipy opencv-python pillow matplotlib
```

The chessboard lightfield from the homework ZIP is not committed. Copy it to
`data/chessboard_lightfield.png`.

My own captures are committed in `data/own/` (see [My captures](#my-captures)).
Decoding my raw NEF files (`src/bonus4.py prepare`) also needs
[`dcraw`](https://www.dechifro.org/dcraw/) (`brew install dcraw`), but the
committed stack is already decoded.

## Usage

Run everything from the repository root:

```bash
python src/main.py <command>
```

| Command | What it does | Final images |
|---|---|---|
| `p1-views` | Problem 1: sub-aperture views and their mosaic | `results/problem1/mosaic_half.jpg` |
| `p1-refocus` | Problem 1: focal stack, Eq. (2) with the full aperture, $d=0,0.1,\dots,1.5$ | `results/problem1/focal_stack/` |
| `p1-dff` | Problem 1: all-in-focus image and depth from focus (needs `p1-refocus`) | `results/problem1/all_in_focus.png`, `depth.png` |
| `p1-confocal` | Problem 1: focal-aperture stack (8 square apertures x 16 depths) and confocal stereo | `results/problem1/confocal_depth.png` |
| `p3` | Problem 3: template-matching refocusing of my video at three figures | `results/problem3/` |
| `p1` | All Problem 1 commands | |
| `all` | All of the above | |

The bonus problems and the report figures have their own scripts:

```bash
python src/bonus2.py                  # Bonus 2, refocus blur model   -> results/bonus2/refocus/
python src/bonus2.py --model paper    # Bonus 2, thin-lens blur model -> results/bonus2/thin_lens/
python src/bonus4.py                  # Bonus 4, my focal-aperture stack -> results/bonus4/
python src/report_figures.py          # report figures -> report/figures/ (run the above first)
```

`python src/main.py -h` and `python src/bonus4.py -h` list the options.
Intermediate images (the full-resolution mosaic, the focal-aperture stack and
the template-matching shifts) go to `output/` (not committed).

Approximate run times on a laptop: `p1-views` < 1 min, `p1-refocus` 2 min,
`p1-dff` < 1 min, `p1-confocal` 10 min, `p3` 2 min, `bonus2.py` < 1 min per
model, `bonus4.py` < 1 min, `report_figures.py` 1 min.

## Code

| File | Contents |
|---|---|
| `src/main.py` | Problems 1 and 3 and the command-line entry point: `read_lightfield` and `mosaic`, `refocus` (Eq. 2, full or square aperture), `merge` (all-in-focus image and depth from focus, Eqs. 3-8), `confocal_depth` (Hasinoff and Kutulakos, Eqs. 7-8), `ncc` (Eq. 9 with `correlate2d` only), `track_template` (search around the previous match) and `shift_and_merge` |
| `src/bonus2.py` | Bonus 2: blur estimation and depth fitting adapted from Suwajanakorn et al., "Depth from Focus with Your Mobile Phone" (CVPR 2015), with a refocus model and the paper's thin-lens model |
| `src/bonus4.py` | Bonus 4: NEF decoding, alignment of the focus settings (ORB + RANSAC similarity), exposure equalization, depth from focus and confocal stereo on my stack |
| `src/report_figures.py` | Figures for the report |
| `src/utils.py` | Course-provided color helpers (`lRGB2XYZ`) |
| `capture/capture_stack.sh` | Captures a focal-aperture stack with `gphoto2` (Nikon D3500): sets f-number, shutter and ISO, drives the focus to one end, then alternates captures and focus steps |
| `capture/analyze_range.py` | Finds the shots in which each part of the scene is sharpest, to choose the focus range of the stack |

## Selected parameters

| Problem | Parameters |
|---|---|
| 1, refocusing | $v=\mathrm{maxUV}-j$ (this lightfield's $v$ axis is mirrored); $d=0,0.1,\dots,1.5$; linear interpolation |
| 1, depth from focus | $\sigma_1=2$, $\sigma_2=6$; $\epsilon=10^{-5}$ added to every weight |
| 1, confocal stereo | square apertures $\lvert u\rvert,\lvert v\rvert\le a$, $a=1,\dots,8$ ($4a^2$ views); variance across apertures summed over RGB; $\arg\min$ over $d$ |
| 3 | templates of $145\times140$, $175\times160$ and $120\times120$ px in frame 0; search $\pm30$ px around the previous match; integer shifts, zero padding, per-pixel count normalization |
| Bonus 2 | refocus model: square PSF, radius $\le12$ px, $\sigma=2$; thin-lens model: disk PSF, radius $\le16$ px, $\sigma=3$ |
| Bonus 4 | similarity alignment per focus setting on $f/11$, chained to F0; per-image gain maps (Gaussian $\sigma=40$ px); DFF on $f/4$ with $\sigma_1=2$, $\sigma_2=10$; confocal: blur $\sigma=1.5$ px before warping, cost at 1/4 resolution with an $11\times11$ window, confidence-weighted fill ($\sigma=25$ px) |

The Bonus 4 pixel sizes are for the 1504-px-wide stack and are scaled with the
width of the input.

## My captures

| Files | Use | Capture |
|---|---|---|
| `data/own/lightfield_video.mp4` | Problem 3 | phone video, 126 frames, 1080 x 1920, 30 fps, camera moved in a plane |
| `data/own/nikon_af/A<f>F<i>.png` | Bonus 4 | Nikon D3500, AF-P 18-55 mm at 24 mm, tripod, tethered with `capture/capture_stack.sh`; $f/4$, $f/5.6$, $f/8$, $f/11$ x 8 focus settings (EXIF focusing distance 8.41, 8.41, 0.89, 0.50, 0.38, 0.32, 0.25, 0.25 m); 1/60 s, ISO 400 |

The Nikon images are linear 16-bit PNGs decoded from the NEF files with
`python src/bonus4.py prepare <NEF dir> data/own/nikon_af --width 752`
(`dcraw -4 -T -w -h -o 1`: linear, camera white balance, sRGB primaries). The
original NEF files (18 MB each) are kept locally because of their size.

**Note:** the Bonus 4 results in `results/bonus4/` and in the report were
computed from a 1504-px-wide decode
(`prepare ... data/nikon_af_full --width 1504`, then
`bonus4.py run --stack data/nikon_af_full`). Running `bonus4.py` on the
committed 752-px copies gives lower-resolution results.

## Results

| Folder | Files |
|---|---|
| `results/problem1/` | `mosaic_half.jpg` (half-resolution mosaic), `focal_stack/d_00.png` ... `d_15.png`, `all_in_focus.png`, `depth.png`, `confocal_depth.png` |
| `results/problem3/` | `refocus_near.png`, `refocus_middle.png`, `refocus_back.png` |
| `results/bonus2/refocus/`, `results/bonus2/thin_lens/` | `all_in_focus.png`, `depth.png`, `confidence.png`, `fit_report.json` |
| `results/bonus4/` | `stack_raw.png`, `stack_aligned_equalized.png`, `align_residual_before_after.png`, `all_in_focus.png`, `dff_depth.png`, `confocal_depth.png`, `confocal_confidence.png`, `confocal_depth_filled.png` (depth maps also as `*_gray.png`) |

## Acknowledgement

The visualizations and the writing of the report were prepared in
collaboration with Claude Code (Claude Opus 5.5).
