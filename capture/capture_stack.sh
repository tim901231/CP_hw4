#!/usr/bin/env bash
# Capture a focal-aperture stack with gphoto2 (no interactive shell).
#
# Camera setup BEFORE running (do it on the camera):
#   - mode dial on M, ISO fixed (Auto ISO off), zoom fixed
#   - focus mode AF-S, lens switch on A (AF)
#   - tripod, static scene
#
# Usage:
#   ./capture_stack.sh check     # list camera + valid f-number / shutter values
#   ./capture_stack.sh test      # 1 aperture, 3 focus steps -> check files and focus
#   ./capture_stack.sh sweep     # widest aperture, full fine sweep -> into $OUT/sweep (check before full)
#   ./capture_stack.sh range     # widest aperture, RANGE_STEPS steps -> into $OUT/range,
#                                #   then: python capture/analyze_range.py stack/range
#   ./capture_stack.sh           # full capture
#   DRY_RUN=1 ./capture_stack.sh # print commands without touching the camera

set -u

# ---- settings (edit these) ---------------------------------------------------
# values must match the camera's choices exactly (./capture_stack.sh check);
# f-numbers are sent as "f/<value>", shutter speeds as listed (e.g. 0.0166s = 1/60)
APERTURES=(4 5.6 8 11 16 22)          # full stops, wide -> narrow
# last range sweep came out at mean 24/255 at f/4, 1/60 s, ISO 200 -> about 4 stops
# too dark; ISO 400 (+1) and 1/8 s at f/4 (+3) fixes that
SHUTTERS=(0.1250s 0.2500s 0.5000s 1.0000s 2.0000s 4.0000s)
                                      # one per aperture, doubling per stop keeps brightness;
                                      # use the same value everywhere to fix the shutter instead
ISO=400
# full capture is RAW for linear data + full EXIF; test/range/sweep stay JPEG so
# analyze_range.py can read them (choices: capture/capture_stack.sh check)
QUALITY_FULL="NEF (Raw)"
QUALITY_PREVIEW="JPEG Fine"
N_FOCUS=20                            # focus positions per aperture
# home (+1000) is the FAR end; negative drives focus nearer. The range sweep
# (steps of -150) had the scene sharp from shot 2 to shot 5-6, i.e. -300 .. -900
# from home, so shot 0 is at -225 and shot 19 at -225 + 19 * -35 = -890
PRE_DRIVE=-225                        # drive after homing, before the first shot
STEP=-35                              # focus drive per step (sign opposite to HOME_DRIVE)
RANGE_STEP=-150                       # coarse step used only by the "range" sweep
HOME_DRIVE=1000                       # big drive to hit the end of the focus range
HOME_REPEAT=3                         # repeat to make sure we really reach the end
RANGE_STEPS=12                        # shots for the "range" sweep
SETTLE=1                             # seconds to wait after a focus step (lens/tripod settle)
OUT=stack

# config paths (check with: gphoto2 --list-config)
CFG_VIEWFINDER=/main/actions/viewfinder
CFG_FOCUSDRIVE=/main/actions/manualfocusdrive
CFG_FNUMBER=/main/capturesettings/f-number
CFG_SHUTTER=/main/capturesettings/shutterspeed
CFG_ISO=/main/imgsettings/iso
CFG_QUALITY=/main/capturesettings/imagequality
# -----------------------------------------------------------------------------

DRY_RUN=${DRY_RUN:-0}

run() {
    echo "+ gphoto2 $*"
    if [ "$DRY_RUN" = 1 ]; then return 0; fi
    gphoto2 "$@"
}

# each gphoto2 call is a new session, so every focus drive re-enters the viewfinder
focus_drive() {
    run --set-config "$CFG_VIEWFINDER=1" --set-config "$CFG_FOCUSDRIVE=$1"
}

go_home() {
    for _ in $(seq 1 "$HOME_REPEAT"); do
        # hitting the end prints "Nikon manual focus stepping too small" - expected
        focus_drive "$HOME_DRIVE" || true
    done
}

capture_aperture() {
    local a=$1 shutter=$2 n=$3
    echo "=== f/$a, shutter $shutter, $n focus steps ==="
    run --set-config "$CFG_FNUMBER=f/$a" --set-config "$CFG_SHUTTER=$shutter" \
        --set-config "$CFG_ISO=$ISO" || { echo "failed to set exposure for f/$a"; exit 1; }
    run --set-config "$CFG_QUALITY=$QUALITY" || { echo "failed to set image quality '$QUALITY'"; exit 1; }
    # read back what the camera actually uses (a mode dial not on M overrides the shutter)
    if [ "$DRY_RUN" != 1 ]; then
        for c in "$CFG_FNUMBER" "$CFG_SHUTTER" "$CFG_ISO" "$CFG_QUALITY"; do
            echo "  $c -> $(gphoto2 --get-config "$c" | sed -n 's/^Current: //p')"
        done
    fi
    go_home
    if [ "$PRE_DRIVE" != 0 ]; then
        focus_drive "$PRE_DRIVE" || echo "WARNING: pre-drive failed for f/$a"
    fi
    for i in $(seq 0 $((n - 1))); do
        local name
        name=$(printf "%s/f%s_d%02d.%%C" "$OUT" "$a" "$i")
        run --set-config "$CFG_VIEWFINDER=1" --capture-image-and-download \
            --filename "$name" --force-overwrite || echo "WARNING: capture failed at f/$a step $i"
        if [ "$i" -lt $((n - 1)) ]; then
            focus_drive "$STEP" || echo "WARNING: focus step failed after f/$a step $i (end of range?)"
            [ "$DRY_RUN" = 1 ] || sleep "$SETTLE"
        fi
    done
}

if ! command -v gphoto2 >/dev/null && [ "$DRY_RUN" != 1 ]; then
    echo "gphoto2 not found (macOS: brew install gphoto2)"; exit 1
fi

# macOS grabs the camera for Photos/Image Capture; free it first
if [ "$(uname)" = Darwin ] && [ "$DRY_RUN" != 1 ]; then
    pkill -f PTPCamera 2>/dev/null || true
fi

mkdir -p "$OUT"

if [ "${1:-full}" = full ]; then QUALITY=$QUALITY_FULL; else QUALITY=$QUALITY_PREVIEW; fi

case "${1:-full}" in
    check)
        run --auto-detect
        run --get-config "$CFG_FNUMBER"
        run --get-config "$CFG_SHUTTER"
        run --get-config "$CFG_ISO"
        run --get-config "$CFG_QUALITY"
        ;;
    test)
        capture_aperture "${APERTURES[0]}" "${SHUTTERS[0]}" 3
        ;;
    range)
        OUT="$OUT/range"; mkdir -p "$OUT"
        STEP=$RANGE_STEP; PRE_DRIVE=0
        capture_aperture "${APERTURES[0]}" "${SHUTTERS[0]}" "$RANGE_STEPS"
        ;;
    sweep)
        OUT="$OUT/sweep"; mkdir -p "$OUT"
        capture_aperture "${APERTURES[0]}" "${SHUTTERS[0]}" "$N_FOCUS"
        ;;
    full)
        if [ "${#APERTURES[@]}" -ne "${#SHUTTERS[@]}" ]; then
            echo "APERTURES and SHUTTERS must have the same length"; exit 1
        fi
        for k in "${!APERTURES[@]}"; do
            capture_aperture "${APERTURES[$k]}" "${SHUTTERS[$k]}" "$N_FOCUS"
        done
        ;;
    *)
        echo "usage: $0 [check|test|range|sweep|full]"; exit 1
        ;;
esac

[ "$DRY_RUN" = 1 ] || gphoto2 --set-config "$CFG_VIEWFINDER=0" >/dev/null 2>&1
echo "done -> $OUT/"
