#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
FRAME_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ai01-demo.XXXXXX")
trap 'rm -rf "$FRAME_DIR"' EXIT HUP INT TERM
cd "$ROOT_DIR"

"$ROOT_DIR/.venv/bin/python" -m src.demo --compact >/dev/null

for source in "$ROOT_DIR"/docs/assets/demo-frame-*.svg; do
  qlmanage -t -s 960 -o "$FRAME_DIR" "$source" >/dev/null
done

ffmpeg -hide_banner -loglevel error -y \
  -framerate 0.4 -pattern_type glob -i "$FRAME_DIR/demo-frame-*.svg.png" \
  -filter_complex \
  "[0:v]crop=960:540:0:0,fps=5,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer:bayer_scale=5" \
  "$ROOT_DIR/docs/assets/demo.gif"

echo "Rendered docs/assets/demo.gif from verified demo output."
