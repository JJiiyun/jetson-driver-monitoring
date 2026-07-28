#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/resize_video.sh INPUT_VIDEO [OUTPUT_VIDEO]

Examples:
  ./scripts/resize_video.sh \
    data/source/final_test_0727_14_30.mp4

  ./scripts/resize_video.sh \
    data/source/final_test_0727_14_30.mp4 \
    data/converted/benchmark_720p.mp4

The output defaults to:
  data/converted/<input-name>_720p.mp4

Conversion settings:
  - Resolution: 1280x720 (16:9 input)
  - Frame rate: 30 FPS
  - Video codec: H.264 (libx264)
  - Audio: copied without re-encoding
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[ERROR] ffmpeg is not installed." >&2
    echo "Install it with: sudo apt update && sudo apt install ffmpeg" >&2
    exit 1
fi

if ! command -v ffprobe >/dev/null 2>&1; then
    echo "[ERROR] ffprobe is not installed." >&2
    echo "Install it with: sudo apt update && sudo apt install ffmpeg" >&2
    exit 1
fi

INPUT_VIDEO="$1"
if [[ ! -f "${INPUT_VIDEO}" ]]; then
    echo "[ERROR] Input video not found: ${INPUT_VIDEO}" >&2
    exit 1
fi

if [[ $# -eq 2 ]]; then
    OUTPUT_VIDEO="$2"
else
    INPUT_NAME="$(basename -- "${INPUT_VIDEO}")"
    INPUT_STEM="${INPUT_NAME%.*}"
    OUTPUT_VIDEO="${PROJECT_ROOT}/data/converted/${INPUT_STEM}_720p.mp4"
fi

if [[ -e "${OUTPUT_VIDEO}" ]]; then
    echo "[ERROR] Output already exists: ${OUTPUT_VIDEO}" >&2
    echo "Move or rename the existing file before running again." >&2
    exit 1
fi

mkdir -p -- "$(dirname -- "${OUTPUT_VIDEO}")"

echo "Input : ${INPUT_VIDEO}"
echo "Output: ${OUTPUT_VIDEO}"
echo "Converting to 1280x720 at 30 FPS..."

ffmpeg \
    -nostdin \
    -i "${INPUT_VIDEO}" \
    -vf "scale=1280:-2:flags=lanczos,fps=30" \
    -c:v libx264 \
    -preset fast \
    -crf 20 \
    -c:a copy \
    -movflags +faststart \
    "${OUTPUT_VIDEO}"

echo
echo "=== Converted video ==="
ffprobe \
    -v error \
    -select_streams v:0 \
    -show_entries stream=width,height,r_frame_rate,duration \
    -of default=noprint_wrappers=1 \
    "${OUTPUT_VIDEO}"

echo
echo "Done: ${OUTPUT_VIDEO}"

