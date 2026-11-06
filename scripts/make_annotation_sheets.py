"""
Prepare everything an annotator needs for one or more lectures.

For each video this writes:
  - data/annotations/sheets/<video_id>.md: a timestamped reading sheet with
    the Whisper transcript (30-second rows) and each new slide text, so the
    annotator can skim the lecture and jump to the right place in the video.
  - data/annotations/gold/<video_id>.<annotator>.json: an empty gold template
    (only with --annotator, and never overwriting an existing file).

The sheet is a navigation aid. Boundaries and query moments must be checked
against the VIDEO itself: ASR and OCR contain errors.

Usage
-----
    python scripts/make_annotation_sheets.py --annotator A1
    python scripts/make_annotation_sheets.py --videos college_lec03 --annotator A2
"""

import argparse
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import PATHS, ensure_dirs
from src.utils.io_utils import format_timestamp, load_features, save_json
from src.utils.segment_text import has_raw_asr, span_transcript

ROW_SEC = 30.0


def write_sheet(video_id: str, out_dir: str) -> float:
    feats = load_features(os.path.join(PATHS.features_dir, f"{video_id}.npz"))
    ts = feats["timestamps"]
    duration = float(ts[-1] + (ts[1] - ts[0]))
    ocr = feats["ocr_texts"]

    slide_changes = []
    prev = None
    for t, text in zip(ts, ocr):
        text = str(text).strip()
        if text and text != prev:
            slide_changes.append((float(t), text))
        prev = text

    lines = [f"# Annotation sheet: {video_id}", "",
             f"Duration: {format_timestamp(duration)} ({duration:.0f}s). "
             f"Transcript source: {'raw Whisper segments' if has_raw_asr(feats) else 'legacy rolling windows (re-run refresh_features.py --asr)'}.",
             "Follow docs/ANNOTATION_GUIDELINES.md. Times in the gold JSON are in SECONDS.", "",
             "| time | seconds | transcript | new slide text |", "|---|---|---|---|"]
    t = 0.0
    si = 0
    while t < duration:
        end = t + ROW_SEC
        speech = span_transcript(feats, t, end).replace("|", "/")
        slides = []
        while si < len(slide_changes) and slide_changes[si][0] < end:
            slides.append(slide_changes[si][1][:200].replace("|", "/"))
            si += 1
        lines.append(f"| {format_timestamp(t)} | {t:.0f} | {speech} | {' // '.join(slides)} |")
        t = end
    with open(os.path.join(out_dir, f"{video_id}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--annotator", default=None, help="Annotator id, e.g. A1. Creates empty gold templates.")
    args = parser.parse_args()
    ensure_dirs()

    sheets_dir = os.path.join(PATHS.annotations_dir, "sheets")
    os.makedirs(sheets_dir, exist_ok=True)
    video_ids = args.videos or [os.path.splitext(os.path.basename(p))[0]
                                for p in sorted(glob.glob(os.path.join(PATHS.features_dir, "*.npz")))]
    for vid in video_ids:
        duration = write_sheet(vid, sheets_dir)
        msg = f"{vid}: sheet -> {os.path.join(sheets_dir, vid + '.md')}"
        if args.annotator:
            path = os.path.join(PATHS.gold_dir, f"{vid}.{args.annotator}.json")
            if os.path.exists(path):
                msg += f" | gold file exists, left untouched: {path}"
            else:
                save_json({"video_id": vid, "annotator": args.annotator,
                           "lecture_span": [0.0, round(duration, 1)],
                           "boundaries_sec": [], "queries": [], "notes": ""}, path)
                msg += f" | template -> {path}"
        print(msg)


if __name__ == "__main__":
    main()
