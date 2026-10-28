"""
Measure Whisper's word error rate (WER) on Indian-English classroom speech.

Step 1 (automatic): pick random 60-second clips that contain speech and write
one reference template per clip:

    python scripts/asr_wer.py sample --per-lecture 2

  -> data/annotations/wer/<video_id>_<start>.txt, each starting with a header
     line "# video_id start end". Below it, a human types EXACTLY what is said
     in that minute of the video (manual step). Whisper's own output is NOT
     shown, so the reference is not biased towards it.

Step 2 (automatic): compare the references with Whisper:

    python scripts/asr_wer.py score

  -> results/asr_wer.json: per-clip and pooled WER (substitutions +
     deletions + insertions over reference words, after lowercasing and
     removing punctuation).
"""

import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, ensure_dirs
from src.utils.io_utils import load_features
from src.utils.segment_text import clean_asr, has_raw_asr, span_transcript

WER_DIR = os.path.join(PATHS.annotations_dir, "wer")


def norm_words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def edit_distance(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)]


def sample(per_lecture, clip_sec, seed):
    os.makedirs(WER_DIR, exist_ok=True)
    rng = random.Random(seed)
    for path in sorted(glob.glob(os.path.join(PATHS.features_dir, "*.npz"))):
        vid = os.path.splitext(os.path.basename(path))[0]
        feats = load_features(path)
        if not has_raw_asr(feats):
            print(f"{vid}: no raw ASR segments yet (run refresh_features.py --asr), skipped")
            continue
        segs = [s for s in clean_asr(feats) if len(s.text.split()) >= 5]
        picked = rng.sample(segs, min(per_lecture, len(segs)))
        for s in picked:
            start = int(s.start)
            out = os.path.join(WER_DIR, f"{vid}_{start}.txt")
            if not os.path.exists(out):
                with open(out, "w", encoding="utf-8") as f:
                    f.write(f"# {vid} {start} {start + clip_sec}\n"
                            f"# Type below exactly what is said between {start // 60}:{start % 60:02d} and "
                            f"{(start + clip_sec) // 60}:{(start + clip_sec) % 60:02d}. Lines starting with # are ignored.\n")
            print("template:", out)


def score():
    rows, tot_err, tot_ref = [], 0, 0
    for path in sorted(glob.glob(os.path.join(WER_DIR, "*.txt"))):
        lines = open(path, encoding="utf-8").read().splitlines()
        _, vid, start, end = lines[0].split()
        ref = norm_words(" ".join(l for l in lines if not l.startswith("#")))
        if not ref:
            print(f"{os.path.basename(path)}: no reference typed yet, skipped")
            continue
        hyp = norm_words(span_transcript(load_features(os.path.join(PATHS.features_dir, f"{vid}.npz")),
                                         float(start), float(end)))
        err = edit_distance(ref, hyp)
        rows.append({"clip": os.path.basename(path), "ref_words": len(ref), "wer": err / len(ref)})
        tot_err, tot_ref = tot_err + err, tot_ref + len(ref)
    if not rows:
        print("No typed references found.")
        return
    out = {"pooled_wer": tot_err / tot_ref, "clips": rows, "ref_words": tot_ref}
    path = os.path.join(PATHS.results_dir, "asr_wer.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"Pooled WER {out['pooled_wer']:.3f} over {len(rows)} clips ({tot_ref} words). Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--per-lecture", type=int, default=2)
    s.add_argument("--clip-sec", type=int, default=60)
    s.add_argument("--seed", type=int, default=7)
    sub.add_parser("score")
    args = ap.parse_args()
    ensure_dirs()
    if args.cmd == "sample":
        sample(args.per_lecture, args.clip_sec, args.seed)
    else:
        score()


if __name__ == "__main__":
    main()
