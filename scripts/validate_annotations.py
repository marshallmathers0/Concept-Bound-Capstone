"""
Check the human gold annotations and measure inter-annotator agreement.

For every data/annotations/gold/<video_id>.<annotator>.json:
  - schema and range errors (these make the file unusable)
  - warnings: very short/long segments or moments, unknown query types,
    duplicate queries, and queries that copy the transcript word-for-word
    (copied queries make lexical search look far better than it is)

For videos with two or more annotators: pairwise boundary agreement
(F1 at +/-15s and +/-30s, Pk, WindowDiff). Report these numbers in the paper.

Writes results/annotation_report.md.

Usage
-----
    python scripts/validate_annotations.py
"""

import glob
import itertools
import os
import re
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import EVAL, PATHS, ensure_dirs
from src.evaluation.boundary_metrics import boundary_f1, pk_windowdiff
from src.stage1_boundary.pseudo_labels import boundaries_from_annotations
from src.utils.gold import QUERY_TYPES, gold_files, list_gold_videos
from src.utils.io_utils import load_features, load_json
from src.utils.segment_text import span_transcript

COPY_NGRAM = 6


def longest_shared_ngram(a: str, b: str) -> int:
    aw, bw = re.findall(r"[a-z0-9]+", a.lower()), re.findall(r"[a-z0-9]+", b.lower())
    best = 0
    for n in range(1, min(len(aw), 12) + 1):
        grams = {tuple(bw[i:i + n]) for i in range(len(bw) - n + 1)}
        if any(tuple(aw[i:i + n]) in grams for i in range(len(aw) - n + 1)):
            best = n
        else:
            break
    return best


def check_file(path: str, feats) -> (list, list, dict):
    errors, warnings = [], []
    ann = load_json(path)
    ts = feats["timestamps"]
    duration = float(ts[-1] + (ts[1] - ts[0]))

    for key in ("video_id", "annotator", "lecture_span", "boundaries_sec", "queries"):
        if key not in ann:
            errors.append(f"missing field '{key}'")
    if errors:
        return errors, warnings, ann

    lo, hi = ann["lecture_span"]
    if not (0 <= lo < hi <= duration + 1):
        errors.append(f"lecture_span {ann['lecture_span']} outside 0..{duration:.0f}s")

    b = [float(x) for x in ann["boundaries_sec"]]
    if b != sorted(b) or len(set(b)) != len(b):
        errors.append("boundaries_sec must be sorted and unique")
    if any(not (lo < x < hi) for x in b):
        errors.append("a boundary lies outside lecture_span")
    cuts = [lo] + sorted(b) + [hi]
    lengths = np.diff(cuts)
    if len(lengths) and lengths.min() < 30:
        warnings.append(f"a segment is only {lengths.min():.0f}s long (concepts are usually minutes)")
    if len(lengths) and lengths.max() > 900:
        warnings.append(f"a segment is {lengths.max() / 60:.1f} min long; is there really no concept change?")

    seen = Counter(q.get("query", "").strip().lower() for q in ann["queries"])
    for i, q in enumerate(ann["queries"]):
        tag = f"query {i + 1} '{q.get('query', '')[:40]}'"
        if not q.get("query", "").strip():
            errors.append(f"{tag}: empty text")
            continue
        s, e = float(q.get("start_time", -1)), float(q.get("end_time", -1))
        if not (lo <= s < e <= hi):
            errors.append(f"{tag}: moment [{s}, {e}] not inside lecture_span")
        elif e - s < 10 or e - s > 900:
            warnings.append(f"{tag}: moment is {e - s:.0f}s long")
        if q.get("type") not in QUERY_TYPES:
            warnings.append(f"{tag}: type '{q.get('type')}' not in {QUERY_TYPES}")
        if seen[q["query"].strip().lower()] > 1:
            warnings.append(f"{tag}: duplicate query text")
        n = longest_shared_ngram(q["query"], span_transcript(feats, s, e))
        if n >= COPY_NGRAM:
            warnings.append(f"{tag}: shares a {n}-word run with the transcript; paraphrase it like a student would")
    return errors, warnings, ann


def main():
    ensure_dirs()
    vids = list_gold_videos()
    if not vids:
        print(f"No gold files in {PATHS.gold_dir}. Create templates with scripts/make_annotation_sheets.py --annotator A1")
        return

    report = ["# Annotation report", ""]
    n_err = 0
    stats = {"videos": 0, "files": 0, "queries": 0, "boundaries": 0, "moment_sec": [], "types": Counter()}
    iaa_rows = []
    for vid in vids:
        feats = load_features(os.path.join(PATHS.features_dir, f"{vid}.npz"))
        files = gold_files(vid)
        stats["videos"] += 1
        anns = {}
        for annotator, path in files.items():
            errors, warnings, ann = check_file(path, feats)
            anns[annotator] = ann
            stats["files"] += 1
            n_err += len(errors)
            if not errors:
                stats["queries"] += len(ann["queries"])
                stats["boundaries"] += len(ann["boundaries_sec"])
                stats["moment_sec"] += [q["end_time"] - q["start_time"] for q in ann["queries"]]
                stats["types"].update(q.get("type", "other") for q in ann["queries"])
            if errors or warnings:
                report.append(f"## {os.path.basename(path)}")
                report += [f"- ERROR: {e}" for e in errors] + [f"- warning: {w}" for w in warnings] + [""]

        for a, b in itertools.combinations(sorted(anns), 2):
            ts = feats["timestamps"]
            lo = max(anns[a]["lecture_span"][0], anns[b]["lecture_span"][0])
            hi = min(anns[a]["lecture_span"][1], anns[b]["lecture_span"][1])
            keep = (ts >= lo) & (ts < hi)
            ba = np.where(boundaries_from_annotations(ts[keep], anns[a]["boundaries_sec"]) == 1)[0]
            bb = np.where(boundaries_from_annotations(ts[keep], anns[b]["boundaries_sec"]) == 1)[0]
            row = {"video": vid, "pair": f"{a}-{b}",
                   "f1@15s": boundary_f1(ba, bb, EVAL.boundary_tolerance_steps)["f1"],
                   "f1@30s": boundary_f1(ba, bb, 2 * EVAL.boundary_tolerance_steps)["f1"]}
            row.update({k: v for k, v in pk_windowdiff(ba, bb, int(keep.sum())).items() if k != "k"})
            iaa_rows.append(row)

    ms = np.array(stats["moment_sec"]) if stats["moment_sec"] else np.zeros(1)
    report[2:2] = [
        f"- videos: {stats['videos']} | annotation files: {stats['files']} | errors: {n_err}",
        f"- queries: {stats['queries']} | boundaries: {stats['boundaries']}",
        f"- query moment length: median {np.median(ms):.0f}s, mean {ms.mean():.0f}s",
        f"- query types: {dict(stats['types'])}", ""]
    if iaa_rows:
        report += ["## Inter-annotator agreement (boundaries)", "",
                   "| video | pair | F1@15s | F1@30s | Pk | WindowDiff |", "|---|---|---|---|---|---|"]
        report += [f"| {r['video']} | {r['pair']} | {r['f1@15s']:.3f} | {r['f1@30s']:.3f} | {r['pk']:.3f} | {r['windowdiff']:.3f} |"
                   for r in iaa_rows]
        report.append(f"| **mean** | | {np.mean([r['f1@15s'] for r in iaa_rows]):.3f} | "
                      f"{np.mean([r['f1@30s'] for r in iaa_rows]):.3f} | {np.mean([r['pk'] for r in iaa_rows]):.3f} | "
                      f"{np.mean([r['windowdiff'] for r in iaa_rows]):.3f} |")
    else:
        report += ["No video has two annotators yet: inter-annotator agreement cannot be reported. "
                   "Double-annotate at least 20-25% of the videos."]

    out = os.path.join(PATHS.results_dir, "annotation_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nSaved {out}")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
