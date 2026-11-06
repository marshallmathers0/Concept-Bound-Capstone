"""
Auto-generate weak (query, segment) TRAINING pairs without manual annotation.

Output: `data/annotations/<video_id>_pseudo_queries.json`
    {"source": "pseudo", "generator": "tfidf-v2",
     "pairs": [{"query": ..., "start_time": ..., "end_time": ..., "style": ...}]}

These are training signal only. They are written to a separate file name so
they can never be mistaken for human annotations, and the evaluation script
never reads them: a pseudo-query is built from its target segment's own text,
so lexical search "finds" it trivially and any score on them is meaningless.

Per segment, up to three query styles are produced:
  - "sentence": the transcript sentence with the highest TF-IDF content
                (not the first words of the segment, which are usually filler
                like "so shall we start")
  - "keywords": the segment's top TF-IDF terms, like a terse search query
  - "slide":    the first line of slide text shown in the segment

Filler words are stripped, near-silent segments are skipped, and a query text
that occurs for more than one segment is dropped (it cannot identify either).

Usage
-----
    python scripts/generate_pseudo_queries.py
    python scripts/generate_pseudo_queries.py --styles sentence keywords
"""

import argparse
import glob
import os
import re
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import PATHS, ensure_dirs
from src.utils.io_utils import load_features, load_json, save_json
from src.utils.segment_text import span_ocr, span_transcript

FILLERS = re.compile(r"\b(um+|uh+|hmm+|okay|ok|right|so|you know|i mean|basically|actually|"
                     r"like|yeah|alright|all right|see|correct)\b[,.]?", re.IGNORECASE)
MIN_SEGMENT_WORDS = 40


def clean(text: str) -> str:
    text = FILLERS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" ,.")


def sentences(text: str):
    return [clean(s) for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-dir", default=PATHS.segments_dir)
    parser.add_argument("--features-dir", default=PATHS.features_dir)
    parser.add_argument("--annotations-dir", default=PATHS.annotations_dir)
    parser.add_argument("--styles", nargs="+", default=["sentence", "keywords", "slide"])
    parser.add_argument("--keywords", type=int, default=4)
    args = parser.parse_args()
    ensure_dirs()

    seg_files = [p for p in sorted(glob.glob(os.path.join(args.segments_dir, "*.json")))]
    video_ids = [os.path.splitext(os.path.basename(p))[0] for p in seg_files]
    if not video_ids:
        print(f"No segmentation files in {args.segments_dir}. Run segment_videos.py first.")
        return

    # Collect segment texts for all videos first: IDF is fitted over the whole
    # corpus so "keywords" are terms specific to a segment, not to the course.
    units = []
    for vid in video_ids:
        feats = load_features(os.path.join(args.features_dir, f"{vid}.npz"))
        for seg in load_json(os.path.join(args.segments_dir, f"{vid}.json"))["segments"]:
            st, et = float(seg["start_time"]), float(seg["end_time"])
            units.append({"video_id": vid, "start_time": st, "end_time": et,
                          "transcript": span_transcript(feats, st, et),
                          "ocr": span_ocr(feats, st, et)})

    # min_df=2 and >=3-letter tokens keep one-off ASR/OCR garbage ("yw", "kiais")
    # out of the vocabulary; it would otherwise get the highest IDF of all.
    tfidf = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5,
                            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    tfidf.fit([clean(u["transcript"] + " " + u["ocr"]) for u in units])
    vocab = tfidf.vocabulary_
    idf = tfidf.idf_
    terms = np.array(tfidf.get_feature_names_out())

    def sentence_score(sent: str) -> float:
        toks = [w for w in re.findall(r"[a-zA-Z]+", sent.lower()) if w in vocab]
        return float(sum(idf[vocab[w]] for w in set(toks)))

    candidates = []
    skipped = 0
    for u in units:
        text = clean(u["transcript"])
        if len(text.split()) < MIN_SEGMENT_WORDS:
            skipped += 1
            continue
        if "sentence" in args.styles:
            sents = [s for s in sentences(u["transcript"]) if 6 <= len(s.split()) <= 25]
            if sents:
                candidates.append((u, max(sents, key=sentence_score), "sentence"))
        if "keywords" in args.styles:
            doc = (text + " " + clean(u["ocr"])).lower()
            vec = tfidf.transform([doc]).toarray()[0]
            tf = Counter(re.findall(r"[a-z]{3,}", doc))
            # Salient = frequent in this segment, rare elsewhere.
            order = [i for i in np.argsort(-vec) if vec[i] > 0 and tf[terms[i]] >= 2]
            top = terms[order[:args.keywords]]
            if len(top) >= 2:
                candidates.append((u, " ".join(top), "keywords"))
        if "slide" in args.styles and u["ocr"]:
            title = " ".join(u["ocr"].split()[:8])
            if len(title.split()) >= 2:
                candidates.append((u, title, "slide"))

    # A query text shared by several segments identifies none of them.
    counts = Counter((u["video_id"], q.lower()) for u, q, _ in candidates)
    by_video = {vid: [] for vid in video_ids}
    dropped = 0
    for u, q, style in candidates:
        if counts[(u["video_id"], q.lower())] > 1:
            dropped += 1
            continue
        by_video[u["video_id"]].append({"query": q, "start_time": u["start_time"],
                                        "end_time": u["end_time"], "style": style})

    total = 0
    for vid, pairs in by_video.items():
        legacy = os.path.join(args.annotations_dir, f"{vid}_queries.json")
        if os.path.exists(legacy):
            # Old pseudo-queries were stored under the name evaluation used to
            # read as ground truth. Rename them so nothing mistakes them for gold.
            os.replace(legacy, legacy.replace("_queries.json", "_queries.legacy_pseudo.json"))
        save_json({"source": "pseudo", "generator": "tfidf-v2", "pairs": pairs},
                  os.path.join(args.annotations_dir, f"{vid}_pseudo_queries.json"))
        total += len(pairs)
        print(f"{vid}: {len(pairs)} pseudo pairs")

    print(f"\nTotal pseudo pairs: {total}  (skipped {skipped} near-silent segments, "
          f"dropped {dropped} duplicate queries)")
    if total == 0:
        print("WARNING: 0 pairs. Check that Whisper/EasyOCR produced text (data/features/*.npz).")


if __name__ == "__main__":
    main()
