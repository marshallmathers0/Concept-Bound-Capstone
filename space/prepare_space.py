"""
Assemble a ready-to-push Hugging Face Space folder (build/hf_space/).

    python space/prepare_space.py
    python space/prepare_space.py --allow-internal-videos   # only with written permission

Copies only what the demo needs: space/app.py, the inference-side source
files, the precomputed RAG index, the Stage-2 checkpoint, pinned
requirements and a Space README with the required YAML header. It refuses
to package an index whose videos are marked "internal" in
data/video_meta.json, because the Space is public.
"""

import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "build", "hf_space")
SRC_FILES = [
    "config.py", "src/__init__.py",
    "src/rag/__init__.py", "src/rag/pipeline.py", "src/rag/schema.py", "src/rag/context.py",
    "src/rag/prompts.py", "src/rag/citations.py", "src/rag/rewrite.py",
    "src/llm/__init__.py", "src/llm/base.py", "src/llm/providers.py", "src/llm/factory.py",
    "src/retrieval/__init__.py", "src/retrieval/search.py", "src/retrieval/reranker.py",
    "src/stage2_retrieval/__init__.py", "src/stage2_retrieval/retrieval.py", "src/stage2_retrieval/encoders.py",
    "src/stage2_retrieval/index_builder.py", "src/stage2_retrieval/fusion_model.py",
    "src/feature_extraction/__init__.py", "src/feature_extraction/asr_extractor.py",
    "src/feature_extraction/clip_extractor.py",
    "src/utils/__init__.py", "src/utils/io_utils.py", "src/utils/text_search.py", "src/utils/gold.py",
]

SPACE_README = """---
title: ConceptBound
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: {license}
---

# ConceptBound: ask questions about lecture videos

Answers are grounded in the indexed lectures and cite `[Lecture, mm:ss]` moments.
Set `LLM_PROVIDER` and `GROQ_API_KEY` (or `GEMINI_API_KEY`) as Space secrets.

**Lecture sources and attribution:**
{attribution}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(ROOT, "data", "rag_index"))
    ap.add_argument("--allow-internal-videos", action="store_true")
    ap.add_argument("--license", default="mit", help="License of YOUR code for the Space card.")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.index, "segments_metadata.json"), encoding="utf-8"))
    vids = sorted({m["video_id"] for m in meta})
    internal = sorted({m["video_id"] for m in meta if "internal" in m.get("license", "").lower()})
    if internal and not args.allow_internal_videos:
        sys.exit(f"Refusing: {internal} are marked internal in data/video_meta.json. Index openly licensed "
                 f"lectures for the public Space (see README 'Deploy'), or pass --allow-internal-videos only "
                 f"if you have written permission to publish them.")

    shutil.rmtree(OUT, ignore_errors=True)
    for rel in SRC_FILES:
        dst = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), dst)
    shutil.copy(os.path.join(ROOT, "space", "app.py"), os.path.join(OUT, "app.py"))
    shutil.copy(os.path.join(ROOT, "space", "requirements.txt"), os.path.join(OUT, "requirements.txt"))
    shutil.copytree(args.index, os.path.join(OUT, "rag_index"))
    os.makedirs(os.path.join(OUT, "checkpoints"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "checkpoints", "stage2_retrieval_model.pt"),
                os.path.join(OUT, "checkpoints", "stage2_retrieval_model.pt"))

    vm = json.load(open(os.path.join(ROOT, "data", "video_meta.json"), encoding="utf-8"))

    # Video files are too large for a Space, so citations play from the URLs in
    # data/video_meta.json (e.g. unlisted YouTube). Refresh them here so adding
    # a link does not require rebuilding the index.
    meta_path = os.path.join(OUT, "rag_index", "segments_metadata.json")
    for m in meta:
        url = (vm.get(m["video_id"]) or {}).get("url") if isinstance(vm.get(m["video_id"]), dict) else None
        if url:
            m["url"], m["is_youtube"] = url, "youtu" in url
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    no_url = sorted({m["video_id"] for m in meta if "youtu" not in m.get("url", "")})
    if no_url:
        print(f"Note: no YouTube URL in data/video_meta.json for {no_url}; their citations will show "
              f"text only (the video files are not shipped to the Space).")

    attribution = "\n".join(f"- {vm.get(v, {}).get('title', v)}: {vm.get(v, {}).get('attribution', '?')} "
                            f"({vm.get(v, {}).get('license', 'license?')}) {vm.get(v, {}).get('url', '')}" for v in vids)
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(SPACE_README.format(license=args.license, attribution=attribution))
    size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fs in os.walk(OUT) for fn in fs) / 1e6
    print(f"Space folder ready: {OUT} ({size:.1f} MB, {len(vids)} lectures). Push it to your Space repo (see README).")


if __name__ == "__main__":
    main()
