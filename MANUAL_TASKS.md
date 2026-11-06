# Manual tasks (only a human can do these)

Everything else is automated. After finishing any item below, re-run:

```
python scripts/run_paper_pipeline.py --eval-only
```

That re-evaluates, re-runs the ablations, and rebuilds both versions of the paper with the new numbers (`paper/ICALT2027_draft_guide.docx`, `paper/ICALT2027_draft_blind.docx`).

## 1. Permissions (do first)
- [ ] Written permission from the department and each instructor to use the recordings for research and publication.
- [ ] Decide what can be released (recommended: annotations, transcripts and features only; no video, or video with student faces blurred).

## 2. Concept boundaries for the 3 pilot lectures
These unlock Table II (segmentation) and the oracle row in Table III.
- Files: `data/annotations/gold/college_lec01.A1.json`, `college_lec02.A1.json`, `college_lec03.A1.json`
- Fill `"lecture_span": [start_sec, end_sec]` (when teaching actually happens; lec02 stops at about 27 min).
- Fill `"boundaries_sec": [...]`, where each concept change is in **seconds** (e.g. 24:50 is 1490).
- Use the sheets in `data/annotations/sheets/*.md` to navigate, and check against the video.
- Rules: `docs/ANNOTATION_GUIDELINES.md`.

## 3. Second annotator (inter-annotator agreement)
- A different person fills `data/annotations/gold/college_lec03.A2.json` **without looking at A1's file**: lecture span, boundaries and their own 10–15 queries.

## 4. More queries
- 10–20 queries per lecture is the target; the pilot has 2–8. Add more to the existing A1 files (same JSON format), especially for lec02 and lec01.
- Paraphrase like a student; don't copy slide or transcript wording (`python scripts/validate_annotations.py` flags copies).

## 5. Whisper accuracy (WER)
- Open each file in `data/annotations/wer/` (10 one-minute clips are already chosen).
- Play that minute of the video and type **exactly** what is said below the header lines.
- Then run `python scripts/asr_wer.py score`; the number goes into the paper automatically.

## 6. Remaining lectures (to reach ~30)
- Put new videos in `data/raw_videos/`, then run `python scripts/extract_features.py` (slow: OCR plus Whisper).
- **Before** annotating them: `python scripts/make_splits.py --force`, which freezes the final train/val/test split.
- Then `python scripts/make_annotation_sheets.py --annotator A1` and annotate the test lectures.

## 7. Paper details
- Author names, emails, guide name, department and acknowledgment: yellow text in `paper/build_paper.js` (search for `tbd(`).
- Repository URL for code release.
- Before submitting, paste the text into IEEE's official `conference-template-letter.docx` (ieee.org → Conferences → Publishing → Templates). ieee.org blocks automated downloads.
- Submit `ICALT2027_draft_blind` as a PDF by **15 Jan 2027** (ICALT, track AISLE).

## 8. RAG layer (ask-your-lectures demo)
- [ ] **Verify the 49 drafted eval questions** (`d..` in `eval/make_questions.py`): check each answer span in the video, fix times or delete. Steps: `eval/README.md`.
- [ ] **Add an LLM key** (free Groq or Gemini) to `.env` (copy `.env.example`), then run `python -m eval.run` to get faithfulness, citation accuracy and LLM latency. Until then those metrics are reported as "not run".
- [ ] **Public demo:** pick 3–5 openly licensed lectures (e.g. MIT OCW, CC BY-NC-SA), fill `data/video_meta.json` (title, YouTube URL, licence, attribution), build the index, then follow README → "Deploy to Hugging Face Spaces". The college recordings must not go public.
- [ ] Mark `lecture_span` for lecture 5 so the roll call (student names, ~40:30–1:02:00) is excluded from the index.
- [ ] Record the demo GIF (`docs/demo.gif`) and add the Space and video links to README.
