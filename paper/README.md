# ICALT 2027 paper draft

Target: **IEEE ICALT 2027**, Full paper (max **5 pages including authors, abstract, figures and references**), track **AISLE** (AI and Smart Learning Environments). Deadline **15 Jan 2027, 23:59 Pacific**; PDF only; double-blind review. Contact: ieeetclt@gmail.com.

| File | What it is |
|---|---|
| `ICALT2027_draft_guide.docx` / `.pdf` | Version for the project guide (placeholder authors) |
| `ICALT2027_draft_blind.docx` / `.pdf` | Anonymised version (author block blank, as ICALT requires) |
| `build_paper.js` | Generates both versions: `node paper/build_paper.js [--blind]` (set `DRAFT_SUFFIX=_v3` to write versioned copies) |
| `make_figure.py` | Fig. 1 (pipeline): `python paper/make_figure.py` |
| `make_figure2.py` | Fig. 2 (real signals of one lecture + pilot spans): `python paper/make_figure2.py college_lec03` |
| `render.ps1` | Word → PDF + page count: `powershell -ExecutionPolicy Bypass -File paper/render.ps1 <docx>` |

Text highlighted in **yellow** is a placeholder. No result in the draft is invented.

## Formatting rules applied
- Layout follows the IEEE conference template (US Letter, single-column title, two 3.5" columns with a 0.25" gap, Times New Roman 10 pt body, small-caps Roman-numeral section headings, italic lettered subsections, 8 pt captions and references, IEEE reference style, last-page columns balanced).
- Abstract ≤ 200 words (currently ≈180), plus a Keywords line.
- Blind version: author lines left empty; no names, institution or acknowledgments of people.
- IEEE AI-content policy: use of an AI assistant is disclosed in the Acknowledgment.

**Before submission:** IEEE's site blocks automated downloads, so the layout was rebuilt to spec, not filled into IEEE's own file. Download the official template (ieee.org → Conferences → Publishing → Templates → "conference-template-letter.docx", the file ICALT links to) and paste the text into it, or ask me to.

## Open items
1. **Pilot annotation** (you): 2–3 lectures, following `docs/ANNOTATION_GUIDELINES.md`, then Tables I–II and Section VI are filled from `python scripts/run_evaluation.py`.
2. Author names, guide, emails, department permission statement.
3. Hardware/runtime line; Whisper WER on a sampled subset.
5. Final dataset numbers after the feature re-extraction and the full 30 lectures.

7. Table I (corpus statistics) is measured on 5 re-processed lectures; refresh after the remaining 6 finish (`logs/refresh_features_part2.log`).
