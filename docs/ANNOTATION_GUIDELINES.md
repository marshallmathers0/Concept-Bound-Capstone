# Annotation guidelines

These annotations are the ground truth of the paper. Every number we publish is measured against them, so consistency matters more than speed.

## 0. Before you start

- **Permission.** Do not annotate or share a recording until the department, the instructor and (for anything released) the college have agreed in writing. Students in the audience are visible; any released frames or clips must have them blurred.
- **Setup.**

  ```bash
  python scripts/make_annotation_sheets.py --annotator A1     # use your own id: A1, A2, ...
  ```

  This creates `data/annotations/sheets/<video>.md`, a timestamped transcript plus slide text to help you navigate. It also creates an empty `data/annotations/gold/<video>.A1.json` for you to fill in.
- **Watch the video, not just the sheet.** ASR and OCR contain mistakes.
- **Work independently.** For the double-annotated videos (at least 20–25% of them, picked before anyone starts), the two annotators must not see each other's files until both are done.

## 1. File format

```json
{
  "video_id": "college_lec03",
  "annotator": "A1",
  "lecture_span": [95.0, 3790.0],
  "boundaries_sec": [412.0, 905.5, 1620.0],
  "queries": [
    {"query": "what does a language model actually predict",
     "start_time": 120.0, "end_time": 260.0,
     "concept": "language_model_definition", "type": "definition"}
  ],
  "notes": ""
}
```

All times are in **seconds** from the start of the file. The `seconds` column of the sheet helps here.

## 2. `lecture_span`

`lecture_span` runs from the moment teaching starts to the moment it ends. Exclude:

- set-up and waiting time
- an empty stage after the class
- long breaks

Some recordings continue for a long time after class: `college_lec02` shows an empty stage for its last ~35 minutes.

## 3. Concept boundaries (`boundaries_sec`)

A boundary is where the instructor **moves from one concept to the next**: a new topic someone could want to jump straight to. The test: *would a student revising for an exam want a separate chapter marker here?*

**Mark a boundary at:**

- the start of a new topic or sub-topic (e.g. "Now let's look at attention")
- the switch from theory to a worked example or demo of a *different* idea
- the return from a digression to the main thread, if the digression is itself a concept

**Do not mark a boundary at:**

- every new slide; several slides often explain one concept
- a question from the audience that stays on the same concept
- writing on the board, or switching windows, within the same idea

Place the boundary at the first sentence of the new concept, accurate to about ±5 s. Typical segments last 2–10 minutes. Under 30 s or over 15 min is unusual, and the validator will warn you.

## 4. Queries

Write questions **a student would type into a search box** while revising. Aim for 10–20 per lecture, spread across the whole lecture.

- **Use student language.** Do not copy phrases from the slide or transcript. "why does the model need softmax at the end" is good. "softmax function applied to the output logits layer" is copied from the slide: bad. The validator flags any query sharing 6 or more consecutive words with the transcript of its moment.
- **Vary the type.** Set `type` to one of `definition`, `explanation`, `example`, `derivation`, `comparison`, `application`, `procedure`, `other`.
- **Keep the moment minimal.** `start_time`/`end_time` should cover the shortest stretch of video that fully answers the question, usually 30 s to 5 min. It does not have to line up with your boundaries.
- **One moment per query.** If the answer is spread over two places, pick the main one and mention the other in `notes`.
- Include some **hard queries**: paraphrases, questions about a diagram or example, questions where the key word is never spoken.
- Fill `concept` with a short snake_case label. It is used to group results.

## 5. Check your work

```bash
python scripts/validate_annotations.py
```

Fix every ERROR. Read every warning and fix it, or leave a reason in `notes`. The report in `results/annotation_report.md` includes inter-annotator agreement once a video has two annotators.

## 6. Test split

`data/splits.json` fixes which lectures are test lectures. Nobody tunes anything (thresholds, prompts, hyperparameters) while looking at test results. If you have to change the split, do it before the first test evaluation and say so in the paper.
