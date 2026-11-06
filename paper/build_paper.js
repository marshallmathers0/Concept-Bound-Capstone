// Builds the ICALT 2027 full-paper draft (IEEE conference layout, US Letter).
//   node paper/build_paper.js            -> paper/ICALT2027_draft_guide.docx   (placeholder authors)
//   node paper/build_paper.js --blind    -> paper/ICALT2027_draft_blind.docx   (author block left blank, per ICALT)
// Yellow-highlighted text = placeholder to be replaced with pilot results / real details.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, SectionType, WidthType, BorderStyle, TabStopType, VerticalAlign,
  LineRuleType,
} = require("docx");

const BLIND = process.argv.includes("--blind");
const SUFFIX = process.env.DRAFT_SUFFIX || "";
const OUT = path.join(__dirname, (BLIND ? "ICALT2027_draft_blind" : "ICALT2027_draft_guide") + SUFFIX + ".docx");

// ---- IEEE conference template geometry (US Letter) -------------------------
const PAGE = { width: 12240, height: 15840 };
const MARGIN = { top: 1080, bottom: 1440, left: 900, right: 900, header: 720, footer: 720 };
const COL_GAP = 360;                       // 0.25"
const COL_W = (PAGE.width - MARGIN.left - MARGIN.right - COL_GAP) / 2;  // 5040 = 3.5"
const FONT = "Times New Roman";
const BODY = 20;                           // 10 pt (half-points)
const LINE = { line: 230, lineRule: LineRuleType.EXACT };  // ~11.5 pt leading

// ---- measured results (scripts/collect_paper_results.py) --------------------------
const RES_PATH = path.join(__dirname, "..", "results", "paper_results.json");
const RES = fs.existsSync(RES_PATH) ? JSON.parse(fs.readFileSync(RES_PATH, "utf8")) : {};
const f3 = (x) => (x === undefined || x === null || Number.isNaN(x)) ? null : x.toFixed(3);
const f0 = (x) => (x === undefined || x === null) ? null : Math.round(x).toLocaleString("en-US");
const pct = (x) => (x === undefined || x === null) ? null : Math.round(100 * x) + "%";

// ---- run helpers --------------------------------------------------------------
// A "seg" is a string or [text, opts]. opts: i, b, sub, sup, sc, tbd
function runs(segs, size = BODY, base = {}) {
  // a bare [text, opts] tuple is one segment, not a list of two
  const isTuple = Array.isArray(segs) && segs.length === 2 && typeof segs[0] === "string"
    && segs[1] !== null && typeof segs[1] === "object" && !Array.isArray(segs[1]);
  return (isTuple ? [segs] : Array.isArray(segs) ? segs : [segs]).map((s) => {
    const [text, o] = typeof s === "string" ? [s, {}] : s;
    const opt = { ...base, ...o };
    return new TextRun({
      text, font: FONT, size,
      italics: !!opt.i, bold: !!opt.b, smallCaps: !!opt.sc,
      subScript: !!opt.sub, superScript: !!opt.sup,
      highlight: opt.tbd ? "yellow" : undefined,
    });
  });
}
const tbd = (t) => [t, { tbd: true }];
// value or yellow placeholder
const V = (x, ph = "\u2013") => (x === null || x === undefined) ? tbd(ph) : String(x);
const it = (t) => [t, { i: true }];
const sub = (t) => [t, { i: true, sub: true }];

const P = (segs, extra = {}) => new Paragraph({
  children: runs(segs), alignment: AlignmentType.JUSTIFIED,
  indent: { firstLine: 202 }, spacing: { after: 0, ...LINE }, ...extra,
});
const PN = (segs) => P(segs, { indent: { firstLine: 0 } });

const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
let h1n = 0, h2n = 0;
const H1 = (t) => {
  h2n = 0;
  return new Paragraph({
    children: runs(`${ROMAN[h1n++]}. ${t}`, BODY, { sc: true }),
    alignment: AlignmentType.CENTER, spacing: { before: 160, after: 80 }, keepNext: true,
  });
};
const H1u = (t) => new Paragraph({   // unnumbered (Acknowledgment, References)
  children: runs(t, BODY, { sc: true }), alignment: AlignmentType.CENTER,
  spacing: { before: 160, after: 80 }, keepNext: true,
});
const H2 = (t) => new Paragraph({
  children: runs(`${String.fromCharCode(65 + h2n++)}. ${t}`, BODY, { i: true }),
  spacing: { before: 100, after: 40 }, keepNext: true,
});

// equation: centered body with number flush right (tab stops in a 3.5" column)
const EQ = (segs, n) => new Paragraph({
  tabStops: [{ type: TabStopType.CENTER, position: COL_W / 2 }, { type: TabStopType.RIGHT, position: COL_W }],
  spacing: { before: 60, after: 60, ...LINE },
  children: [new TextRun({ text: "\t", font: FONT, size: BODY }), ...runs(segs),
             new TextRun({ text: `\t(${n})`, font: FONT, size: BODY })],
});

const CAP_FIG = (n, text) => new Paragraph({
  children: runs([`Fig. ${n}. `, ...[].concat(text)], 16),
  alignment: AlignmentType.JUSTIFIED, spacing: { before: 60, after: 120 },
});
const CAP_TAB = (n, text) => [
  new Paragraph({ children: runs(`TABLE ${ROMAN[n - 1]}`, 16), alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true }),
  new Paragraph({ children: runs(text, 16, { sc: true }), alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true }),
];

// IEEE-style table: rules on top, under header, bottom; 8 pt text.
function TABLE(header, rows, widths, notes) {
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const rule = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
  const cell = (segs, w, top, bottom, bold, align) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    margins: { top: 20, bottom: 20, left: 50, right: 50 },
    verticalAlign: VerticalAlign.CENTER,
    borders: { top: top ? rule : none, bottom: bottom ? rule : none, left: none, right: none },
    children: [new Paragraph({ alignment: align, children: runs(segs, 16, { b: bold }) })],
  });
  const total = widths.reduce((a, b) => a + b, 0);
  const tRows = [new TableRow({
    tableHeader: true,
    children: header.map((h, i) => cell(h, widths[i], true, true, true, i ? AlignmentType.CENTER : AlignmentType.LEFT)),
  })];
  rows.forEach((r, ri) => tRows.push(new TableRow({
    children: r.map((c, i) => cell(c, widths[i], false, ri === rows.length - 1, false, i ? AlignmentType.CENTER : AlignmentType.LEFT)),
  })));
  const out = [new Table({ columnWidths: widths, width: { size: total, type: WidthType.DXA }, alignment: AlignmentType.CENTER, rows: tRows })];
  if (notes) out.push(new Paragraph({ children: runs(notes, 14), spacing: { before: 40, after: 120 }, alignment: AlignmentType.JUSTIFIED }));
  return out;
}

const BULLET = (label, segs) => new Paragraph({
  children: runs([label + " ", ...[].concat(segs)]), alignment: AlignmentType.JUSTIFIED,
  indent: { left: 202, hanging: 202 }, spacing: { after: 0, ...LINE },
});

// ---- references (IEEE style; numbered in order of first citation) -------------
const REFS = [
  ["P. Arora, S. Pehlivan, and J. Laaksonen, “Text-to-multimodal retrieval with bimodal input fusion in shared cross-modal transformer,” in ", it("Proc. LREC-COLING"), ", 2024, pp. 15823–15834."],
  ["H. Yang and C. Meinel, “Content based lecture video retrieval using speech and video text information,” ", it("IEEE Trans. Learn. Technol."), ", vol. 7, no. 2, pp. 142–154, 2014."],
  ["X. Zhang, C. Li, S.-W. Li, and V. Zue, “Automated segmentation of MOOC lectures towards customized learning,” in ", it("Proc. IEEE 16th Int. Conf. Advanced Learning Technologies (ICALT)"), ", 2016, pp. 20–22."],
  ["D. Singh S, A. Gupta, C. V. Jawahar, and M. Tapaswi, “Unsupervised audio-visual lecture segmentation,” in ", it("Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)"), ", 2023, pp. 5232–5241."],
  ["K. Ghosh, S. R. Nangi, Y. Kanchugantla, P. G. Rayapati, P. K. Bhowmick, and P. Goyal, “Augmenting video lectures: Identifying off-topic concepts and linking to relevant video lecture segments,” ", it("Int. J. Artif. Intell. Educ."), ", vol. 32, no. 2, pp. 382–412, 2022."],
  ["J. Lei, T. L. Berg, and M. Bansal, “Detecting moments and highlights in videos via natural language queries,” in ", it("Advances in Neural Information Processing Systems (NeurIPS)"), ", vol. 34, 2021."],
  ["M. A. Hearst, “TextTiling: Segmenting text into multi-paragraph subtopic passages,” ", it("Comput. Linguist."), ", vol. 23, no. 1, pp. 33–64, 1997."],
  ["H. Yu, C. Deng, Q. Zhang, J. Liu, Q. Chen, and W. Wang, “Multimodal fusion and coherence modeling for video topic segmentation,” in ", it("Findings of the Association for Computational Linguistics: ACL 2025"), ", 2025, pp. 17577–17593."],
  ["K. Dutta, M. Mathew, P. Krishnan, and C. V. Jawahar, “Localizing and recognizing text in lecture videos,” in ", it("Proc. 16th Int. Conf. Frontiers in Handwriting Recognition (ICFHR)"), ", 2018, pp. 235–240."],
  ["J. Lei, L. Yu, T. L. Berg, and M. Bansal, “TVR: A large-scale dataset for video-subtitle moment retrieval,” in ", it("Proc. European Conf. Computer Vision (ECCV)"), ", 2020, pp. 447–463."],
  ["A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in ", it("Proc. 40th Int. Conf. Machine Learning (ICML)"), ", 2023, pp. 28492–28518."],
  ["Y. Baek, B. Lee, D. Han, S. Yun, and H. Lee, “Character region awareness for text detection,” in ", it("Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)"), ", 2019, pp. 9357–9366."],
  ["A. Radford ", it("et al."), ", “Learning transferable visual models from natural language supervision,” in ", it("Proc. 38th Int. Conf. Machine Learning (ICML)"), ", 2021, pp. 8748–8763."],
  ["N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in ", it("Proc. EMNLP-IJCNLP"), ", 2019, pp. 3980–3990."],
  ["A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv:1807.03748, 2018."],
  ["J. Johnson, M. Douze, and H. Jégou, “Billion-scale similarity search with GPUs,” ", it("IEEE Trans. Big Data"), ", vol. 7, no. 3, pp. 535–547, 2021."],
  ["S. Robertson and H. Zaragoza, “The probabilistic relevance framework: BM25 and beyond,” ", it("Found. Trends Inf. Retr."), ", vol. 3, no. 4, pp. 333–389, 2009."],
  ["S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, and J.-Y. Nie, “C-Pack: Packed resources for general Chinese embeddings,” in ", it("Proc. 47th Int. ACM SIGIR Conf."), ", 2024, pp. 641–649."],
  ["G. V. Cormack, C. L. A. Clarke, and S. Buettcher, “Reciprocal rank fusion outperforms Condorcet and individual rank learning methods,” in ", it("Proc. 32nd Int. ACM SIGIR Conf."), ", 2009, pp. 758–759."],
  ["R. Nogueira and K. Cho, “Passage re-ranking with BERT,” arXiv:1901.04085, 2019."],
  ["D. Beeferman, A. Berger, and J. Lafferty, “Statistical models for text segmentation,” ", it("Mach. Learn."), ", vol. 34, no. 1–3, pp. 177–210, 1999."],
  ["L. Pevzner and M. A. Hearst, “A critique and improvement of an evaluation metric for text segmentation,” ", it("Comput. Linguist."), ", vol. 28, no. 1, pp. 19–36, 2002."],
];
const REF_PARAS = REFS.map((r, i) => new Paragraph({
  children: runs([`[${i + 1}]\t`, ...r], 16),
  tabStops: [{ type: TabStopType.LEFT, position: 360 }],
  indent: { left: 360, hanging: 360 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 20 },
}));

// ---- figure ------------------------------------------------------------------
function pngSize(buf) { return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) }; }
const figBuf = fs.readFileSync(path.join(__dirname, "fig1_pipeline.png"));
const fs1 = pngSize(figBuf);
const figW = 330;  // px at 96 dpi ~ 3.44"
const FIG = new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 80 }, keepNext: true,
  children: [new ImageRun({ type: "png", data: figBuf, transformation: { width: figW, height: Math.round(figW * fs1.h / fs1.w) },
    altText: { title: "Fig. 1", description: "System pipeline", name: "fig1" } })],
});

const fig2Buf = fs.readFileSync(path.join(__dirname, "fig2_signals.png"));
const fs2 = pngSize(fig2Buf);
const FIG2 = new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 80 }, keepNext: true,
  children: [new ImageRun({ type: "png", data: fig2Buf, transformation: { width: figW, height: Math.round(figW * fs2.h / fs2.w) },
    altText: { title: "Fig. 2", description: "Per-step signals of one lecture with annotated query spans", name: "fig2" } })],
});

// ---- title block ---------------------------------------------------------------
const title = new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 160 },
  children: runs("ConceptBound: Concept-Aware Retrieval of Student Queries Across a Collection of Classroom Lecture Recordings", 48),
});
const authorBlock = BLIND
  // ICALT: leave the author section blank, with empty lines for the camera-ready version.
  ? [0, 1, 2, 3].map(() => new Paragraph({ children: runs(" ", 22), alignment: AlignmentType.CENTER }))
  : [
      new Paragraph({ alignment: AlignmentType.CENTER, children: runs([tbd("Author One, Author Two, Author Three, Author Four")], 22) }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: runs([tbd("Dept. of Computer Science and Engineering (AI & ML)")], 20, { i: true }) }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: runs([tbd("PES University, Bengaluru, India")], 20) }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: runs([tbd("{author1, author2, author3, author4}@example.edu  |  Guide: Prof. Guide Name")], 20) }),
    ];

// ---- body ---------------------------------------------------------------------
const body = [];
const push = (...x) => body.push(...x.flat());

push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 80 },
  children: runs([["Abstract", { i: true, b: true }], ["—", { b: true }],
    ["Recorded classroom lectures are a growing learning resource, yet a student who wants to revisit one concept must scrub through an hour of single-camera video. We study concept-level moment retrieval in this setting: given a free-text student question, search the whole collection of recorded lectures and return the lecture and time span in which it is answered. We present ConceptBound, a two-stage system that (i) segments a lecture into concept-coherent units with a contrastively trained boundary detector over slide-text change, transcript topic drift, visual change and CLIP features, and (ii) ranks the units with a cross-modal fusion encoder, adapted from text-to-multimodal retrieval, over screen-cropped OCR, Whisper transcripts and frame embeddings. We also contribute a human-annotated benchmark of auditorium recordings from an undergraduate course, with concept boundaries, student-style queries and inter-annotator agreement, and an evaluation protocol that compares every segmenter–retriever pair with strong text baselines. ", { b: true }],
    tbd("In a pilot on N annotated lectures, [main finding]."),
    [" We also report practical lessons for classroom capture—silent OCR failure, ASR hallucination on silence and dead air—that affect any lecture search system.", { b: true }]], 18),
}));
push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
  children: runs([["Keywords—", { i: true, b: true }],
    ["lecture video retrieval, video moment retrieval, topic segmentation, multimodal fusion, smart learning environments", { i: true, b: true }]], 18),
}));

push(H1("Introduction"));
push(P("Many institutions now record every classroom lecture with a fixed camera, so students can revise from video rather than notes. A one-hour recording is, however, hard to use for revision: a student who wants “the part where the instructor explains why the model outputs a probability distribution” has to scrub through the timeline. Text-to-multimodal retrieval models fuse several modalities of a video in a shared embedding space [1], but are evaluated on short web clips. Lecture search systems have long indexed speech and slide text [2], and lecture segmentation methods split videos into navigable units [3], [4], but most of this work uses studio-recorded MOOCs or presentation captures, where slides are clean and topic changes coincide with slide changes."));
push(P("Ordinary classroom capture is harder. The projected screen occupies part of a wide auditorium shot, the camera also records the audience and burnt-in overlays, the instructor walks in front of the screen, speech is accented and spontaneous, and recordings contain dead air before and after class. Closest to our setting, Ghosh et al. [5] link concepts to segments of Indian NPTEL lectures, but their queries are concept names and the videos are studio recordings. General moment-retrieval benchmarks [6] target visual events in short web videos rather than conceptual questions answered mostly in speech and slide text."));
push(P("We therefore study concept-level moment retrieval in single-camera classroom recordings and ask two questions. RQ1: does segmenting a lecture into concept-coherent units, instead of fixed windows, improve retrieval of the moment that answers a student question? RQ2: does multimodal fusion of slide text, speech and visual features beat strong text-only retrieval over the transcript? Our contributions are:"));
push(BULLET("1)", "ConceptBound, a two-stage system that searches across all lectures of a course: a contrastively trained pedagogical boundary detector followed by a cross-modal segment encoder adapted from text-to-multimodal retrieval [1];"));
push(BULLET("2)", "a human-annotated benchmark of classroom lecture recordings with concept boundaries, free-text student queries, moment spans and inter-annotator agreement, together with an evaluation harness that scores every segmenter–retriever pair against strong text baselines with confidence intervals;"));
push(BULLET("3)", "an analysis of failure modes specific to classroom capture and how they were detected and fixed."));

push(H1("Related Work"));
push(H2("Lecture Video Segmentation and Search"));
push(P("Yang and Meinel [2] indexed lecture videos with OCR on key frames and ASR transcripts for keyword search. Zhang et al. [3] segmented MOOC lectures at visual transitions to support customized learning, and AVLectures [4] introduced a large benchmark with unsupervised audio-visual segmentation. Transcript-based segmentation descends from TextTiling [7]; recent work fuses visual and textual cues for video topic segmentation and argues that genuinely continuous long recordings are under-represented [8]. Text in classroom frames is hard to read: LectureVideoDB [9] showed that standard text detectors perform poorly on lecture frames. Ghosh et al. [5] retrieve NPTEL segments relevant to off-topic concepts, the closest prior work in the Indian context."));
push(H2("Moment Retrieval and Multimodal Fusion"));
push(P("Video moment retrieval localizes the span of a video that matches a natural-language query [6]; TVR [10] extends this to corpus-level search with subtitles. Arora et al. [1] proposed text-to-multimodal retrieval in which unimodal experts are combined by a shared cross-modal transformer over modality pairs and trained with pairwise contrastive losses. We adapt this idea to lecture moments, where the modalities are slide text, speech and frame appearance rather than audio and video of short web clips."));
push(H2("Text Retrieval Baselines"));
push(P("Because lecture content is carried mostly by speech and slide text, a learned multimodal model must be compared with strong text retrieval over the same units. We use BM25 [17], sparse lexical ranking that remains hard to beat on out-of-domain text; bi-encoders that embed queries and passages independently, Sentence-BERT [14] and BGE [18]; reciprocal-rank fusion [19], which combines rankers without calibrating their scores; and cross-encoder re-ranking [20], which reads the query and passage jointly. All are used off the shelf, without training on our data, so they also show what a lecture search tool could achieve today with no annotation at all."));

push(H1("Benchmark"));
push(H2("Recordings"));
push(P(["The corpus consists of one-hour auditorium recordings of an undergraduate course on large language models, captured by the institution’s fixed camera (1080×608, 25 fps). The prototype contains 11 lectures (11.7 h); ", tbd("the full set will contain about 30 lectures"), ". The camera frames the projected screen in the left half of the image together with the stage and part of the audience; recorder overlays (clock, watermark) are burnt into every frame. Framing differs between sessions, and some files continue long after class ends (one recording shows an empty stage for its last 37 minutes)."]));
const C = RES.corpus || {}, G = RES.gold || {};
push(...CAP_TAB(1, "Corpus statistics (prototype)"));
push(...TABLE(
  ["Property", "Value"],
  [["Lectures (prototype / planned)", [V(C.n_lectures, "11"), " / ", tbd("\u224830")]],
   ["Duration per lecture (total)", ["64 min (", V(C.hours, "11.7"), " h)"]],
   ["Video", "1080\u00D7608, 25 fps, one fixed camera"],
   ["Sampling step (steps per lecture)", "3 s (1,281)"],
   ["Transcript words per lecture\u2020", C.words_min ? `${f0(C.words_min)}\u2013${f0(C.words_max)} (mean ${f0(C.words_mean)})` : tbd("\u2013")],
   ["Whisper segments removed as loops\u2020", C.removed_max !== undefined ? `${C.removed_min}\u2013${C.removed_max} per lecture` : tbd("\u2013")],
   ["Frames with slide text found\u2020", C.ocr_min !== undefined ? `${pct(C.ocr_min).replace("%", "")}\u2013${pct(C.ocr_max)}` : tbd("\u2013")],
   ["Distinct slide-text states per lecture\u2020", C.slides_min !== undefined ? `${C.slides_min}\u2013${C.slides_max}` : tbd("\u2013")],
   ["Gold queries (lectures)", G.queries ? `${G.queries} (${G.lectures}); median span ${f0(G.median_span)} s` : tbd("\u2013")]],
  [3000, 2040],
  C.n_processed === C.n_lectures && C.n_lectures
    ? "\u2020 Measured on all lectures with the corrected pipeline."
    : ["\u2020 Measured on the ", V(C.n_processed, "5"), " lectures re-processed with the corrected pipeline so far; ", tbd("update for all lectures"), "."],
));
push(H2("Annotation Protocol"));
push(P(["For each lecture an annotator marks (a) the lecture span in which teaching takes place, (b) concept boundaries, defined as points where a student revising for an exam would want a separate chapter marker, and (c) 10–20 student-style questions, each with the minimal span that fully answers it and a type (definition, explanation, example, derivation, comparison, application, procedure). Annotators work from the video, aided by a timestamped transcript sheet, and are instructed to paraphrase rather than copy slide or transcript wording; a validator flags any query that shares six or more consecutive words with the transcript of its span, because copied queries inflate lexical baselines. ", tbd("N"), " lectures are annotated independently by two annotators to measure agreement. Splits are made at the lecture level and the test split is frozen before any experiment. ", "The pilot annotation covers 3 lectures with 17 queries (median answer span 123 s, mean 201 s; 7 explanation, 6 definition, 1 each of example, comparison, derivation and other). ", tbd("Boundary annotation and a second annotator for agreement are pending.")]));

push(H1("Method"));
push(FIG, CAP_FIG(1, "Overview of ConceptBound. Stage 1 segments a lecture into concept-coherent units; Stage 2 embeds each unit from three modalities and ranks units for a student query. Text-only baselines rank the same units."));
push(H2("Multimodal Signals"));
push(P(["Frames are sampled every 3 s. Speech is transcribed with Whisper [11]; segments with low confidence, high compression ratio or short phrases repeated in a 60-s window are discarded, since Whisper hallucinates on silence and room noise. Slide text is read with EasyOCR (CRAFT detector [12]) inside the projected screen, located automatically as the largest bright, screen-shaped region of the median frame of each 5-min chunk; cropping removes the clock overlay, which would otherwise register as a text change every step. From these we derive three per-step scalars—slide-text change (1−similarity of consecutive OCR strings), transcript topic drift (cosine distance of Sentence-BERT [14] embeddings of 30-s windows) and optical-flow magnitude—concatenated with a 32-d projection of the CLIP [13] frame embedding. With OCR string ", it("c"), ["t", { i: true, sub: true }], " and Sentence-BERT window embedding ", it("e"), ["t", { i: true, sub: true }], ","]));
push(EQ([it("o"), ["t", { i: true, sub: true }], " = 1 \u2212 SM(", it("c"), ["t", { i: true, sub: true }], ", ", it("c"), ["t\u22121", { i: true, sub: true }], "),   ", it("d"), ["t", { i: true, sub: true }], " = 1 \u2212 cos(", it("e"), ["t", { i: true, sub: true }], ", ", it("e"), ["t\u22121", { i: true, sub: true }], ")"], 1));
push(PN(["where SM is the normalized longest-matching-subsequence similarity of two strings; ", it("o"), ["t", { i: true, sub: true }], " and ", it("d"), ["t", { i: true, sub: true }], " are the slide-text change and topic-drift signals."]));
push(FIG2, CAP_FIG(2, "Smoothed per-step signals for one pilot lecture; shaded bands are the answer spans of the 8 annotated queries. Slide-text change is sparse and several spans begin at one of its peaks, whereas topic drift is noisy."));
push(P("Fig. 2 illustrates why no single signal suffices: slide changes are reliable but rare (the instructor may explain several ideas on one slide), and transcript drift reacts to every digression. Stage 1 therefore learns boundaries from all signals jointly instead of thresholding any one of them."));
push(H2("Stage 1: Pedagogical Boundary Detector"));
push(P(["A temporal encoder (BiLSTM; CNN and Transformer variants) maps the per-step features to L2-normalized embeddings ", it("z"), ["t", { i: true, sub: true }], ". The boundary score between steps ", it("t"), " and ", it("t"), "+1 is"]));
push(EQ([it("b"), ["t", { i: true, sub: true }], " = 1 − ", it("z"), ["t", { i: true, sub: true }], ["⊤", { sup: true }], it(" z"), ["t+1", { i: true, sub: true }], "."], 2));
push(PN(["Training pulls same-segment neighbours together and pushes boundary-crossing neighbours apart with margin ", it("m"), ","]));
push(EQ([it("L"), ["adj", { sub: true }], " = mean", ["t", { i: true, sub: true }], " [(1−", it("y"), ["t", { i: true, sub: true }], ")(1−", it("s"), ["t", { i: true, sub: true }], ")", ["2", { sup: true }], " + ", it("y"), ["t", { i: true, sub: true }], " max(0, ", it("s"), ["t", { i: true, sub: true }], "−(1−", it("m"), "))", ["2", { sup: true }], "]"], 3));
push(PN(["where ", it("s"), ["t", { i: true, sub: true }], " is the cosine similarity of neighbouring embeddings and ", it("y"), ["t", { i: true, sub: true }], " the boundary label, plus a segment-level InfoNCE term [15] with in-segment positives and cross-segment negatives (", it("L"), " = ", it("L"), ["adj", { sub: true }], " + 0.5 ", it("L"), ["NCE", { sub: true }], "). Labels come from human boundaries where available and otherwise from peaks of the input signals; because such pseudo-labels are derived from the model’s own inputs, the peak heuristic is always reported as a baseline, and checkpoints are selected by boundary F1 on annotated validation lectures. Candidate peaks above mean+std of ", it("b"), ["t", { i: true, sub: true }], " are filtered by non-maximum suppression to segments of 60–600 s."]));
push(PN(["The segment term draws, for each step, one positive ", it("z"), ["t", { i: true, sub: true }], ["+", { sup: true }], " from the same segment and ", it("K"), " = 4 negatives ", it("z"), ["k", { i: true, sub: true }], ["\u2212", { sup: true }], " from other segments:"]));
push(EQ([it("L"), ["NCE", { sub: true }], " = \u2212mean", ["t", { i: true, sub: true }], " log [", "exp(", it("z"), ["t", { i: true, sub: true }], "\u00B7", it("z"), ["t", { i: true, sub: true }], ["+", { sup: true }], "/\u03C4", ["1", { sub: true }], ") / \u03A3 exp(\u00B7)]"], 4));
push(PN(["where the sum runs over the positive and the ", it("K"), " negatives, with \u03C4", ["1", { sub: true }], " = 0.1."]));
push(H2("Stage 2: Cross-Modal Segment Encoder"));
push(P(["Each segment is represented by three tokens: Sentence-BERT embeddings of its distinct slide text and of its transcript (long texts are chunked and averaged rather than truncated) and the mean CLIP embedding. Following the fusion idea of [1], modality-specific projections with learned type embeddings pass through a transformer block whose output is pooled into ", it("z"), ["seg", { sub: true }], "; a query projector maps the Sentence-BERT query embedding into the same space. Training uses symmetric InfoNCE with temperature τ,"]));
push(EQ([it("L"), ["2", { sub: true }], " = −½[log σ(", it("q"), ["i", { i: true, sub: true }], "·", it("Z"), "/τ)", ["i", { i: true, sub: true }], " + log σ(", it("z"), ["i", { i: true, sub: true }], "·", it("Q"), "/τ)", ["i", { i: true, sub: true }], "]"], 5));
push(PN(["where σ is the softmax over the batch and in-batch pairs that share a target segment are masked, since they are not true negatives. Without human queries, training pairs are generated from each segment’s most informative transcript sentence, salient keywords and slide title; these pseudo-queries are never used for evaluation. A concatenation variant and a pairwise-fusion variant closer to [1] serve as ablations. Units are indexed with FAISS [16]."]));
push(P(["The pairwise variant follows [1] more closely: each modality first passes through its own unimodal expert (a residual two-layer MLP), one shared transformer block then fuses each modality pair (slide text\u2013speech, speech\u2013visual, slide text\u2013visual) with the same weights, ", it("z"), ["seg", { sub: true }], " is pooled from the three pair representations, and training adds a contrastive term between the query and each pair representation (weight 0.5), mirroring the pairwise loss combinations of [1]."]));
push(H2("Student-Facing Interface"));
push(P(["A web service exposes the index to students: a free-text query returns a ranked list of moments across the whole lecture collection, each with its lecture, time range, transcript and slide-text snippet and a link that opens the recording at the start of the moment. The same service can run any retriever of Section V, so a deployed tool can fall back to text search for new lectures before any model is retrained. ", ...(RES.runtime ? [`Median query latency for course-level search on ${RES.runtime.hardware.cpu} (CPU only) is ${f0(RES.runtime.query_latency["hybrid(bm25+dense:all-MiniLM-L6-v2)"].median_ms)} ms for the text hybrid and ${f0((RES.runtime.query_latency.fusion || {}).median_ms)} ms for the fusion model.`] : [tbd("Median query latency on [hardware]: [x] ms.")])]));

push(H1("Experimental Setup"));
push(P(["Every segmenter (fixed 30/60/120-s windows, shot-change peaks, TextTiling-style topic-drift peaks [7], the pseudo-label heuristic, Stage 1 and, as an oracle, the annotated segmentation) is combined with every retriever: BM25 [17], two dense encoders (all-MiniLM-L6-v2 [14] and BGE-small [18]), their reciprocal-rank fusion [19], a cross-encoder re-ranker [20] over the fused top 20, and the Stage 2 model. All retrievers see identical text for a unit, and the search space is the whole lecture collection: every query is matched against the segments of all evaluated lectures, so a system must find both the right video and the right moment within it. Segmentation is scored against the primary annotator with boundary F1 at ±15 s and ±30 s, Pk [21] and WindowDiff [22]. Retrieval counts a hit when a result in the right lecture overlaps the gold span with temporal IoU ≥ 0.3; we report R@1/5/10, R@1 at IoU ≥ 0.5, MRR and mean IoU of the top result, with 95% bootstrap confidence intervals over queries and a paired bootstrap test against fixed-60-s BM25. Stage 1 uses hidden size 128, margin 0.5 and AdamW (lr 10", ["−3", { sup: true }], ", 30 epochs); Stage 2 uses 384-d fusion, 4 heads, τ = 0.07 and AdamW (lr 2·10", ["−4", { sup: true }], ", 20 epochs). ", ...(RES.runtime ? [`All experiments run on a laptop CPU (${RES.runtime.hardware.cpu}, ${RES.runtime.hardware.ram} RAM, no GPU); feature extraction takes a median of ${f0(RES.runtime.extraction.ocr.median_sec / 60)} min (OCR) and ${f0(RES.runtime.extraction.asr.median_sec / 60)} min (ASR) per one-hour lecture.`] : [tbd("Hardware: [CPU/GPU model]; runtime per lecture: [x] min.")])]));
push(PN(["For a retrieved unit ", it("p"), " and gold span ", it("g"), " in the same lecture, and the rank ", it("r"), ["q", { i: true, sub: true }], " of the first hit for query ", it("q"), ","]));
push(EQ(["IoU(", it("p"), ", ", it("g"), ") = |", it("p"), " \u2229 ", it("g"), "| / |", it("p"), " \u222A ", it("g"), "|,   MRR = ", "mean", ["q", { i: true, sub: true }], " 1/", it("r"), ["q", { i: true, sub: true }]], 6));
push(PN(["with 1/", it("r"), ["q", { i: true, sub: true }], " = 0 when no hit is retrieved. Pk is the probability that two steps ", it("k"), " apart are wrongly judged to be in the same or different segments, with ", it("k"), " half the mean gold segment length; WindowDiff instead compares the number of boundaries inside each window. Both are error rates (lower is better)."]));
push(H2("Ablations"));
push(P(["Beyond the main comparison we run five ablations: (i) signal removal, retraining Stage 1 without slide-text change, topic drift, visual change or CLIP features in turn; (ii) the Stage 1 encoder (BiLSTM, CNN, Transformer); (iii) fusion design, comparing three-token attention, concatenation and pairwise fusion as in [1]; (iv) segment length, varying the 60\u2013600 s bounds; and (iv) a length-matched control, comparing Stage 1 with fixed windows of the same mean length, so that gains from concept-aware boundaries are not confused with gains from simply using longer or shorter units. Segmentation variants are scored with the same text retriever (RRF of BM25 and MiniLM) so that differences come from the units; ablations are reported, not used to choose the main configuration."]));
push(H2("Reproducibility"));
push(P(["Splits are fixed at the lecture level before any test run, training uses three seeds, and every evaluation run stores its configuration, code version, per-query rankings and summary metrics. Pseudo-labels and pseudo-queries are kept in separate files from human annotations so they cannot enter evaluation. ", tbd("Code and annotations will be released at [repository URL] after review.")]));

const D = tbd("–");
const SEG = RES.segmentation || {}, RET = RES.retrieval || {};
const segRow = (label) => { const m = SEG[label]; return m ? [label, f3(m["f1@5"]), f3(m["f1@10"]), f3(m.pk), f3(m.windowdiff), m.n_pred_segments.toFixed(1)] : [label, D, D, D, D, D]; };
const retRow = (label, bold) => { const m = RET[label]; const L = bold ? [[label, { b: true }]] : label; return m ? [L, f3(m["R@1"]), f3(m["R@5"]), f3(m.MRR), f3(m["mIoU@1"])] : [L, D, D, D, D]; };
push(H1("Pilot Results"));
push(...CAP_TAB(2, "Segmentation on pilot lectures"));
if (!Object.keys(SEG).length) push(PN([tbd("Table II needs concept-boundary annotations (not yet made); it fills automatically once they exist.")]));
push(...TABLE(
  ["Segmenter", "F1@15s", "F1@30s", "Pk↓", "WD↓", "#seg"],
  ["Fixed 60 s", "Fixed 120 s", "Shot change", "Topic drift [7]", "Pseudo-label peaks", "ConceptBound Stage 1 (ours)"].map((l) => segRow(l)),
  [1640, 700, 700, 650, 650, 700],
  "Gold: primary annotator. WD = WindowDiff; #seg = mean segments per lecture (gold: –).",
));
push(...CAP_TAB(3, "Retrieval on pilot queries (multi-video search)"));
push(...TABLE(
  ["Segmentation + retriever", "R@1", "R@5", "MRR", "mIoU"],
  [retRow("Fixed 60 s + BM25"), retRow("Fixed 60 s + BGE"), retRow("Fixed 60 s + RRF + rerank"),
   retRow("ConceptBound seg. + BM25"), retRow("ConceptBound seg. + RRF + rerank"), retRow("ConceptBound full (ours)", true),
   retRow("Ground-truth seg. + RRF + rerank (oracle)")],
  [2760, 570, 570, 570, 570],
  "Every query is searched across all pilot lectures. Hit: right lecture and IoU ≥ 0.3. “ConceptBound seg.” = our segmentation with a standard text retriever; oracle = annotators’ segmentation (upper bound). Values are means with 95% bootstrap CIs in the full results.",
));
const AB = (RES.ablations || {}).rows || [];
if (AB.length) {
  const pick = AB.filter((r) => !r.group.startsWith("F") && !r.group.startsWith("B"));
  push(...CAP_TAB(4, "Ablations on pilot queries"));
  push(...TABLE(
    ["Group / variant", "R@1", "MRR", "units"],
    pick.map((r) => [r.group.split(": ")[1] + " \u2014 " + r.variant, f3(r["R@1"]), f3(r.MRR), String(r.n_units)]),
    [3240, 600, 600, 600],
    "Segmentation rows use the RRF(BM25+MiniLM) retriever; fusion rows use their own Stage 2 model on Stage 1 units.",
  ));
}
push(P([tbd("[To be written after the pilot: which segmenter is closest to annotators; whether concept-aware units beat fixed windows of matched mean length (RQ1); whether fusion beats the RRF + re-rank text baseline (RQ2); per-query-type breakdown, e.g., definition vs. example questions; and three representative failure cases from the per-query rankings.]")]));
push(P("A finding we can already report concerns data quality. In the first pipeline version every OCR signal was zero because the OCR engine silently fell back to an empty backend when its weights failed to download; after fixing this and cropping to the detected screen, slide text is found on 69\u201393% of sampled frames in the five lectures processed so far (Table I). Two further silent errors were found: per-step transcripts were 30-s rolling windows, so concatenating them repeated each word about ten times and let the 256-token encoder see only a segment’s first seconds, and Whisper inserted repeated phrases during pre-class noise. All three would have quietly favoured or penalized particular systems, so the released pipeline fails loudly instead of falling back."));

push(H1("Limitations and Ethics"));
push(P(["The benchmark comes from one institution, one course and one camera setup; results may not transfer to other rooms, subjects or languages, and we position the data as a test collection rather than a training set. Whisper’s accuracy on Indian-English classroom speech has not yet been measured ", tbd("(WER on a sampled subset: TBD)"), ". Recordings show students; they were used with ", tbd("written permission of the department and instructors"), ", and only annotations, transcripts and derived features will be released, with faces blurred in any published frame."]));

push(H1("Conclusion"));
push(P("We framed concept-level moment retrieval in single-camera classroom recordings as a benchmark problem, built a two-stage segment-then-retrieve system with a learned boundary detector and cross-modal fusion, and designed an evaluation that pits it against strong text baselines. Future work will complete annotation of the full course, add a student-facing study of revision time, and test whether the findings hold across courses and recording setups."));

push(H1u("Acknowledgment"));
push(P([BLIND ? "Acknowledgments of people and institutions are omitted for double-blind review." : tbd("We thank [guide name] for guidance and [department] for access to the recordings."), " In line with IEEE policy on AI-generated content, we disclose that an AI assistant (Claude, Anthropic) was used to help develop parts of the software pipeline and to draft and edit portions of this manuscript; the authors reviewed all content and take full responsibility for it."]));

push(H1u("References"));
push(...REF_PARAS);

// ---- document -------------------------------------------------------------------
const pageProps = { page: { size: PAGE, margin: MARGIN } };
const doc = new Document({
  creator: BLIND ? "" : "Authors",
  title: "ConceptBound (ICALT 2027 draft)",
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  sections: [
    { properties: { ...pageProps }, children: [title, ...authorBlock, new Paragraph({ children: [], spacing: { after: 240 } })] },
    { properties: { ...pageProps, type: SectionType.CONTINUOUS, column: { count: 2, space: COL_GAP, equalWidth: true } }, children: body },
    // empty continuous section: balances the two columns on the last page (IEEE template rule)
    { properties: { ...pageProps, type: SectionType.CONTINUOUS }, children: [new Paragraph({ children: [] })] },
  ],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
