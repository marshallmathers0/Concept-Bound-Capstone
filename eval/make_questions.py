"""
Write eval/questions.jsonl from two sources:

  1. DRAFTED: questions written by the assistant from the lecture transcripts,
     paraphrased like a student would ask, each with the answer span (video,
     start, end in seconds). verified=False until a human checks the span
     against the video (see eval/README.md). Times below are mm:ss.
  2. PILOT (human): the queries annotators wrote in
     data/annotations/gold/*.json (verified=True by construction).

Plus out-of-scope questions that must be refused. Each carries the key terms
that must NOT appear in any indexed segment; this script checks that against
the current index and warns if a "out-of-scope" topic is actually covered.

    python -m eval.make_questions
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)


def t(mmss: str) -> float:
    parts = [int(x) for x in mmss.split(":")]
    return float(parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) == 3 else 0))


# (video, question, start mm:ss, end mm:ss, type)
DRAFTED = [
    # Lecture 1: course intro + ML/DL recap
    ("college_lec01", "Which modules does this course cover?", "15:33", "18:00", "other"),
    ("college_lec01", "Which textbooks are recommended for the course?", "18:00", "21:30", "other"),
    ("college_lec01", "What is the difference between binary and multi-class classification?", "24:51", "26:03", "comparison"),
    ("college_lec01", "How is deep learning different from classical machine learning?", "28:18", "34:03", "comparison"),
    ("college_lec01", "Why do we prefer sigmoid or tanh over a step activation function?", "34:03", "37:30", "explanation"),
    ("college_lec01", "What goes wrong with the sigmoid's gradient for very large or very small inputs?", "37:30", "39:33", "explanation"),
    ("college_lec01", "Why were leaky ReLUs introduced after ReLU?", "39:33", "40:33", "explanation"),
    ("college_lec01", "Why do we put a softmax at the output of a classifier?", "48:33", "49:48", "explanation"),
    ("college_lec01", "What is special about the derivatives of tanh and sigmoid?", "49:48", "52:39", "explanation"),
    ("college_lec01", "How does gradient descent move towards the optimum, using linear regression as the example?", "52:39", "58:03", "explanation"),
    ("college_lec01", "What counts as a hyperparameter in a neural network?", "58:03", "01:00:03", "definition"),
    # Lecture 2: RNNs, long-term dependencies, LSTM, autoencoders
    ("college_lec02", "How is a recurrent neural network different from a feed-forward network?", "03:12", "08:00", "comparison"),
    ("college_lec02", "How do we apply backpropagation to an RNN?", "08:00", "09:33", "explanation"),
    ("college_lec02", "Why does an RNN struggle to remember words from far back in a long sentence?", "09:33", "11:39", "explanation"),
    ("college_lec02", "What problem was the LSTM designed to solve?", "10:33", "11:39", "explanation"),
    ("college_lec02", "What are the parts of an autoencoder?", "11:39", "13:00", "definition"),
    ("college_lec02", "Which video series does the instructor suggest for revising deep learning?", "15:30", "17:27", "other"),
    # Lecture 3: LLM basics, architectures, prompting, decoding
    ("college_lec03", "Are tokens and words the same thing for a language model?", "18:00", "20:00", "comparison"),
    ("college_lec03", "What happens to the next-word probability distribution as each new token is added to the context?", "22:00", "24:00", "explanation"),
    ("college_lec03", "What is a masked language model and how is it trained?", "28:00", "30:00", "definition"),
    ("college_lec03", "Why can the output length differ from the input length in an encoder-decoder model?", "30:00", "32:00", "explanation"),
    ("college_lec03", "How can sentiment analysis be done as conditional generation with an LLM?", "34:00", "36:00", "application"),
    ("college_lec03", "How can question answering be turned into a word prediction problem?", "36:00", "38:00", "application"),
    ("college_lec03", "What is prompt engineering?", "38:00", "40:00", "definition"),
    ("college_lec03", "Does in-context learning change the weights of the model?", "40:00", "43:00", "explanation"),
    ("college_lec03", "How are the model's raw output scores converted into probabilities?", "44:00", "51:00", "explanation"),
    ("college_lec03", "What is greedy decoding and why is it not always a good idea?", "52:00", "56:00", "explanation"),
    ("college_lec03", "How is random sampling different from always picking the most likely word?", "56:00", "58:00", "comparison"),
    # Lecture 4: n-gram language models
    ("college_lec04", "What exactly does a language model predict?", "20:00", "22:30", "definition"),
    ("college_lec04", "What do we mean by the context when predicting the next word?", "22:30", "25:00", "definition"),
    ("college_lec04", "How is the probability of a whole sentence computed with the chain rule?", "27:30", "32:30", "derivation"),
    ("college_lec04", "What is the Markov assumption in n-gram models?", "32:30", "37:30", "definition"),
    ("college_lec04", "How does a trigram model approximate the full history?", "37:30", "40:00", "explanation"),
    ("college_lec04", "Why can't n-gram models handle long-distance dependencies?", "40:00", "42:30", "explanation"),
    ("college_lec04", "Where are n-gram models still useful compared to LLMs?", "45:00", "47:30", "application"),
    ("college_lec04", "How do you estimate bigram probabilities from counts?", "47:30", "50:00", "procedure"),
    ("college_lec04", "How do you read the bigram probability table built from a corpus?", "50:00", "55:00", "example"),
    ("college_lec04", "Why are language model probabilities stored as log probabilities?", "55:00", "57:30", "explanation"),
    # Lecture 5: LM evaluation, sampling, intro to LLMs
    ("college_lec05", "What is the difference between extrinsic and intrinsic evaluation of a language model?", "00:00", "04:00", "comparison"),
    ("college_lec05", "What is perplexity used for?", "03:00", "08:00", "definition"),
    ("college_lec05", "Why must the test set be different from the training data when evaluating a language model?", "04:00", "06:03", "explanation"),
    ("college_lec05", "How do you sample a word from a unigram distribution using probability intervals?", "08:00", "15:21", "procedure"),
    ("college_lec05", "How can a bigram model be used to generate a sentence?", "15:21", "17:00", "procedure"),
    ("college_lec05", "How does a simple n-gram model assign probabilities to word sequences?", "21:33", "24:57", "explanation"),
    ("college_lec05", "Where does a large language model get its knowledge from?", "24:57", "27:03", "explanation"),
    ("college_lec05", "What does an LLM take as input and what does it output?", "27:03", "29:18", "definition"),
    ("college_lec05", "How does an LLM generate a whole passage of text?", "29:18", "32:00", "explanation"),
    ("college_lec05", "What is the difference between decoder-only, encoder-only and encoder-decoder models?", "32:00", "38:30", "comparison"),
    ("college_lec05", "Which tasks are encoder-decoder models popular for?", "38:30", "39:33", "application"),
]

# (question, regex of terms that must not occur in any indexed segment)
OUT_OF_SCOPE = [
    ("What is the time complexity of quicksort?", r"quick ?sort"),
    ("Explain Dijkstra's shortest path algorithm.", r"dijkstra"),
    ("How do I configure a VLAN on a Cisco switch?", r"vlan|cisco"),
    ("What happens during photosynthesis?", r"photosynth|chlorophyll"),
    ("Who won the 2011 cricket world cup?", r"cricket|world cup"),
    ("How does TLS use public key cryptography?", r"\btls\b|public key|cryptograph"),
    ("State Newton's second law of motion.", r"newton"),
    ("How do you rebalance a red-black tree after an insertion?", r"red.black"),
    ("How does Kubernetes autoscale pods?", r"kubernetes|autoscal"),
    ("What are the steps of the TCP three-way handshake?", r"handshake|\btcp\b"),
]


def pilot_questions():
    from src.utils.gold import list_gold_videos, load_gold
    out = []
    for vid in list_gold_videos():
        for i, q in enumerate(load_gold(vid)["queries"]):
            out.append({"question": q["query"], "video_id": vid, "start": float(q["start_time"]),
                        "end": float(q["end_time"]), "type": q.get("type", "other"),
                        "source": "pilot_human", "verified": True, "annotator": q.get("annotator")})
    return out


def main():
    rows = []
    for i, (vid, q, s, e, typ) in enumerate(DRAFTED, start=1):
        rows.append({"id": f"d{i:02d}", "question": q, "video_id": vid, "start": t(s), "end": t(e),
                     "type": typ, "in_scope": True, "source": "drafted", "verified": False})
    for i, q in enumerate(pilot_questions(), start=1):
        rows.append({"id": f"h{i:02d}", "in_scope": True, **q})
    meta_path = os.path.join(ROOT, "data", "rag_index", "segments_metadata.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else []
    for i, (q, pat) in enumerate(OUT_OF_SCOPE, start=1):
        found = [m["segment_id"] for m in meta if re.search(pat, (m["transcript_text"] + " " + m["ocr_text"]).lower())]
        if found:
            print(f"WARNING: out-of-scope question {q!r} matches indexed segments {found[:3]}; replace it.")
        rows.append({"id": f"o{i:02d}", "question": q, "in_scope": False, "source": "drafted",
                     "verified": False, "absent_terms": pat})
    path = os.path.join(ROOT, "eval", "questions.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_in = sum(r["in_scope"] for r in rows)
    print(f"Wrote {len(rows)} questions ({n_in} in-scope: {len(DRAFTED)} drafted + {n_in - len(DRAFTED)} human; "
          f"{len(rows) - n_in} out-of-scope) -> {path}")


if __name__ == "__main__":
    main()
