"""Text normalisation shared by every lexical retriever."""

import re
from typing import List

# Small English stop-word list (kept local to avoid an NLTK download).
STOPWORDS = set("""
a an the and or but if then else of to in on at by for with from into onto over under
is are was were be been being am do does did done have has had having it its this that
these those there here what which who whom whose when where why how all any both each
few more most other some such no nor not only own same so than too very can will just
should could would may might must shall i you he she we they me him her us them my your
his our their mine yours ours theirs about above below up down out off again further once
let lets okay ok um uh yeah right like actually basically
""".split())


def bm25_tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens without stop words. Keeps digits so that
    e.g. 'f1', 'l2' or 'n-gram' style terms survive as 'f1', 'l2', 'n', 'gram'."""
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t not in STOPWORDS]
