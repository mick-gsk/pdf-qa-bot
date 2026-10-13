"""Search index: BM25 keyword search, optionally fused with embedding search.

BM25 runs offline with no dependencies. When an embedding function is given,
results from both rankings are merged with Reciprocal Rank Fusion, which keeps
exact names and numbers findable (BM25) while also matching paraphrases
(embeddings).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import snowballstemmer

from .ingest import Chunk

EmbedFn = Callable[[Sequence[str]], list[list[float]]]

_STOPWORDS = set(
    """a an and are as at be by for from has have how i in is it of on or that the this to was what when where
    which who why will with you your can do does did my me we our much many there about get
    should would could often long take takes need mean means tell any
    der die das und ist ein eine einen zu mit von im den des für auf nicht wie
    was wer wann wo bei sich es kann ich mein wir viel viele gibt lange soll muss""".split()
)
_TOKEN = re.compile(r"\w+", re.UNICODE)
_stem = snowballstemmer.stemmer("english").stemWord
# compared after stemming, so "getting" is dropped like "get"
_STOP_STEMS = {_stem(w) for w in _STOPWORDS}


def tokenize(text: str) -> list[str]:
    """Lower-case words without stopwords, stemmed so that "descaling" matches "descale"."""
    stems = (_stem(t) for t in _TOKEN.findall(text.lower()) if len(t) > 1 or t.isdigit())
    return [s for s in stems if s not in _STOP_STEMS]


class Index:
    def __init__(self, chunks: list[Chunk] | None = None, embed: EmbedFn | None = None,
                 k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.embed = embed
        self.chunks: list[Chunk] = []
        self._tokens: list[Counter] = []
        self._lengths: list[int] = []
        self._df: Counter = Counter()
        self._vectors: list[list[float]] = []
        if chunks:
            self.add(chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tokens = Counter(tokenize(chunk.text))
            self.chunks.append(chunk)
            self._tokens.append(tokens)
            self._lengths.append(sum(tokens.values()))
            self._df.update(tokens.keys())
        if self.embed and chunks:
            self._vectors.extend(self.embed([c.text for c in chunks]))

    def idf(self, term: str) -> float:
        """BM25 inverse document frequency; terms that appear nowhere get the highest weight."""
        n, df = len(self.chunks), self._df[term]
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def documents(self) -> list[str]:
        return sorted({c.doc for c in self.chunks})

    # --- ranking -------------------------------------------------------

    def bm25(self, query: str) -> list[tuple[int, float]]:
        terms = tokenize(query)
        n = len(self.chunks)
        if not n or not terms:
            return []
        avg_len = sum(self._lengths) / n
        scores = []
        for i, tokens in enumerate(self._tokens):
            score = 0.0
            for term in terms:
                tf = tokens.get(term, 0)
                if not tf:
                    continue
                idf = self.idf(term)
                norm = tf + self.k1 * (1 - self.b + self.b * self._lengths[i] / avg_len)
                score += idf * tf * (self.k1 + 1) / norm
            if score > 0:
                scores.append((i, score))
        return sorted(scores, key=lambda s: s[1], reverse=True)

    def vector(self, query: str) -> list[tuple[int, float]]:
        if not self.embed or not self._vectors:
            return []
        q = self.embed([query])[0]
        scores = [(i, _cosine(q, v)) for i, v in enumerate(self._vectors)]
        return sorted(scores, key=lambda s: s[1], reverse=True)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        keyword = self.bm25(query)
        semantic = self.vector(query)
        if not semantic:
            return [(self.chunks[i], s) for i, s in keyword[:k]]
        fused = _rrf([keyword, semantic])
        return [(self.chunks[i], s) for i, s in fused[:k]]

    # --- persistence ---------------------------------------------------

    def save(self, path: str | Path) -> None:
        data = {"chunks": [asdict(c) for c in self.chunks], "vectors": self._vectors,
                "embedding_model": _model_name(self.embed) if self._vectors else ""}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, embed: EmbedFn | None = None) -> Index:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(embed=None)
        index.add([Chunk(**c) for c in data["chunks"]])
        index.embed = embed
        index._vectors = data.get("vectors") or []
        if embed and index.chunks and (len(index._vectors) != len(index.chunks)
                      or data.get("embedding_model") != _model_name(embed)):
            # saved without vectors or with another embedding model: rebuild them once and
            # save, so the next start doesn't embed every chunk again
            index._vectors = embed([c.text for c in index.chunks])
            index.save(path)
        return index


def _model_name(embed: EmbedFn | None) -> str:
    """Which embedding model made the vectors; get_embedder() sets `model` on its functions."""
    if embed is None:
        return ""
    return str(getattr(embed, "model", None) or getattr(embed, "__qualname__", ""))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (i, _) in enumerate(ranking, start=1):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda s: s[1], reverse=True)
