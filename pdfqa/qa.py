"""Answer questions from the indexed documents, always with page citations."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from .index import Index, tokenize
from .ingest import Chunk, is_heading, split_sentences
from .llm import LLM

NOT_FOUND = "I couldn't find this in the uploaded documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered excerpts provided.
Rules:
- Cite every statement with the excerpt number in square brackets, e.g. [1] or [2][3].
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
- Do not use outside knowledge. Do not guess.
- Answer in the language of the question. Be concise."""


@dataclass
class Source:
    n: int
    citation: str
    text: str
    score: float
    doc: str = ""
    page: int = 0


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    mode: str = "extractive"

    @property
    def found(self) -> bool:
        return self.text != NOT_FOUND


def build_prompt(question: str, sources: list[Source]) -> str:
    excerpts = "\n\n".join(f"[{s.n}] ({s.citation})\n{s.text}" for s in sources)
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def retrieve(question: str, index: Index, k: int = 4,
             hits: list[tuple[Chunk, float]] | None = None) -> list[Source]:
    """The top k chunks as numbered sources; pass `hits` to reuse a search that already ran."""
    if hits is None:
        hits = index.search(question, k=k)
    return [Source(n, chunk.citation, chunk.text, round(score, 4), chunk.doc, chunk.page)
            for n, (chunk, score) in enumerate(hits, start=1)]


def answer(question: str, index: Index, llm: LLM | None = None, k: int = 4,
           hits: list[tuple[Chunk, float]] | None = None) -> Answer:
    return _answer(question, retrieve(question, index, k, hits), index, llm)


def _answer(question: str, sources: list[Source], index: Index, llm: LLM | None) -> Answer:
    if not sources:
        return Answer(NOT_FOUND, [], "none")
    if llm is None:
        return _finish(_extractive(question, sources, index), sources, "extractive")
    return _finish(llm.complete(SYSTEM_PROMPT, build_prompt(question, sources)), sources, llm.name)


def answer_stream(question: str, index: Index, llm: LLM | None = None,
                  k: int = 4) -> Iterator[tuple[str, str | Answer]]:
    """Like answer(), but yields ("token", text) pieces as the model writes and ends with ("done", Answer).

    The final Answer is authoritative: if the model ends up saying it found nothing, the
    client should replace the streamed text with it.
    """
    sources = retrieve(question, index, k)
    if not sources or llm is None:
        result = _answer(question, sources, index, llm)  # reuse the search above
        yield "token", result.text
        yield "done", result
        return
    parts = []
    for piece in llm.stream(SYSTEM_PROMPT, build_prompt(question, sources)):
        parts.append(piece)
        yield "token", piece
    yield "done", _finish("".join(parts), sources, llm.name)


def _finish(text: str, sources: list[Source], mode: str) -> Answer:
    text = text.strip()
    if not text or NOT_FOUND.rstrip(".") in text:
        return Answer(NOT_FOUND, [], mode)
    return Answer(text, _cited_only(text, sources), mode)


def _cited_only(text: str, sources: list[Source]) -> list[Source]:
    """Keep only the sources the model actually cited, so the UI never shows unused pages."""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    return [s for s in sources if s.n in cited] or sources


def _extractive(question: str, sources: list[Source], index: Index, max_sentences: int = 3,
                min_coverage: float = 0.5) -> str:
    """No-LLM fallback: return the sentences that cover most of the question.

    Question words are weighted by IDF, so a sentence that only shares a common word
    ("company", "machine") with the question does not count as an answer, and a question
    whose key word appears nowhere in the documents is answered with NOT_FOUND.
    """
    # An unseen word counts like a rare one, not more: it is as likely a synonym ("limit" for
    # "reimbursed up to") as a sign that the topic is missing.
    weights = {t: index.idf(t, min_df=1) for t in set(tokenize(question))}
    total = sum(weights.values())
    if not total:
        return NOT_FOUND
    scored, seen = [], set()
    for source in sources:
        for position, sentence in enumerate(split_sentences(source.text)):
            if is_heading(sentence) or len(sentence.split()) < 3 or sentence in seen:
                continue
            seen.add(sentence)  # chunks overlap, so the same sentence can appear twice
            coverage = sum(w for t, w in weights.items() if t in set(tokenize(sentence))) / total
            if coverage:
                scored.append((coverage, -source.n, -position, sentence, source.n))
    if not scored:
        return NOT_FOUND
    scored.sort(reverse=True)
    top = scored[0][0]
    if top < min_coverage:
        return NOT_FOUND
    # drop sentences that match much less of the question than the best one
    best = [s for s in scored if s[0] * 2 >= top][:max_sentences]
    return " ".join(f"{sentence} [{n}]" for *_, sentence, n in best)
