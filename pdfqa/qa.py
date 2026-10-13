"""Answer questions from the indexed documents, always with page citations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .index import Index, tokenize
from .ingest import is_heading, split_sentences
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


def answer(question: str, index: Index, llm: LLM | None = None, k: int = 4) -> Answer:
    hits = index.search(question, k=k)
    if not hits:
        return Answer(NOT_FOUND, [], "none")
    sources = [Source(n, chunk.citation, chunk.text, round(score, 4))
               for n, (chunk, score) in enumerate(hits, start=1)]
    if llm is None:
        text = _extractive(question, sources)
        return Answer(text, _cited_only(text, sources) if text != NOT_FOUND else [], "extractive")
    text = llm.complete(SYSTEM_PROMPT, build_prompt(question, sources)).strip()
    if NOT_FOUND.rstrip(".") in text:
        return Answer(NOT_FOUND, [], llm.name)
    return Answer(text, _cited_only(text, sources), llm.name)


def _cited_only(text: str, sources: list[Source]) -> list[Source]:
    """Keep only the sources the model actually cited, so the UI never shows unused pages."""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    return [s for s in sources if s.n in cited] or sources


def _extractive(question: str, sources: list[Source], max_sentences: int = 3) -> str:
    """No-LLM fallback: return the sentences that overlap most with the question."""
    terms = set(tokenize(question))
    scored = []
    for source in sources:
        for position, sentence in enumerate(split_sentences(source.text)):
            if is_heading(sentence):
                continue
            overlap = len(terms & set(tokenize(sentence)))
            if overlap:
                scored.append((overlap, -source.n, -position, sentence, source.n))
    if not scored:
        return NOT_FOUND
    scored.sort(reverse=True)
    top = scored[0][0]
    # drop sentences that match much less of the question than the best one
    best = [s for s in scored if s[0] * 2 >= top][:max_sentences]
    return " ".join(f"{sentence} [{n}]" for *_, sentence, n in best)
