"""Scoring: claim extraction, deterministic checks, and fact-recall / faithfulness judges.

Two judge implementations share each interface:
- LLM judges (production): rubric prompts, per-item verdicts, evidence verification.
- Heuristic judges (offline): token overlap. Good enough for the demo and tests, not for real scores.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable

from .llm import LLM, extract_json
from .prompts import FACT_PROMPT, FACT_SYSTEM, FAITH_PROMPT, FAITH_SYSTEM
from .schema import Citation, Task
from .text import containment, normalize_numbers, tokens

# ---------------------------------------------------------------- claim extraction

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
# "...in 2021. [3] Next sentence" -> "...in 2021 [3]. Next sentence", so markers stay with their sentence
_TRAILING_MARKERS = re.compile(r"([.!?])((?:\s*\[\d+(?:\s*,\s*\d+)*\])+)")
_SOURCES_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*|\*\*)?(?:sources|references|citations|bibliography)\b.*$", re.I | re.M)


@dataclass
class Claim:
    idx: int
    text: str
    markers: list[int]
    urls: list[str]


def strip_sources_section(report: str) -> str:
    m = _SOURCES_HEADING.search(report)
    return report[: m.start()] if m else report


def extract_claims(report: str, citations: list[Citation]) -> list[Claim]:
    """Every sentence carrying a [n] marker becomes a claim, mapped to the URLs it cites."""
    url_by_marker = {c.marker: c.url for c in citations}
    claims: list[Claim] = []
    for para in strip_sources_section(report).split("\n"):
        para = para.strip().lstrip("-*•>").strip()
        if not para or para.startswith("#") or para.startswith("|"):
            continue
        for sent in _SENT_SPLIT.split(_TRAILING_MARKERS.sub(r"\2\1", para)):
            markers = list(dict.fromkeys(int(x) for m in _MARKER.findall(sent) for x in m.split(",")))
            if not markers:
                continue
            text = re.sub(r"\s+([.,;:!?])", r"\1", _MARKER.sub("", sent)).strip()
            if len(text) < 8:
                continue
            urls = [url_by_marker[m] for m in markers if m in url_by_marker]
            claims.append(Claim(len(claims), text, markers, urls))
    return claims


# ---------------------------------------------------------------- deterministic numeric check

def numeric_fidelity(task: Task, report: str) -> tuple[float | None, list[str]]:
    """Fraction of required numbers that appear verbatim. Catches parsers that mangle digits."""
    required = [(kf.id, n) for kf in task.key_facts for n in kf.numbers]
    if not required:
        return None, []
    norm = normalize_numbers(report)
    missing = [
        f"{fid}:{n}" for fid, n in required
        if not re.search(rf"(?<![\d.]){re.escape(normalize_numbers(n))}(?!\d)", norm)
    ]
    return 1 - len(missing) / len(required), missing


# ---------------------------------------------------------------- judgments

@dataclass
class FactJudgment:
    fact_id: str
    fact_text: str
    present: bool          # final verdict used for scoring (requires verified evidence)
    judge_present: bool    # what the judge said before evidence verification
    evidence: str
    verified: bool
    method: str


@dataclass
class ClaimJudgment:
    claim_idx: int
    claim: str
    urls: list[str]
    verdict: str           # supported | partial | unsupported | no_source
    rationale: str
    method: str


VERDICT_SCORE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0, "no_source": 0.0}


def evidence_in_report(evidence: str, report: str, threshold: float = 0.85) -> bool:
    """Guard against judges 'quoting' sentences that are not actually in the report."""
    ev = tokens(evidence)
    if not ev:
        return False
    if " ".join(evidence.split()).lower() in " ".join(report.split()).lower():
        return True
    rep = tokens(report)
    return containment(ev, set(rep)) >= threshold


@dataclass
class JudgeUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_calls: int = 0

    def add(self, resp) -> None:
        self.calls += 1
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.cached_calls += int(getattr(resp, "cached", False))

    def as_dict(self) -> dict:
        return asdict(self)


class JudgeError(RuntimeError):
    pass


async def _complete_json(llm: LLM, usage: JudgeUsage, system: str, prompt: str, max_tokens: int):
    last: Exception | None = None
    for _ in range(2):  # one retry on malformed JSON
        resp = await llm.complete(system, prompt, max_tokens=max_tokens)
        usage.add(resp)
        try:
            data = extract_json(resp.text)
            if isinstance(data, dict) and isinstance(data.get("judgments"), list):
                return data["judgments"]
            raise ValueError("missing 'judgments' list")
        except ValueError as e:
            last = e
    raise JudgeError(f"judge returned unparseable output: {last}")


class LLMFactJudge:
    method = "llm"

    def __init__(self, llm: LLM, max_tokens: int = 1500):
        self.llm = llm
        self.max_tokens = max_tokens
        self.usage = JudgeUsage()

    async def judge(self, task: Task, report: str) -> list[FactJudgment]:
        if not report.strip():
            return [FactJudgment(f.id, f.text, False, False, "", False, self.method) for f in task.key_facts]
        facts_block = "\n".join(f"- {f.id}: {f.text}" for f in task.key_facts)
        prompt = FACT_PROMPT.replace("<<REPORT>>", report).replace("<<FACTS>>", facts_block)
        items = await _complete_json(self.llm, self.usage, FACT_SYSTEM, prompt, self.max_tokens)
        by_id = {str(i.get("fact_id")): i for i in items if isinstance(i, dict)}
        out = []
        for f in task.key_facts:
            item = by_id.get(f.id, {})
            said = bool(item.get("present", False))
            evidence = str(item.get("evidence") or "")
            verified = said and evidence_in_report(evidence, report)
            out.append(FactJudgment(f.id, f.text, said and verified, said, evidence, verified, self.method))
        return out


class HeuristicFactJudge:
    method = "heuristic"

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.usage = JudgeUsage()

    async def judge(self, task: Task, report: str) -> list[FactJudgment]:
        sentences = [s for s in _SENT_SPLIT.split(_MARKER.sub("", report)) if s.strip()]
        sent_tokens = [set(tokens(s)) for s in sentences]
        out = []
        for f in task.key_facts:
            ft = tokens(f.text)
            best = max(((containment(ft, st), s) for st, s in zip(sent_tokens, sentences)),
                       default=(0.0, ""), key=lambda x: x[0])
            present = best[0] >= self.threshold
            ev = best[1].strip() if present else ""
            out.append(FactJudgment(f.id, f.text, present, present, ev, present, self.method))
        return out


SourceLookup = Callable[[str], "str | None"]


def _select(claims: list[Claim], cap: int) -> list[Claim]:
    """Evenly spaced, deterministic subsample so long reports don't blow up judge cost."""
    if len(claims) <= cap:
        return claims
    step = len(claims) / cap
    return [claims[int(i * step)] for i in range(cap)]


class LLMFaithfulnessJudge:
    method = "llm"

    def __init__(self, llm: LLM, max_claims: int = 12, batch_size: int = 6,
                 max_source_chars: int = 2500, max_tokens: int = 1500):
        self.llm = llm
        self.max_claims = max_claims
        self.batch_size = batch_size
        self.max_source_chars = max_source_chars
        self.max_tokens = max_tokens
        self.usage = JudgeUsage()

    async def judge(self, claims: list[Claim], lookup: SourceLookup) -> list[ClaimJudgment]:
        results: dict[int, ClaimJudgment] = {}
        pending: list[tuple[Claim, list[tuple[str, str]]]] = []
        for c in _select(claims, self.max_claims):
            sources = [(u, t) for u in c.urls if (t := lookup(u))]
            if not sources:  # cites nothing it retrieved: dangling marker or phantom URL
                results[c.idx] = ClaimJudgment(c.idx, c.text, c.urls, "no_source",
                                               "cited source was never retrieved", self.method)
            else:
                pending.append((c, sources))
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            blocks = []
            for j, (c, sources) in enumerate(batch, 1):
                src = "\n".join(f'<source url="{u}">\n{t[: self.max_source_chars]}\n</source>' for u, t in sources)
                blocks.append(f"### C{j}\nClaim: {c.text}\nSource text:\n{src}")
            prompt = FAITH_PROMPT.replace("<<CLAIMS>>", "\n\n".join(blocks))
            items = await _complete_json(self.llm, self.usage, FAITH_SYSTEM, prompt, self.max_tokens)
            by_id = {str(i.get("claim_id")): i for i in items if isinstance(i, dict)}
            for j, (c, _) in enumerate(batch, 1):
                item = by_id.get(f"C{j}", {})
                verdict = str(item.get("verdict", "unsupported")).lower()
                if verdict not in VERDICT_SCORE:
                    verdict = "unsupported"
                results[c.idx] = ClaimJudgment(c.idx, c.text, c.urls, verdict,
                                               str(item.get("rationale", "")), self.method)
        return [results[k] for k in sorted(results)]


class HeuristicFaithfulnessJudge:
    method = "heuristic"

    def __init__(self, max_claims: int = 12):
        self.max_claims = max_claims
        self.usage = JudgeUsage()

    async def judge(self, claims: list[Claim], lookup: SourceLookup) -> list[ClaimJudgment]:
        out = []
        for c in _select(claims, self.max_claims):
            texts = [t for u in c.urls if (t := lookup(u))]
            if not texts:
                out.append(ClaimJudgment(c.idx, c.text, c.urls, "no_source",
                                         "cited source was never retrieved", self.method))
                continue
            overlap = containment(tokens(c.text), set(tokens(" ".join(texts))))
            verdict = "supported" if overlap >= 0.8 else "partial" if overlap >= 0.5 else "unsupported"
            out.append(ClaimJudgment(c.idx, c.text, c.urls, verdict, f"token overlap {overlap:.2f}", self.method))
        return out


def faithfulness_score(judgments: list[ClaimJudgment]) -> float | None:
    if not judgments:
        return None
    return sum(VERDICT_SCORE[j.verdict] for j in judgments) / len(judgments)
