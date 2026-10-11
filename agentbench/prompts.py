"""Judge prompts. Placeholders use <<NAME>> so JSON braces need no escaping."""

FACT_SYSTEM = (
    "You are a strict grader of research reports. You check whether specific facts are stated "
    "in a report. Never use outside knowledge: a fact counts only if the report itself states it."
)

FACT_PROMPT = """<report>
<<REPORT>>
</report>

For each fact below, decide whether the report states it.
- Present: the report states the same information, even in different words or clearly equivalent units.
- Absent: the report omits it, contradicts it, or only hints at it vaguely.
- If a number, date or name in the report differs from the fact, the fact is absent.

Facts:
<<FACTS>>

For each present fact, copy the single report sentence that states it, word for word, as evidence.
Respond with only JSON and no other text:
{"judgments": [{"fact_id": "F1", "present": true, "evidence": "exact sentence from the report"}]}"""

FAITH_SYSTEM = (
    "You verify whether claims in a report are supported by the sources they cite. "
    "Judge only from the source text provided, never from your own knowledge."
)

FAITH_PROMPT = """For each claim, read the source text it cites and give a verdict:
- "supported": the source text states the claim's information.
- "partial": the source supports the gist, but a specific detail in the claim (a number, date, name) is not in the source.
- "unsupported": the source does not state the claim, or contradicts it.

<<CLAIMS>>

Respond with only JSON and no other text:
{"judgments": [{"claim_id": "C1", "verdict": "supported", "rationale": "one short sentence"}]}"""
