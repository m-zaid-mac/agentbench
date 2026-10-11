import asyncio

from agentbench.judges import (LLMFactJudge, LLMFaithfulnessJudge, evidence_in_report,
                               extract_claims, faithfulness_score, numeric_fidelity)
from agentbench.llm import extract_json
from agentbench.schema import Citation, KeyFact, Task
from tests.helpers import ScriptedLLM

REPORT = """# Report

JWST launched on 25 December 2021 [1]. Its mirror is 6.5 metres wide. [2] It sits near L2 [1, 3].

## Sources
[1] NASA - https://nasa.gov/jwst
[2] ESA - https://esa.int/jwst
"""
CITES = [Citation(1, "https://nasa.gov/jwst"), Citation(2, "https://esa.int/jwst")]
TASK = Task("jwst", "q", "space", [KeyFact("F1", "JWST launched in 2021", ["2021"]),
                                    KeyFact("F2", "mirror is 6.5 m", ["6.5"]),
                                    KeyFact("F3", "made of 18 segments", ["18"])])


def test_extract_claims_handles_marker_positions_and_sources_section():
    claims = extract_claims(REPORT, CITES)
    assert [c.markers for c in claims] == [[1], [2], [1, 3]]
    assert claims[1].text == "Its mirror is 6.5 metres wide."
    assert claims[2].urls == ["https://nasa.gov/jwst"]  # [3] has no source entry
    assert all("nasa.gov" not in c.text for c in claims)


def test_numeric_fidelity_catches_mangled_digits():
    score, missing = numeric_fidelity(TASK, REPORT)
    assert abs(score - 2 / 3) < 1e-9 and missing == ["F3:18"]
    mangled = REPORT.replace("6.5", "5")
    score, missing = numeric_fidelity(TASK, mangled)
    assert "F2:6.5" in missing
    assert numeric_fidelity(Task("x", "q", "c", [KeyFact("F1", "c is 299,792,458", ["299,792,458"])]),
                            "speed 299792458 m/s")[0] == 1.0


def test_numeric_fidelity_no_partial_matches():
    t = Task("x", "q", "c", [KeyFact("F1", "86", ["86"])])
    assert numeric_fidelity(t, "height 8848.86 m")[0] == 0.0


def test_fact_judge_overturns_unverifiable_evidence():
    llm = ScriptedLLM([{"judgments": [
        {"fact_id": "F1", "present": True, "evidence": "JWST launched on 25 December 2021 [1]."},
        {"fact_id": "F2", "present": True, "evidence": "The mirror is gold-plated beryllium."},  # not in report
        {"fact_id": "F3", "present": False, "evidence": ""},
    ]}])
    out = asyncio.run(LLMFactJudge(llm).judge(TASK, REPORT))
    assert [f.present for f in out] == [True, False, False]
    assert out[1].judge_present and not out[1].verified


def test_fact_judge_retries_bad_json_then_succeeds():
    llm = ScriptedLLM(["not json at all", '```json\n{"judgments": []}\n```'])
    out = asyncio.run(LLMFactJudge(llm).judge(TASK, REPORT))
    assert len(llm.prompts) == 2 and not any(f.present for f in out)


def test_faithfulness_judge_marks_unretrieved_sources_without_llm_call():
    claims = extract_claims(REPORT, CITES)
    sources = {"https://nasa.gov/jwst": "JWST launched 25 December 2021 and orbits L2."}
    llm = ScriptedLLM([{"judgments": [{"claim_id": "C1", "verdict": "supported"},
                                      {"claim_id": "C2", "verdict": "partial"}]}])
    out = asyncio.run(LLMFaithfulnessJudge(llm).judge(claims, sources.get))
    assert [j.verdict for j in out] == ["supported", "no_source", "partial"]
    assert len(llm.prompts) == 1
    assert faithfulness_score(out) == 0.5


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure! ```json\n{"a": [1, 2]}\n``` done') == {"a": [1, 2]}


def test_evidence_check():
    assert evidence_in_report("JWST launched on 25 December 2021", REPORT)
    assert not evidence_in_report("Hubble launched in 1990", REPORT)
