"""Cleaning Agent -- picks the right SAP master equipment row for one worklist
row and returns a cleaned EquipmentNo / SubLoc1 with evidence.

Tool-calling LangChain agent (via create_agent) with:
  - Mandatory tool-call prompt (model must call submit_cleaning_result, cannot
    reply in natural language) -- matches the proven backup's strict-JSON contract.
  - Robust result capture: thread-local primary path + message-scan fallback,
    so we always get the args even if the @tool body didn't physically execute.
  - Rate-limit retry via with_rate_limit_retry.
  - Langfuse tracing via CallbackHandler.

Public API (unchanged):
    cleaning_agent(row, candidates_df, cfg) -> dict
"""

import json
import threading

from langchain.agents import create_agent
from langchain_core.tools import tool

from config import Config, load_config
from utils import (
    PREFIX_TO_SUBLOC1,
    build_langfuse,
    build_langfuse_config,
    build_llm,
    with_rate_limit_retry,
)


# =========================================================================
# Tool (thread-local result slot kept as the primary capture path)
# =========================================================================
_result_tls = threading.local()


@tool
def submit_cleaning_result(
    suggested_equipment_no: str,
    equipment_no_confidence: str,
    equipment_no_evidence: str,
    suggested_subloc1: str,
    subloc1_confidence: str,
    subloc1_evidence: str,
    needs_review: bool,
) -> str:
    """Submit the FINAL cleaning decision for the worklist row. You MUST call
    this exactly ONCE on every invocation -- even when no candidate is
    plausible (use suggested_equipment_no='null' + 'Low' confidence then).

    Arguments:
      suggested_equipment_no: the picked master EquipmentNo, or 'null' if none.
      equipment_no_confidence: 'High' | 'Medium' | 'Low'.
      equipment_no_evidence: one sentence explaining what signal convinced you.
      suggested_subloc1: the worklist SubLoc1 alias (corrected if you overrode it).
      subloc1_confidence: 'High' | 'Medium' | 'Low'.
      subloc1_evidence: one sentence; mention platform override if any.
      needs_review: true if a human should double-check this row.
    """
    _result_tls.result = _args_to_result({
        "suggested_equipment_no": suggested_equipment_no,
        "equipment_no_confidence": equipment_no_confidence,
        "equipment_no_evidence": equipment_no_evidence,
        "suggested_subloc1": suggested_subloc1,
        "subloc1_confidence": subloc1_confidence,
        "subloc1_evidence": subloc1_evidence,
        "needs_review": needs_review,
    })
    return "Cleaning decision recorded."


def _args_to_result(args: dict) -> dict:
    """Normalize the tool-call args into our canonical result dict shape."""
    return {
        "suggested_equipment_no":
            None if str(args.get("suggested_equipment_no", "")).strip().lower() == "null"
            else args.get("suggested_equipment_no"),
        "equipment_no_confidence": args.get("equipment_no_confidence", "Low"),
        "equipment_no_evidence":   args.get("equipment_no_evidence", ""),
        "suggested_subloc1":       args.get("suggested_subloc1", ""),
        "subloc1_confidence":      args.get("subloc1_confidence", "Low"),
        "subloc1_evidence":        args.get("subloc1_evidence", ""),
        "needs_review":            bool(args.get("needs_review", True)),
    }


# =========================================================================
# Module state (lazy init on first call)
# =========================================================================
_cfg: Config | None = None
_llm = None
_langfuse_client = None
_langfuse_handler = None
_agent = None


def _init(cfg: Config | None) -> None:
    global _cfg, _llm, _langfuse_client, _langfuse_handler, _agent
    if _agent is not None and (cfg is None or cfg is _cfg):
        return
    _cfg = cfg or load_config()
    print(f"[cleaning_agent] initialising LLM ({_cfg.llm.model}, max_tokens={_cfg.llm.max_tokens_cleaning}) + Langfuse handler + 1 tool")
    _llm = build_llm(_cfg, max_tokens=_cfg.llm.max_tokens_cleaning)
    _langfuse_client, _langfuse_handler = build_langfuse()
    _agent = create_agent(
        model=_llm,
        tools=[submit_cleaning_result],
        system_prompt=_system_prompt(_cfg),
    )
    print("[cleaning_agent] agent ready")


# =========================================================================
# System prompt (matches the proven backup verbatim, with a hardened "you MUST
# call the tool" output contract so Claude cannot reply in natural language)
# =========================================================================
def _system_prompt(cfg: Config) -> str:
    reverse = "\n".join(f"  {k} -> {v}" for k, v in PREFIX_TO_SUBLOC1.items())
    loc = cfg.plant.location_filter.title()
    return f"""You are mapping ONE row of a Petronas {loc} SAP maintenance worklist to its correct equipment in the SAP equipment master.

CONTEXT -- how the two data sources look:

Worklist row (human-entered, often messy):
  - EquipmentNo: a free-form field. Sometimes a clean SAP tag ("060SDV-002"), sometimes a
    slash-separated multi-tag ("JB-7702/JB-7771"), sometimes a tag buried in descriptive text
    ("M-3 P-0110B", "ML-3 BO 605 (WHCP)", "SI Line Replacement - 50-SI-68011-13411X"), and
    sometimes almost pure description.
  - SubLoc1: a short platform alias (BOCPP-A, BODP-F, BOK-A, ...). Usually correct but
    occasionally mislabeled (~1% of rows).
  - Discipline, EqptType, Job Description, FuncLoc: additional context.

Master row (clean, authoritative SAP data):
  - TechnicalIdentNo: structured as "<PREFIX>-<TAIL>" where PREFIX is a 4-letter platform code
    (BOCP, BOKA, BOPA, BOVA, BODA, BODB, BODC, BODD, BODE, BODF) and TAIL is the real SAP tag.
    Examples: "BOKA-UPS7700A", "BODD-P0110B", "BODF-50-SI-68011-13411X", "BOCP-WHCP-ML3-BO605".
  - EquipmentNo: a short SAP code (often numeric, e.g. "11061761", or "PIPE-C-LP-0258").
  - FunctionalLocation: hierarchical, e.g. "BOKA.PWR.UPS.UPS7700A".
  - ObjectDescription, FunctLocDescrip, CatalogProfile: human-readable descriptors.

The candidates in the user message (up to 100) are ALL the master equipment under the platform
implied by this row's SubLoc1 -- that is the ONLY narrowing. There is no catalog_profile /
Discipline pre-filter, so the right equipment is in here but you must find it yourself. The
exact-match step has already run and failed, so the worklist tag and master TID are NOT
byte-identical -- you need to RECOGNIZE the match from naming patterns, descriptions, and
FunctionalLocation.

YOUR TASK:
Pick the single master row that IS the equipment this worklist row refers to. Use your judgment:
  - The real SAP tag inside the worklist EquipmentNo may be surrounded by module numbers,
    action words, descriptive text, or punctuation. Find it.
  - A match is strong when a candidate's TechnicalIdentNo TAIL (the part after the 4-letter
    prefix) appears inside the worklist EquipmentNo, or vice versa, after you ignore
    whitespace, dashes, and parentheses.
  - When no string overlap works, use the Job Description to pick the most likely candidate
    semantically against the candidate's ObjectDescription and FunctLocDescrip.
  - Only return nulls (confidence Low) when no candidate is plausible under ANY of these angles.

Then compute Suggested SubLoc1:
  Take the PREFIX from the picked TechnicalIdentNo (text before the FIRST '-') and map it:
{reverse}
  If this differs from the row's original SubLoc1, OVERRIDE it and explain in subloc1_evidence.

Also decide whether a human should double-check this row (needs_review):
  Set needs_review = true WHENEVER you want a human to verify your pick, EVEN IF your confidence
  is High. Examples:
    - Two candidates were almost equally plausible and you picked the better one by a small margin.
    - You overrode SubLoc1 (platform disagreement).
    - The worklist EquipmentNo was mostly free-text and you had to infer the tag.
    - Multi-tag worklist input where you had to pick one of several possible equipment.
  Set needs_review = false only when your match is unambiguous and nothing is unusual about it.

Rules:
  - Pick only from the candidates provided; never invent an EquipmentNo.
  - For each of the two output fields (EquipmentNo, SubLoc1), return confidence (High / Medium /
    Low) and a one-sentence evidence describing what signal convinced you.

OUTPUT CONTRACT -- MANDATORY:
You MUST call the submit_cleaning_result tool EXACTLY ONCE as your final action.
Do NOT reply in natural language. Do NOT describe your reasoning in text. The
tool call IS your response. Even when no candidate is plausible, you must still
call the tool with suggested_equipment_no='null' and 'Low' confidence -- never
skip the call. There is no acceptable response that is not a tool call.
"""


# =========================================================================
# Result capture: inspect the invoke response for the tool_call args as a
# robust fallback to the thread-local write (which only fires if the @tool
# body physically executes).
# =========================================================================
def _extract_result_from_response(response) -> dict | None:
    messages = response.get("messages", []) if isinstance(response, dict) else []
    for msg in messages:
        tcs = getattr(msg, "tool_calls", None) or []
        for tc in tcs:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name == "submit_cleaning_result":
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                return _args_to_result(args or {})
    return None


# =========================================================================
# Public entry point (signature unchanged for main.py)
# =========================================================================
def cleaning_agent(row, candidates_df, cfg: Config | None = None) -> dict:
    """Resolve ONE worklist row against its candidate master equipment. Thread-safe.

    Returns a dict with: suggested_equipment_no, equipment_no_confidence,
    equipment_no_evidence, suggested_subloc1, subloc1_confidence,
    subloc1_evidence, needs_review -- OR {"_error": "..."} on failure.
    """
    _init(cfg)
    _result_tls.result = None

    payload = {
        "worklist_row": {
            "EquipmentNo":     str(row.get("EquipmentNo", "")),
            "FuncLoc":         str(row.get("FuncLoc", "")),
            "Discipline":      str(row.get("Discipline", "")),
            "EqptType":        str(row.get("EqptType", "")),
            "SubLoc1":         str(row.get("SubLoc1", "")),
            "Job Description": str(row.get("Job Description", "")),
        },
        "candidates": candidates_df[[
            "EquipmentNo", "TechnicalIdentNo", "FunctionalLocation",
            "ObjectDescription", "FunctLocDescrip", "CatalogProfile",
        ]].to_dict(orient="records"),
    }
    worklist_id = str(row.get("WorklistID") or row.get("ID") or "unknown")

    config = build_langfuse_config(
        _cfg,
        run_name=f"{_cfg.langfuse.agent_name} - Cleaning {worklist_id}",
        session_id=_cfg.langfuse.session_cleaning,
        callback_handler=_langfuse_handler,
        tags=["cleaning", worklist_id],
        metadata={
            "worklist_id": worklist_id,
            "subloc1": str(row.get("SubLoc1", "")),
            "candidates": len(candidates_df),
        },
    )

    print(f"[cleaning_agent] invoking agent for WorklistID={worklist_id} "
          f"(SubLoc1={row.get('SubLoc1','')}, {len(candidates_df)} candidates)")

    def _invoke():
        return _agent.invoke(
            {"messages": [{"role": "user",
                           "content": json.dumps(payload, ensure_ascii=False, default=str)}]},
            config=config,
        )

    try:
        response = with_rate_limit_retry(_invoke, cfg=_cfg)
    except Exception as e:
        print(f"[cleaning_agent] FAILED for {worklist_id}: {e}")
        return {"_error": f"agent invocation failed: {e}"}

    # Primary capture: thread-local slot (set by the @tool body on execution).
    result = getattr(_result_tls, "result", None)

    # Fallback: scan the response messages for a submit_cleaning_result tool call.
    if not result:
        result = _extract_result_from_response(response)

    if result:
        picked = result.get("suggested_equipment_no") or "null"
        print(f"[cleaning_agent] -> {worklist_id}: picked={picked} "
              f"conf={result.get('equipment_no_confidence','?')} "
              f"needs_review={result.get('needs_review')}")
        return result

    print(f"[cleaning_agent] -> {worklist_id}: agent did NOT call submit_cleaning_result")
    return {"_error": "agent did not call submit_cleaning_result"}
