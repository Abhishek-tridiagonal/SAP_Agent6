"""Classify worklist rows that the Cleaning Agent could NOT match against the
SAP equipment master. Instead of silently dropping them, give each row a valid
Equipment Class at a coarser hierarchy level (based on Discipline + EqptType
+ Job Description) so they still land in the final Job Library.

Approach (per unresolved row):
  Step A - deterministic: Discipline -> Level-1 class code (free, no LLM).
  Step B - LLM narrowing: walk fallback_levels fine->coarse under the Step A
           root; agent picks the single best class code at the finest level
           where any plausible match exists.
  Fallback: Level-1 code itself (always succeeds).

Output: cfg.unresolved_items -- an xlsx deliverable with
  raw worklist fields + Classified_Class + Classified_Level + Classifier_Evidence
  + blank MANUAL_EquipmentNo / MANUAL_TechnicalIdentNo / REVIEWER_NOTES
  columns for a human to override on the next run.

The classified rows are also returned so the orchestrator can inject them into
the Job Library Agent run as "virtual" worklist rows under their inferred
Class (so counters are created for them).

Public API:
    classify_unresolved(cfg, cleaned_worklist, equipment_class_df)
        -> (classified_df, written_path)
"""

import json
import threading
from pathlib import Path

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool

from config import Config, load_config
from utils import (
    DISCIPLINE_TO_LEVEL1,
    UNKNOWN_LEVEL1,
    build_langfuse,
    build_langfuse_config,
    build_llm,
    classes_at_level_under_root,
    load_equipment_class_lookup,
    with_rate_limit_retry,
)


# =========================================================================
# Module state
# =========================================================================
_cfg: Config | None = None
_llm = None
_langfuse_client = None
_langfuse_handler = None
_agent = None

# Thread-local per-call result slot (so parallel calls don't collide).
_result_tls = threading.local()


def _init(cfg: Config | None) -> None:
    global _cfg, _llm, _langfuse_client, _langfuse_handler, _agent
    if _agent is not None and (cfg is None or cfg is _cfg):
        return
    _cfg = cfg or load_config()
    print(f"[unresolved_classifier] initialising LLM ({_cfg.llm.model}, max_tokens=2000) + Langfuse handler + 1 tool")
    _llm = build_llm(_cfg, max_tokens=2000)
    _langfuse_client, _langfuse_handler = build_langfuse()
    _agent = create_agent(
        model=_llm,
        tools=[submit_class_decision],
        system_prompt=_SYSTEM_PROMPT,
    )
    print("[unresolved_classifier] agent ready")


# =========================================================================
# Tool
# =========================================================================
@tool
def submit_class_decision(
    chosen_class: str,
    confidence: str,
    evidence: str,
) -> str:
    """Submit the FINAL class choice for an unresolved worklist row.

    Arguments:
      chosen_class: the best-matching Equipment Class code from the candidates
                    shown in the user message, OR 'NONE' if nothing is plausible.
      confidence:   'High' | 'Medium' | 'Low'.
      evidence:     one sentence describing what signal convinced you.
    """
    _result_tls.result = {
        "chosen_class": (None if str(chosen_class).strip().upper() == "NONE"
                         else str(chosen_class).strip().upper()),
        "confidence": confidence,
        "evidence": evidence,
    }
    return "Decision recorded."


# =========================================================================
# System prompt
# =========================================================================
_SYSTEM_PROMPT = """You are the Unresolved Equipment Classifier for the SD & TA Agent.

The user message gives you ONE worklist row for which the SAP equipment-master
match FAILED, plus a list of candidate Equipment Class codes at the current
target level under the row's discipline root.

YOUR JOB
Pick the SINGLE Equipment Class whose scope best fits the row's Discipline,
EqptType, Job Description and SubLoc1. Use the class descriptions to reason
semantically.

HOW TO PICK (be decisive)
1. Read the row's EqptType and Job Description carefully. These usually name
   the equipment type directly (e.g. "Transformer", "Pump", "Control Valve",
   "Switchgear", "UPS", "Battery", "Heater", "Compressor", "Piping", "Vessel",
   "Instrument", "Junction Box", "Cable", "Motor", "Fan").
2. Scan ALL candidate class descriptions and find the one whose description
   covers that equipment type.
3. If a candidate's description clearly covers the equipment's domain, PICK IT.
   You do not need a perfect match -- "best reasonable fit" is enough.

PICK -- CONCRETE EXAMPLES (ALL REAL)
- EqptType "Transformer" / Job "Transformer PPM"      -> ETLDRY (HV/LV POWER TRANSFORMER DRY TYPE)
- EqptType "Pump"                                     -> MRP   (PUMP)
- EqptType "Control Valve (PCV/FCV/LCV)"              -> IVCV  (CONTROL VALVES)
- EqptType "Shutdown Valve / SDV / ESDV"              -> IVSD  (SHUTDOWN VALVE)
- EqptType "Switchgear" (HV)                          -> ESH   (HV ELECTRICAL SWITCHGEAR)
- EqptType "Switchgear" (LV)                          -> ESL   (LV ELECTRICAL SWITCHGEAR)
- EqptType "UPS"                                      -> EUA1 / EUA2 (AC UPS)
- EqptType "Battery"                                  -> EUB2  (BATTERY, VENTED NICAD)
- EqptType "Heater" / "Process Heater"                -> EHP   (HEATER, PROCESS)
- EqptType "Compressor"                               -> MRK   (COMPRESSOR)
- EqptType "Piping" / Job "Piping Replacement"        -> MSP   (PIPING)
- EqptType "Pressure Vessel" / "Separator"            -> MSV   (VESSEL, UNFIRED & PRESSURIZED)
- EqptType "Heat Exchanger"                           -> MSE   (UNFIRED HEAT EXCHANGER)
- EqptType "Junction Box"                             -> EJJBX (JUNCTION BOX)
- EqptType "Control Panel" / "DCS Panel"              -> EJCL  (CONTROL PANEL)
- EqptType "Marshalling Cabinet"                      -> EJMC  (MARSHALLING CABINET)

NONE -- WHEN AND ONLY WHEN
Return chosen_class='NONE' only when:
  * The equipment's domain genuinely is not represented in the candidate list
    at this level (e.g. finer Level-4 codes don't yet cover transformers), or
  * The EqptType is too generic ("Electrical Equipment", "Instrument") AND
    nothing in the Job Description or SubLoc1 narrows it, AND no candidate
    description covers it.

DO NOT return NONE just because:
  - You're not 100% sure.
  - The EqptType string isn't a word-for-word match to a candidate description.
  - You think a coarser level might have a "better" match. (It won't -- the
    finest level that reasonably covers the equipment is the right answer.)

If you can see a candidate description that clearly covers the equipment's
domain (e.g. "transformer" word in a Transformer row's class description),
pick it and give confidence=High. If the match is reasonable but not perfect,
pick it with confidence=Medium.

OUTPUT
Call submit_class_decision ONCE with chosen_class, confidence (High/Medium/Low),
evidence (one sentence citing the key signal). After the tool returns, reply
with one short sentence summarising your choice. Do not re-call tools.
"""


# =========================================================================
# Per-row invoke
# =========================================================================
def _classify_one_row(row, class_lookup: dict, fallback_levels: list[int]) -> dict:
    """Classify one unresolved worklist row. Returns a dict with chosen_class,
    classified_level, evidence."""
    discipline = str(row.get("Discipline", "")).strip()
    eqpt_type  = str(row.get("EqptType", "")).strip()
    job_desc   = str(row.get("Job Description", "")).strip()
    subloc1    = str(row.get("SubLoc1", "")).strip()
    worklist_id = str(row.get("WorklistID") or row.get("ID") or "unknown")

    # Step A: Discipline -> Level-1 code.
    root = DISCIPLINE_TO_LEVEL1.get(discipline, UNKNOWN_LEVEL1)

    # If we don't even know the discipline root, the only answer is UNKNOWN.
    if root == UNKNOWN_LEVEL1:
        return {"chosen_class": UNKNOWN_LEVEL1, "classified_level": 1,
                "evidence": f"Discipline '{discipline}' not mapped to a Level-1 root."}

    # Step B: walk fallback levels fine -> coarse.
    print(f"           root='{root}' (from Discipline='{discipline}'); walking levels {fallback_levels}")
    for lvl in fallback_levels:
        candidates = classes_at_level_under_root(class_lookup, root, lvl)
        if not candidates:
            print(f"           Level {lvl}: no candidates under root '{root}', skipping")
            continue
        print(f"           Level {lvl}: {len(candidates)} candidates under '{root}' -> asking LLM")
        result = _ask_llm_pick(row_info={
            "worklist_id": worklist_id, "discipline": discipline,
            "eqpt_type": eqpt_type, "job_description": job_desc, "subloc1": subloc1,
            "discipline_root": root, "target_level": lvl,
        }, candidates=candidates)
        if result and result.get("chosen_class"):
            print(f"           Level {lvl}: picked {result['chosen_class']} "
                  f"(conf={result.get('confidence','?')})")
            return {"chosen_class": result["chosen_class"],
                    "classified_level": lvl,
                    "evidence": result.get("evidence", "")}
        print(f"           Level {lvl}: LLM returned NONE, trying next coarser level")

    # Fallback: Level-1 code itself.
    root_desc = class_lookup.get(root, {}).get("description", "")
    print(f"           Fallback: no match at any finer level, defaulting to Level-1 root '{root}'")
    return {"chosen_class": root, "classified_level": 1,
            "evidence": f"No match at finer levels; defaulted to Level-1 root '{root}' ({root_desc})."}


def _ask_llm_pick(row_info: dict, candidates: list[dict]) -> dict | None:
    """Invoke the agent for one (row, candidates) pair. Returns the submit tool
    result or None on failure."""
    _result_tls.result = None
    payload = {"worklist_row": row_info, "candidates": candidates}
    config = build_langfuse_config(
        _cfg,
        run_name=f"{_cfg.langfuse.agent_name} - Unresolved {row_info['worklist_id']} L{row_info['target_level']}",
        session_id=_cfg.langfuse.session_unresolved,
        callback_handler=_langfuse_handler,
        tags=["unresolved", row_info["worklist_id"], f"level_{row_info['target_level']}"],
        metadata={"worklist_id": row_info["worklist_id"],
                  "target_level": row_info["target_level"],
                  "discipline_root": row_info["discipline_root"]},
    )

    def _invoke():
        _agent.invoke(
            {"messages": [{"role": "user",
                           "content": json.dumps(payload, default=str)}]},
            config=config,
        )

    try:
        with_rate_limit_retry(_invoke, cfg=_cfg)
    except Exception as e:
        print(f"    [unresolved] invoke failed for {row_info['worklist_id']}: {e}")
        return None

    return getattr(_result_tls, "result", None)


# =========================================================================
# Public: classify every unresolved row + write deliverable
# =========================================================================
def classify_unresolved(cfg: Config, cleaned_worklist: pd.DataFrame,
                        equipment_class_df: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    """Classify every cleaned-worklist row with no Suggested EquipmentNo.
    Writes cfg.unresolved_items. Returns (classified_df, output_path)."""
    _init(cfg)
    print(f"[unresolved_classifier] classify_unresolved  "
          f"fallback_levels={cfg.unresolved_classifier.fallback_levels}")

    # Identify unresolved rows: Suggested EquipmentNo empty / NaN / 'nan' / None.
    def _is_unresolved(v):
        if pd.isna(v):
            return True
        s = str(v).strip().lower()
        return s in {"", "nan", "none", "null"}

    print("[unresolved_classifier] Step 1/3 : identifying unresolved rows in cleaned worklist")
    unresolved = cleaned_worklist[
        cleaned_worklist["Suggested EquipmentNo"].apply(_is_unresolved)
    ].copy()
    print(f"[unresolved_classifier]            unresolved: {len(unresolved)} / {len(cleaned_worklist)} rows")
    if unresolved.empty:
        print("[unresolved_classifier]            nothing to classify -- writing empty deliverable")
        empty = pd.DataFrame()
        cfg.unresolved_items.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(cfg.unresolved_items, engine="openpyxl") as xw:
            empty.to_excel(xw, sheet_name="Unresolved", index=False)
        return empty, cfg.unresolved_items

    print("[unresolved_classifier] Step 2/3 : loading equipment class lookup for fallback walk")
    class_lookup = load_equipment_class_lookup(equipment_class_df)
    print(f"[unresolved_classifier]            {len(class_lookup)} class codes in lookup")

    print(f"[unresolved_classifier] Step 3/3 : classifying {len(unresolved)} unresolved rows "
          f"(fallback walk {cfg.unresolved_classifier.fallback_levels})")
    rows_out = []
    for i, (_, r) in enumerate(unresolved.iterrows(), start=1):
        wid = str(r.get("WorklistID") or r.get("ID") or f"row{i}")
        print(f"  [{i}/{len(unresolved)}] {wid}  Discipline={r.get('Discipline','')}  "
              f"EqptType={r.get('EqptType','')}")
        res = _classify_one_row(r, class_lookup, cfg.unresolved_classifier.fallback_levels)
        print(f"           -> class={res['chosen_class']} level={res['classified_level']}")
        rows_out.append({
            "WorklistID":       r.get("WorklistID", ""),
            "EquipmentNo":      r.get("EquipmentNo", ""),
            "SubLoc1":          r.get("SubLoc1", ""),
            "Discipline":       r.get("Discipline", ""),
            "EqptType":         r.get("EqptType", ""),
            "Job Description":  r.get("Job Description", ""),
            "FuncLoc":          r.get("FuncLoc", ""),
            "Classified_Class": res["chosen_class"],
            "Classified_Level": res["classified_level"],
            "Classifier_Evidence": res["evidence"],
            "TaskListGroup":    f"{cfg.plant.code}{res['chosen_class']}" if res["chosen_class"] else "",
            "MANUAL_EquipmentNo":    "",
            "MANUAL_TechnicalIdentNo": "",
            "REVIEWER_NOTES":        "",
        })

    classified = pd.DataFrame(rows_out)
    cfg.unresolved_items.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(cfg.unresolved_items, engine="openpyxl") as xw:
        classified.to_excel(xw, sheet_name="Unresolved", index=False)

    print(f"\nWrote unresolved items -> {cfg.unresolved_items}")
    counts = classified["Classified_Level"].value_counts().sort_index().to_dict()
    print(f"  classifications by level: {counts}")

    _langfuse_client.flush()
    return classified, cfg.unresolved_items
