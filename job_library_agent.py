"""Job Library Agent -- builds a per-year Job Library.

Pipeline (orchestrated per year by main.py):
  1. Join the cleaned worklist to the equipment master -> get CatalogProfile.
  2. Resolve each equipment to an Equipment Class at the preferred level.
  3. For each class, invoke the tool-calling agent to cluster its equipments
     into Group Counters (semantic clustering, 40-char limit, detail-first).
  4. Write 3-sheet workbook: EquipmentClassMap / JobLibrary / CounterEquipments.

Public API:
    run_for_year(cfg, cleaned_worklist, equipment_master, tasklist,
                 equipment_class_df) -> Path
"""

import json
from pathlib import Path

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool

from config import Config, load_config
from utils import (
    build_langfuse,
    build_langfuse_config,
    build_llm,
    load_equipment_class_lookup,
    normalize_cell,
    resolve_class,
    with_rate_limit_retry,
    write_job_library,
)


# =========================================================================
# Module state (shared mutable store the tools write into)
# =========================================================================
_cfg: Config | None = None
_llm = None
_langfuse_client = None
_langfuse_handler = None
_agent = None

# Set by run_for_year before invoking the agent.
JOINED: pd.DataFrame | None = None
# tool -> store;   class_code -> {"group","desc","counters":[...]}
JOB_LIBRARY_STORE: dict[str, dict] = {}


# =========================================================================
# Tools
# =========================================================================
@tool
def get_class_equipments(class_code: str) -> str:
    """Return every equipment in the given equipment class together with its
    ordered task sequence and per-task durations (hours). Output is JSON:
    {equipment_class, class_description, task_list_group,
     equipments:[{id, equipment_no, tasks[], durations_hrs[]}]}."""
    cls = str(class_code).strip().upper()
    assert JOINED is not None, "run_for_year must prime the JOINED frame"
    block = JOINED[JOINED["Class"] == cls]
    if block.empty:
        return json.dumps({"error": f"No equipments found for class {cls}"})

    group = block["TaskListGroup"].iloc[0]
    desc  = block["ClassDescription"].iloc[0]
    equipments = []
    for eid, (eq_no, erows) in enumerate(
        block.sort_values(["Suggested EquipmentNo", "TaskNo"])
             .groupby("Suggested EquipmentNo", sort=True), start=1,
    ):
        equipments.append({
            "id": eid,
            "equipment_no": str(eq_no),
            "tasks": erows["Task"].tolist(),
            "durations_hrs": [float(d) if pd.notna(d) else 0.0
                              for d in erows["Dur (Hrs)"].tolist()],
        })
    return json.dumps({
        "equipment_class": f"{cls} - {desc}",
        "class_description": desc,
        "task_list_group": group,
        "equipments": equipments,
    }, default=str)


@tool
def save_job_counters(class_code: str, counters_json: str) -> str:
    """Persist the final Group Counters for one equipment class.

    counters_json: a JSON string shaped as
    {"counters":[{"header":"...",
                  "operations":[{"description":"...","duration_hrs":<float>}, ...],
                  "equipment_ids":[<int>, ...]}, ...]}

    Every equipment_id returned by get_class_equipments MUST appear in exactly
    one counter. Call this ONCE per class, after clustering."""
    cls = str(class_code).strip().upper()
    try:
        payload = json.loads(counters_json) if isinstance(counters_json, str) else counters_json
    except Exception as e:
        return f"save_job_counters FAILED: invalid JSON ({e})"

    counters = payload.get("counters", [])
    if not isinstance(counters, list) or not counters:
        return "save_job_counters FAILED: counters list is empty"

    assert JOINED is not None
    block = JOINED[JOINED["Class"] == cls]
    group = block["TaskListGroup"].iloc[0] if not block.empty else f"{_cfg.plant.code}{cls}"
    desc  = block["ClassDescription"].iloc[0] if not block.empty else ""
    JOB_LIBRARY_STORE[cls] = {"group": group, "desc": desc, "counters": counters}
    return f"Saved {len(counters)} counters for class {cls} ({group})."


# =========================================================================
# System prompt (unchanged behaviour; wording identical to prior version)
# =========================================================================
_SYSTEM_PROMPT = """You are the Shutdown & Turnaround Job Library creation agent, building a SAP PM Job Library for a Petronas offshore turnaround.

For ONE equipment class you receive, do exactly this:

STEP 1 -- FETCH DATA
Call get_class_equipments(class_code) exactly once. You will get every equipment
in the class with its ordered task sequence and per-step durations (hours).

STEP 2 -- CLUSTER BY SHARED MAINTENANCE INTENT (not shared wording)
Think like a turnaround planner: group equipments whose task sequences describe
the SAME MAINTENANCE PROCEDURE -- same work intent and purpose -- even when the
wording, phrasing, or level of detail is quite different.

Examples of what counts as the SAME procedure:
  - "Megger test on motor", "Insulation resistance (IR) test", "Perform IR
    testing on power cables" -> SAME (insulation testing).
  - "Remove head of pump P-101", "Disassemble pump casing", "Open pump bonnet"
    -> SAME (pump teardown).
  - "Visual inspection", "Check for corrosion", "Walkdown & inspection"
    -> SAME (visual inspection).
  - "Function test of SDV", "Stroke test of shutdown valve", "SDV partial-stroke
    test" -> SAME (SDV functional test).

Equipment numbers, tag numbers, module IDs, and line sizes are IRRELEVANT to
clustering -- do not let them split a cluster.

Only create a separate counter when the WORK ITSELF is genuinely different
(different sub-system, different craft, different end-goal). PREFER FEWER,
BROADER COUNTERS over many narrow ones.

Each counter is a complete maintenance package for that cluster of equipments.
Every equipment_id MUST appear in exactly ONE counter.

STEP 3 -- WRITE OPERATIONS (SAP 40-CHAR HARD LIMIT)
For each counter, produce the generalised operation list:
  - Strip equipment numbers / tags / line IDs from each step.
  - Keep original task sequence order (TA/shutdown order matters).
  - Deduplicate near-identical steps.
  - When several equipments in the same counter describe the SAME step at
    different levels of detail, KEEP THE MOST DETAILED, MOST SPECIFIC VERSION
    as the canonical operation. Do NOT fall back to the shortest/vaguest
    wording. More specific is better -- it carries the real procedure.
    Example: if one equipment says "Test insulation" and another says
    "Insulation resistance test with 5kV megger on cable", use the second
    one (shortened to <= 40 chars via STEP 3's splitting rule if needed).

HARD RULE: every operation "description" must be <= 40 characters INCLUDING
spaces. SAP will reject anything longer. Count characters before you write.

If a step cannot fit in 40 chars, SPLIT it into 2 or more sub-steps that
together describe the same work. Allocate the ORIGINAL step's duration ACROSS
the sub-steps IN PROPORTION TO EFFORT -- NOT equally, NOT averaged. The sum of
sub-step durations must equal the original step's duration.

Example of a correct split:
  Original step  : "Remove head of pump, inspect bearings, replace gasket"
                    duration = 1.5 h
  Split into:
    "Remove pump head"                 ~ 0.3 h  (quick mechanical work)
    "Inspect bearings and shaft"       ~ 0.5 h  (detailed inspection)
    "Replace gasket and reassemble"    ~ 0.7 h  (longer rebuild)
  Sum = 1.5 h, matching the original.

Across equipments in the same counter, pick a REALISTIC duration per step
(median or craft-reasonable value). Do NOT blindly average -- averaging
smooths out reality and gives wrong manhours.

STEP 4 -- SAVE RESULT
Call save_job_counters(class_code, counters_json) exactly ONCE with the final
clusters, then reply with one short sentence summarising the clusters.
Do not re-call tools.

COUNTER HEADER: short canonical procedure name, title case, <= 40 chars incl. spaces (SAP KTEXT limit). Minimise wording to fit; a header is a single label and cannot be split.

SCHEMA passed to save_job_counters:
{
  "counters": [
    {
      "header": "...",
      "operations": [
        {"description": "<=40 chars incl. spaces", "duration_hrs": <float>},
        ...
      ],
      "equipment_ids": [<int>, ...]
    },
    ...
  ]
}

FULL EXAMPLE (real Bokor 2024 data -- Transformer class, Transformer PPM 2Y)
Input class = ETLDRY (HV/LV POWER TRANSFORMER (DISTR.) DRY TYPE) with 3 equipments:
  eq 1 (11054251) tasks:
      ["Perform insulation resistance test (cable and transformer)",
       "Perform winding resistance test (cable and transformer)",
       "Perform polarization index test (cable)",
       "Improvement of genral contion of transformer"]
      durations: [1.0, 1.0, 1.0, 1.0]
  eq 2 (11054252) tasks: (same as eq 1)
  eq 3 (11054256) tasks: (same as eq 1)

All three equipments share the same transformer PPM procedure -> ONE counter.
Original task 1 "Perform insulation resistance test (cable and transformer)" is
58 chars (> 40) -- SPLIT into two sub-steps (cable vs transformer winding test),
allocating the 1.0 h by effort (transformer side takes longer). Same split for
task 2. Tasks 3 and 4 are rewritten to fit <= 40 chars without splitting.

Correct output:
{
  "counters": [
    {
      "header": "Transformer PPM 2Y Elec Test",
      "operations": [
        {"description": "Insulation resistance test - cable",   "duration_hrs": 0.4},
        {"description": "Insulation resistance test - txfmr",   "duration_hrs": 0.6},
        {"description": "Winding resistance test - cable",      "duration_hrs": 0.4},
        {"description": "Winding resistance test - txfmr",      "duration_hrs": 0.6},
        {"description": "Polarization index test on cable",     "duration_hrs": 1.0},
        {"description": "Improve transformer general condition","duration_hrs": 1.0}
      ],
      "equipment_ids": [1, 2, 3]
    }
  ]
}
"""


# =========================================================================
# Agent setup (lazy)
# =========================================================================
def _init(cfg: Config | None) -> None:
    global _cfg, _llm, _langfuse_client, _langfuse_handler, _agent
    if _agent is not None and (cfg is None or cfg is _cfg):
        return
    _cfg = cfg or load_config()
    print(f"[job_library_agent] initialising LLM ({_cfg.llm.model}) + Langfuse handler + 2 tools")
    _llm = build_llm(_cfg)
    _langfuse_client, _langfuse_handler = build_langfuse()
    _agent = create_agent(
        model=_llm,
        tools=[get_class_equipments, save_job_counters],
        system_prompt=_SYSTEM_PROMPT,
    )
    print("[job_library_agent] agent ready")


# =========================================================================
# Per-class invoke
# =========================================================================
def _run_agent_for_class(cls_code: str, cls_desc: str) -> None:
    question = (
        f"Build Job Library Group Counters for equipment class {cls_code} "
        f"({cls_desc}). Follow the system procedure exactly."
    )
    config = build_langfuse_config(
        _cfg,
        run_name=f"{_cfg.langfuse.agent_name} - {cls_code}",
        session_id=_cfg.langfuse.session_job_library,
        callback_handler=_langfuse_handler,
        tags=[str(_cfg.year), cls_code],
        metadata={"equipment_class": cls_code, "class_description": cls_desc,
                  "year": _cfg.year},
    )

    def _invoke():
        _agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )

    with_rate_limit_retry(_invoke, cfg=_cfg)


# =========================================================================
# Public: build the per-year library
# =========================================================================
def run_for_year(
    cfg: Config,
    cleaned_worklist: pd.DataFrame,
    equipment_master: pd.DataFrame,
    tasklist: pd.DataFrame,
    equipment_class_df: pd.DataFrame,
) -> Path:
    """Build the per-year Job Library and write it to cfg.job_library_new.
    Returns the output path."""
    _init(cfg)
    global JOINED, JOB_LIBRARY_STORE
    JOB_LIBRARY_STORE = {}
    print(f"[job_library_agent] run_for_year  year={cfg.year}  class_level={cfg.job_library_agent.class_level}")

    # ----- Step 1: Join cleaned worklist to master to get CatalogProfile
    print("[job_library_agent] Step 1/4 : joining cleaned worklist with equipment master "
          "(for CatalogProfile + TechnicalIdentNo + FunctionalLocation)")
    em = equipment_master[[
        "EquipmentNo", "ObjectDescription", "CatalogProfile",
        "TechnicalIdentNo", "FunctionalLocation",
    ]].copy()
    em["EquipmentNo"] = em["EquipmentNo"].astype(str).str.strip()

    wl = cleaned_worklist.copy()
    wl["Suggested EquipmentNo"] = wl["Suggested EquipmentNo"].astype(str).str.strip()
    wl = wl.merge(em, left_on="Suggested EquipmentNo", right_on="EquipmentNo",
                  how="left")
    print(f"[job_library_agent]           merged -> {len(wl)} worklist rows")

    # ----- Step 2: Resolve class
    print(f"[job_library_agent] Step 2/4 : resolving Equipment Class "
          f"(prefer Level {cfg.job_library_agent.class_level}, walk up if needed)")
    lookup = load_equipment_class_lookup(equipment_class_df)
    cls_rows = wl["CatalogProfile"].apply(
        lambda c: resolve_class(c, lookup, cfg.job_library_agent.class_level))
    wl["Class"] = [r[0] for r in cls_rows]
    wl["ClassLevel"] = [r[1] for r in cls_rows]
    wl["ClassDescription"] = [r[2] for r in cls_rows]
    wl["TaskListGroup"] = wl["Class"].apply(
        lambda c: f"{cfg.plant.code}{c}" if c else None)
    resolved_cnt = wl["Class"].notna().sum()
    print(f"[job_library_agent]           {resolved_cnt}/{len(wl)} rows resolved to a class")

    # ----- Step 3: Join with Tasklist on WorklistID
    print("[job_library_agent] Step 3/4 : joining with Tasklist on WorklistID")
    tl = tasklist[["WorklistID", "EquipmentNo", "TaskNo", "Task", "Dur (Hrs)"]].copy()
    tl["Task"] = tl["Task"].astype(str).str.strip()
    JOINED = tl.merge(
        wl[["WorklistID", "Suggested EquipmentNo", "TaskListGroup",
            "Class", "ClassDescription"]],
        on="WorklistID", how="inner",
    ).dropna(subset=["TaskListGroup", "Task"])
    print(f"[job_library_agent]           joined -> {len(JOINED)} task rows")

    # ----- Step 4: Run agent per class
    classes = (JOINED[["Class", "ClassDescription"]]
               .dropna(subset=["Class"]).drop_duplicates().sort_values("Class"))
    print(f"\n[job_library_agent] Step 4/4 : running agent for {len(classes)} equipment classes\n")
    for idx, (_, row) in enumerate(classes.iterrows(), start=1):
        cls, desc = row["Class"], row["ClassDescription"]
        n_eq = JOINED[JOINED["Class"] == cls]["Suggested EquipmentNo"].nunique()
        print(f"  [{idx}/{len(classes)}] [{cfg.plant.code}{cls}] {desc} ({n_eq} equipments)")
        try:
            _run_agent_for_class(cls, desc)
        except Exception as e:
            print(f"    FAILED: {e}")

    # ----- Assemble counter dicts for shared writer (resolve equipment_ids to equipment_nos)
    counters_by_group: dict[str, list] = {}
    for cls, stored in sorted(JOB_LIBRARY_STORE.items()):
        group = stored["group"]
        block = JOINED[JOINED["Class"] == cls]
        eq_lookup = {
            eid: str(eq_no) for eid, (eq_no, _) in enumerate(
                block.sort_values(["Suggested EquipmentNo", "TaskNo"])
                     .groupby("Suggested EquipmentNo", sort=True), start=1,
            )
        }
        for cnt_idx, cnt in enumerate(stored["counters"], start=1):
            ctr = f"{cnt_idx:02d}"
            header = str(cnt.get("header", "")).strip() or f"{stored['desc']} #{ctr}"
            ops = [{
                "item_no": f"{i * 10:04d}",
                "description": str(o.get("description", "")).strip(),
                "duration_hrs": round(float(o.get("duration_hrs") or 0), 2),
            } for i, o in enumerate(cnt.get("operations", []) or [], start=1)]

            # Resolve equipment_ids to equipment_nos; enrich from master.
            eq_rows = []
            for eid in cnt.get("equipment_ids", []):
                eq_no = eq_lookup.get(eid, f"<unknown id {eid}>")
                master_row = em[em["EquipmentNo"] == eq_no]
                if not master_row.empty:
                    r = master_row.iloc[0]
                    eq_rows.append({
                        "equipment_no": eq_no,
                        "description": normalize_cell(r.get("ObjectDescription")),
                        "technical_ident_no": normalize_cell(r.get("TechnicalIdentNo")),
                        "functional_location": normalize_cell(r.get("FunctionalLocation")),
                    })
                else:
                    eq_rows.append({
                        "equipment_no": eq_no, "description": "",
                        "technical_ident_no": "", "functional_location": "",
                    })

            counters_by_group.setdefault(group, []).append({
                "group": group, "counter": ctr, "header": header,
                "operations": ops,
                "total_hrs": round(sum(o["duration_hrs"] for o in ops), 2),
                "equipments": eq_rows, "discipline": "",
            })

    equipment_map = wl.dropna(subset=["TaskListGroup"])[[
        "WorklistID", "Suggested EquipmentNo", "ObjectDescription",
        "CatalogProfile", "Class", "ClassDescription", "TaskListGroup",
        "Discipline", "Job Description",
    ]].rename(columns={"Suggested EquipmentNo": "EquipmentNo"})

    write_job_library(cfg.job_library_new, counters_by_group, equipment_map,
                      plant=cfg.plant.werks)
    print(f"\nWrote {cfg.job_library_new}")
    print(f"  Classes saved by agent: {len(JOB_LIBRARY_STORE)} / {len(classes)}")

    # ---- 40-char audit
    all_ops = [(g, c["counter"], o)
               for g, cs in counters_by_group.items()
               for c in cs for o in c["operations"]]
    over = [(g, ctr, o) for g, ctr, o in all_ops if len(o["description"]) > 40]
    print(f"  40-char SAP limit audit: {len(all_ops)} ops, {len(over)} over limit")
    for g, ctr, o in over[:10]:
        print(f"    [{g} #{ctr}] ({len(o['description'])} chars) {o['description']}")
    if len(over) > 10:
        print(f"    ...and {len(over) - 10} more")

    _langfuse_client.flush()
    return cfg.job_library_new
