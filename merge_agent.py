"""Merge a new-year Job Library into the consolidated Job Library.

Tool-calling LangChain agent. Iterates ONE agent invocation per new-year
counter (not per equipment):
  - fetches existing counters in the same TaskListGroup
  - decides match-or-create against them (semantic clustering)
  - on match: assigns new equipments + merges operations with detail-first rule
  - on new : creates a new counter in the consolidated library

First-year base case: if `cfg.job_library_consolidated` doesn't exist yet,
the new-year library is copied in as the starting point (no agent calls).

Public API:
    run_merge(cfg) -> Path     # path to the merged consolidated library
"""

import copy as _copy_mod
import json
import shutil
from pathlib import Path

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool

from config import Config, load_config
from utils import (
    build_langfuse,
    build_langfuse_config,
    build_llm,
    load_job_library,
    with_rate_limit_retry,
    write_job_library,
)


# =========================================================================
# Module state (shared with tool handlers)
# =========================================================================
_cfg: Config | None = None
_llm = None
_langfuse_client = None
_langfuse_handler = None
_agent = None

CONSOLIDATED: dict = {"counters": {}, "equipment_map": pd.DataFrame()}

MERGE_STATS = {
    "new_counters_created":  0,
    "existing_matched":      0,
    "existing_improved":     0,
    "existing_unchanged":    0,
    "operations_upgraded":   0,
    "operations_split":      0,
    "operations_inserted":   0,
    "new_equipments_added":  0,
}


def _reset_stats() -> None:
    for k in MERGE_STATS:
        MERGE_STATS[k] = 0


def _next_counter_number(group: str) -> str:
    existing = CONSOLIDATED["counters"].get(group, [])
    n = max((int(c["counter"]) for c in existing if c["counter"].isdigit()), default=0) + 1
    return f"{n:02d}"


# =========================================================================
# Tools
# =========================================================================
@tool
def get_counters_in_group(tasklist_group: str) -> str:
    """Return every existing Group Counter in the consolidated library for the
    given TaskListGroup (e.g. 'SKET'). JSON:
    {task_list_group, counter_count, counters:[{counter, header, operations[],
    total_hrs, equipments[]}, ...]}. counter_count=0 if nothing exists yet."""
    g = str(tasklist_group).strip().upper()
    counters = CONSOLIDATED["counters"].get(g, [])
    return json.dumps({
        "task_list_group": g, "counter_count": len(counters), "counters": counters,
    }, default=str)


@tool
def assign_to_counter(
    tasklist_group: str,
    group_counter: str,
    new_equipments_json: str,
    operation_merge_decisions_json: str,
) -> str:
    """Assign a NEW year's counter to an EXISTING consolidated counter and merge
    its operations in. Call this when get_counters_in_group showed a counter
    whose procedure matches the new counter.

    Arguments:
      tasklist_group: e.g. 'SKET'.
      group_counter:  the existing counter number, e.g. '01'.
      new_equipments_json: JSON list of new-year equipments to append.
      operation_merge_decisions_json: JSON list of per-op decisions:
        {"action":"keep_existing",     "existing_item_no":"0010"}
        {"action":"upgrade_wording",   "existing_item_no":"0010",
                                       "new_description":"<=40 chars"}
        {"action":"replace_with_split","existing_item_no":"0010",
                                       "new_ops":[{"description":"...","duration_hrs":0.4}, ...]}
        {"action":"insert_new",        "after_item_no":"0030",
                                       "new_op":{"description":"...","duration_hrs":1.0}}
    """
    g = str(tasklist_group).strip().upper()
    c = str(group_counter).strip()
    try:
        new_eqs = json.loads(new_equipments_json) if isinstance(new_equipments_json, str) else new_equipments_json
        decisions = json.loads(operation_merge_decisions_json) if isinstance(operation_merge_decisions_json, str) else operation_merge_decisions_json
    except Exception as e:
        return f"assign_to_counter FAILED: invalid JSON ({e})"

    counters = CONSOLIDATED["counters"].get(g, [])
    target = next((x for x in counters if x["counter"] == c), None)
    if target is None:
        return f"assign_to_counter FAILED: counter {g}-{c} not found"

    ops = target["operations"]
    by_item = {op["item_no"]: i for i, op in enumerate(ops)}
    counter_changed = False

    for d in decisions or []:
        action = str(d.get("action", "")).strip().lower()
        if action == "keep_existing":
            continue
        if action == "upgrade_wording":
            iid = str(d.get("existing_item_no", "")).strip()
            if iid in by_item:
                ops[by_item[iid]]["description"] = str(d.get("new_description", ""))[:40]
                MERGE_STATS["operations_upgraded"] += 1
                counter_changed = True
            continue
        if action == "replace_with_split":
            iid = str(d.get("existing_item_no", "")).strip()
            if iid in by_item:
                idx = by_item[iid]
                new_split = d.get("new_ops", []) or []
                ops[idx:idx + 1] = [{
                    "item_no": "",
                    "description": str(o.get("description", ""))[:40],
                    "duration_hrs": round(float(o.get("duration_hrs") or 0), 2),
                } for o in new_split]
                by_item = {op["item_no"]: i for i, op in enumerate(ops)}
                MERGE_STATS["operations_split"] += 1
                counter_changed = True
            continue
        if action == "insert_new":
            after = str(d.get("after_item_no", "")).strip()
            new_op = d.get("new_op") or {}
            op_obj = {"item_no": "",
                      "description": str(new_op.get("description", ""))[:40],
                      "duration_hrs": round(float(new_op.get("duration_hrs") or 0), 2)}
            if after in by_item:
                ops.insert(by_item[after] + 1, op_obj)
            else:
                ops.append(op_obj)
            by_item = {op["item_no"]: i for i, op in enumerate(ops)}
            MERGE_STATS["operations_inserted"] += 1
            counter_changed = True
            continue

    for i, op in enumerate(ops, start=1):
        op["item_no"] = f"{i * 10:04d}"
    target["operations"] = ops
    target["total_hrs"] = round(sum(float(o["duration_hrs"]) for o in ops), 2)

    existing_eqs = {e["equipment_no"] for e in target["equipments"]}
    added = 0
    for e in new_eqs or []:
        if e.get("equipment_no") and e["equipment_no"] not in existing_eqs:
            target["equipments"].append(e)
            existing_eqs.add(e["equipment_no"])
            added += 1

    MERGE_STATS["existing_matched"] += 1
    MERGE_STATS["existing_improved" if counter_changed else "existing_unchanged"] += 1
    MERGE_STATS["new_equipments_added"] += added
    return (f"Updated counter {g}-{c}: {len(ops)} ops, "
            f"{added} new equipments appended (total {len(target['equipments'])}).")


@tool
def create_new_counter(
    tasklist_group: str,
    header: str,
    operations_json: str,
    new_equipments_json: str,
) -> str:
    """Create a brand-new Group Counter when NO existing counter matches.

    Arguments:
      tasklist_group: e.g. 'SKMR'.
      header: counter header, <= 40 chars incl. spaces (SAP KTEXT limit).
      operations_json: JSON list [{"description":"<=40 chars","duration_hrs":<float>}, ...]
      new_equipments_json: JSON list of equipments.
    """
    g = str(tasklist_group).strip().upper()
    try:
        ops = json.loads(operations_json) if isinstance(operations_json, str) else operations_json
        new_eqs = json.loads(new_equipments_json) if isinstance(new_equipments_json, str) else new_equipments_json
    except Exception as e:
        return f"create_new_counter FAILED: invalid JSON ({e})"

    c = _next_counter_number(g)
    counter_obj = {
        "group": g, "counter": c, "header": str(header)[:40].strip(),
        "operations": [{
            "item_no": f"{i * 10:04d}",
            "description": str(o.get("description", ""))[:40],
            "duration_hrs": round(float(o.get("duration_hrs") or 0), 2),
        } for i, o in enumerate(ops, start=1)],
        "equipments": [e for e in (new_eqs or []) if e.get("equipment_no")],
        "discipline": "",
    }
    counter_obj["total_hrs"] = round(sum(float(o["duration_hrs"]) for o in counter_obj["operations"]), 2)
    CONSOLIDATED["counters"].setdefault(g, []).append(counter_obj)

    MERGE_STATS["new_counters_created"] += 1
    MERGE_STATS["new_equipments_added"] += len(counter_obj["equipments"])
    return f"Created new counter {g}-{c} with {len(counter_obj['operations'])} ops."


# =========================================================================
# System prompt (unchanged behaviour; same few-shot as before)
# =========================================================================
_SYSTEM_PROMPT = """You are SD & TA Agent, merging a new-year Job Library into an existing
consolidated SAP PM Job Library for a Petronas offshore turnaround and shutdown.

You are called ONCE per new-year counter. For each call, do exactly this:

STEP 1 -- FETCH EXISTING COUNTERS
Call get_counters_in_group(tasklist_group) with the new counter's TaskListGroup.
You get back every existing consolidated counter in that group with its full
operation list and equipment list.

STEP 2 -- DECIDE: MATCH or NEW
Compare the new counter's operations (shown in the user message) with every
existing counter in that group. Two counters MATCH when they describe the SAME
MAINTENANCE PROCEDURE -- same intent and purpose, even if wording differs.

Examples of what counts as the SAME procedure:
  - "Insulation Resistance Test" (consolidated) matches "IR Test" (new)
  - "Pump Head Service" matches "Dismantle Pump Head"
  - "Transformer PPM Testing" matches "4Y Transformer PPM"

Equipment numbers / tag numbers / module IDs are IRRELEVANT. If no existing
counter's procedure matches, this is a brand-new procedure -> go to Step 4.

STEP 3 -- MATCH: build the operation-merge decisions
If an existing counter matches, call assign_to_counter(...) ONCE with:
  - tasklist_group, group_counter   (identifies the existing counter)
  - new_equipments_json             (new-year equipments to add to it)
  - operation_merge_decisions_json  (per-op decisions, listed in order)

For every one of the EXISTING counter's operations, you pick an action:
  - "keep_existing"       : the existing op is fine; new year has nothing better.
  - "upgrade_wording"     : the new year's wording for this step is MORE DETAILED
                            or MORE SPECIFIC than the existing one. Replace it (<=40 chars).
  - "replace_with_split"  : the new year splits one existing op into a finer
                            sequence. Replace with the finer split. Durations
                            must sum to the original op's duration.
  - "insert_new"          : the new counter has a step not in the existing counter.
                            Insert at the right position using after_item_no.

RULES:
  - DETAIL-FIRST: keep the more detailed version. If the existing library already
    has a finer split and the new year only has a coarser version, KEEP the
    existing split. Do NOT downgrade.
  - PROCEDURE ORDER: preserve TA/shutdown sequence when inserting new ops.
  - EQUIPMENT UNIQUENESS: assign_to_counter dedupes by equipment_no automatically.
  - 40-CHAR LIMIT: every description you emit must be <= 40 chars incl. spaces.
  - REALISTIC DURATION: when splitting, allocate time by effort, NOT equally.

STEP 4 -- NEW: no match, create a new counter
If no existing counter matches, call create_new_counter(tasklist_group, header,
operations_json, new_equipments_json). Keep header <= 40 chars; preserve order.

AFTER THE TOOL CALL: reply with ONE short sentence summarising what you did.
Do not re-call tools.

=============== FEW-SHOT (real Bokor 2024 + 2026 data) ===============

EXISTING CONSOLIDATED COUNTER in group SKET (from 2024):
  counter 01, header "Transformer PPM 2Y Elec Test", operations (6, total 4.0 h):
    0010 "Insulation resistance test - cable"  0.4h
    0020 "Insulation resistance test - txfmr"  0.6h
    0030 "Winding resistance test - cable"     0.4h
    0040 "Winding resistance test - txfmr"     0.6h
    0050 "Polarization index test on cable"    1.0h
    0060 "Improve transformer general cond."   1.0h

INCOMING NEW-YEAR (2026) COUNTER for SKET:
  header "4Y Transformer PPM OCEANCARE", operations (7):
    "Isolate the ACB/VCB"                                   1.0h
    "Perform inspection, cleaning, IR and PI test"          2.5h
    "Perform turns ratio test"                              2.0h
    "Transformer winding (micro-ohm) resistance meas."      2.0h
    "NER resistance measurement (if applicable)"            1.0h
    "Normalize transformer"                                 1.0h
    "Normalise the ACB/VCB"                                 1.0h

REASONING:
  - Same procedure intent (transformer PPM electrical testing). MATCH counter SKET-01.
  - "Isolate the ACB/VCB" -> INSERT at the top (new step before existing 0010).
  - "Perform inspection, cleaning, IR and PI test" (2.5h) is COARSER than the
    existing split of IR cable/IR txfmr/PI test -> KEEP existing split, don't downgrade.
  - "Perform turns ratio test" -> NEW step, insert after Winding test rows.
  - "Transformer winding resistance" is same intent as existing 0030/0040 ->
    KEEP existing winding split (detail-first).
  - "NER resistance measurement" -> NEW step, insert after turns ratio.
  - "Normalize transformer" + "Normalise ACB/VCB" -> NEW closing steps, insert at end.
"""


# =========================================================================
# Agent setup (lazy)
# =========================================================================
def _init(cfg: Config | None) -> None:
    global _cfg, _llm, _langfuse_client, _langfuse_handler, _agent
    if _agent is not None and (cfg is None or cfg is _cfg):
        return
    _cfg = cfg or load_config()
    print(f"[merge_agent] initialising LLM ({_cfg.llm.model}) + Langfuse handler + 3 tools")
    _llm = build_llm(_cfg)
    _langfuse_client, _langfuse_handler = build_langfuse()
    _agent = create_agent(
        model=_llm,
        tools=[get_counters_in_group, assign_to_counter, create_new_counter],
        system_prompt=_SYSTEM_PROMPT,
    )
    print("[merge_agent] agent ready")


# =========================================================================
# Per-counter invoke
# =========================================================================
def _run_merge_for_counter(nc: dict) -> None:
    g, c = nc["group"], nc["counter"]
    user_msg = {
        "task_list_group": g,
        "new_year_counter": {
            "header": nc["header"],
            "operations": nc["operations"],
            "total_hrs": nc.get("total_hrs", 0),
            "equipments": nc["equipments"],
        },
    }
    config = build_langfuse_config(
        _cfg,
        run_name=f"{_cfg.langfuse.agent_name} - Merge {g}-{c}",
        session_id=_cfg.langfuse.session_merge,
        callback_handler=_langfuse_handler,
        tags=["merge", g, c],
        metadata={"tasklist_group": g, "new_counter": c},
    )

    def _invoke():
        _agent.invoke(
            {"messages": [{"role": "user",
                           "content": json.dumps(user_msg, default=str)}]},
            config=config,
        )

    with_rate_limit_retry(_invoke, cfg=_cfg)


# =========================================================================
# Public: run the merge
# =========================================================================
def run_merge(cfg: Config) -> Path:
    """Fold cfg.job_library_new into cfg.job_library_consolidated. Returns
    the consolidated output path."""
    existing = cfg.job_library_consolidated
    new      = cfg.job_library_new
    out      = cfg.job_library_consolidated

    print("[merge_agent] run_merge")
    print(f"[merge_agent]   existing consolidated library: {existing}")
    print(f"[merge_agent]   new-year library             : {new}")
    print(f"[merge_agent]   output                       : {out}\n")

    if not existing.exists():
        print(f"[merge_agent] no existing consolidated library at {existing.name}")
        print(f"[merge_agent] FIRST-YEAR BASE CASE -- copying {new.name} -> {out.name} (no agent calls)")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(new, out)
        print("[merge_agent] done")
        return out

    _init(cfg)
    _reset_stats()
    CONSOLIDATED["counters"].clear()

    print(f"[merge_agent] Step 1/3 : loading existing library ({existing.name})")
    existing_data = load_job_library(existing)
    print(f"[merge_agent]            {sum(len(cs) for cs in existing_data['counters'].values())} counters "
          f"across {len(existing_data['counters'])} TaskListGroups")

    print(f"[merge_agent] Step 2/3 : loading new-year library ({new.name})")
    new_data = load_job_library(new)
    print(f"[merge_agent]            {sum(len(cs) for cs in new_data['counters'].values())} counters "
          f"across {len(new_data['counters'])} TaskListGroups")

    CONSOLIDATED["counters"] = {g: _copy_mod.deepcopy(c)
                                for g, c in existing_data["counters"].items()}
    em_df = existing_data["equipment_map"]
    new_em = new_data["equipment_map"]
    if "EquipmentNo" in new_em.columns and "EquipmentNo" in em_df.columns:
        CONSOLIDATED["equipment_map"] = pd.concat(
            [em_df, new_em], ignore_index=True
        ).drop_duplicates(subset=["EquipmentNo"])
    else:
        CONSOLIDATED["equipment_map"] = em_df

    new_counters = [c for cs in new_data["counters"].values() for c in cs]
    print(f"\n[merge_agent] Step 3/3 : running agent for {len(new_counters)} new-year counters "
          f"(one invocation per counter)\n")
    for idx, nc in enumerate(new_counters, start=1):
        print(f"  [{idx}/{len(new_counters)}] {nc['group']}-{nc['counter']}  \"{nc['header']}\"  "
              f"({len(nc['operations'])} ops, {len(nc['equipments'])} eqs)")
        try:
            _run_merge_for_counter(nc)
        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\n[merge_agent] writing consolidated library -> {out.name}")
    write_job_library(out, CONSOLIDATED["counters"], CONSOLIDATED["equipment_map"],
                      plant=cfg.plant.werks)

    print(f"\nWrote consolidated library -> {out}")
    group_cnt = len(CONSOLIDATED["counters"])
    counter_cnt = sum(len(cs) for cs in CONSOLIDATED["counters"].values())
    print(f"  {group_cnt} TaskListGroups, {counter_cnt} total counters")

    s = MERGE_STATS
    processed = s["new_counters_created"] + s["existing_matched"]
    print("\n============== MERGE SUMMARY ==============")
    print(f"  New-year counters processed    : {processed}")
    print(f"  New counters created           : {s['new_counters_created']}")
    print(f"  Existing counters matched      : {s['existing_matched']}")
    print(f"    -> improved (op-level edits) : {s['existing_improved']}")
    print(f"    -> unchanged (kept as-is)    : {s['existing_unchanged']}")
    print(f"  Operation-level changes:")
    print(f"    wording upgrades             : {s['operations_upgraded']}")
    print(f"    replaced-with-finer-splits   : {s['operations_split']}")
    print(f"    new ops inserted             : {s['operations_inserted']}")
    print(f"  New equipments added           : {s['new_equipments_added']}")
    print("===========================================")

    _langfuse_client.flush()
    return out
