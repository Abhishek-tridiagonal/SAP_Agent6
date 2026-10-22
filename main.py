"""Entry point for the SD & TA Agent (Shutdown & Turnaround / Job Library).

Runs the full agent pipeline for one year of worklist data, driven entirely
by config.ini. No CLI arguments.

Pipeline stages (each gated by the matching `[flags].run_*` setting):
  Stage 1 - Cleaning      : map messy worklist equipment -> SAP master
  Stage 2 - Unresolved    : classify rows the cleaning agent couldn't match
                            to a coarser Equipment Class
  Stage 3 - Job Library   : per-year clustering of equipments into Group Counters
  Stage 4 - Merge         : fold the per-year library into the consolidated one
  Stage 5 - SAP Export    : convert consolidated library to IA05 upload workbook

Run with:
    python main.py

The frontend API will call `run_pipeline(cfg)` directly after loading / overriding
the config in memory.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from cleaning_agent import cleaning_agent
from config import Config, load_config
from kg_extractors import (
    fetch_equipment_class,
    fetch_equipment_master,
    load_worklist_xlsx,
)
from job_library_agent import run_for_year as build_job_library_for_year
from merge_agent import run_merge
from sap_exporter import export as export_sap
from unresolved_classifier import classify_unresolved
from utils import PREFIX_TO_SUBLOC1, SUBLOC1_TO_PREFIX


# =========================================================================
# Stage 1 helpers (deterministic matching + LLM path)
# =========================================================================
def _normalize(s):
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def _tid_prefix_and_tail(tid):
    s = str(tid or "")
    if "-" not in s:
        return None, ""
    pfx, _, tail = s.partition("-")
    return pfx.upper(), _normalize(tail)


def _subloc1_implies_prefix(subloc1):
    s = str(subloc1 or "").strip().upper()
    for alias, pfx in SUBLOC1_TO_PREFIX.items():
        if s.startswith(alias.upper()):
            return pfx
    return None


def _extract_tokens(equipmentno):
    """Pull candidate tag tokens for the deterministic step."""
    raw = re.sub(r"\([^)]*\)", " ", str(equipmentno or ""))
    sub_tags = re.split(r"[/&;,]", raw)
    tokens = [_normalize(t) for t in sub_tags if t.strip()]
    return sorted({t for t in tokens if len(t) >= 3})


def _resolve_deterministic(row, master):
    tokens = _extract_tokens(row.get("EquipmentNo"))
    if not tokens:
        return None
    subloc_pfx = _subloc1_implies_prefix(row.get("SubLoc1"))

    if subloc_pfx:
        targets = {subloc_pfx + t for t in tokens}
        hits = master[master["_TIDNorm"].isin(targets)]
        if len(hits) == 1:
            return _build_det_result(row, hits.iloc[0], "deterministic_prefix_match")
        if len(hits) > 1:
            return _build_det_result(row, hits.iloc[0], "deterministic_prefix_match_first")

    tail_hits = master[master["_TIDTail"].isin(tokens)]
    if len(tail_hits) == 1:
        return _build_det_result(row, tail_hits.iloc[0], "deterministic_tail_only")
    return None


def _build_det_result(row, master_row, method):
    orig_subloc1 = str(row.get("SubLoc1") or "").strip()
    master_pfx = master_row["_TIDPrefix"]
    subloc_pfx = _subloc1_implies_prefix(orig_subloc1)
    result = {
        "Suggested TechnicalId":   master_row["TechnicalIdentNo"],
        "Suggested EquipmentNo":   master_row["EquipmentNo"],
        "EquipmentNo Confidence":  "High",
        "EquipmentNo Evidence":    f"Tail '{master_row['_TIDTail']}' matched TID '{master_row['TechnicalIdentNo']}' ({method}).",
        "MatchMethod":             method,
        "NeedsReview":             False,
    }
    if subloc_pfx and master_pfx and master_pfx != subloc_pfx:
        suggested = PREFIX_TO_SUBLOC1.get(master_pfx, orig_subloc1)
        result.update({
            "Suggested SubLoc1": suggested,
            "SubLoc1 Confidence": "Medium",
            "SubLoc1 Evidence": (f"Original SubLoc1 '{orig_subloc1}' implies '{subloc_pfx}', "
                                 f"but master has this tag under '{master_pfx}'. "
                                 f"Recommend '{suggested}' - verify."),
            "NeedsReview": True,
        })
    else:
        result.update({
            "Suggested SubLoc1": orig_subloc1,
            "SubLoc1 Confidence": "High",
            "SubLoc1 Evidence": "Consistent with master prefix.",
        })
    return result


def _run_llm_path(row, master, cfg: Config):
    subloc_pfx = _subloc1_implies_prefix(row.get("SubLoc1"))
    pool = master[master["_TIDPrefix"] == subloc_pfx] if subloc_pfx else master
    if subloc_pfx:
        print(f"  Pool: {len(pool)} rows under prefix '{subloc_pfx}'")
    else:
        print(f"  SubLoc1 unmappable - using full master: {len(pool)} rows")

    if pool.empty:
        return _no_match_result(row, f"No candidates under SubLoc1 prefix '{subloc_pfx}'.")

    # Guardrail: huge pools make the LLM path unproductive and burn rate-limit
    # quota. Hand the row to the Unresolved Classifier instead.
    pool_cap = cfg.cleaning_agent.max_pool_size_for_llm
    if len(pool) > pool_cap:
        print(f"  Pool exceeds max_pool_size_for_llm ({len(pool)} > {pool_cap}); "
              f"skipping LLM path -> routed to Unresolved Classifier.")
        return _no_match_result(
            row,
            f"Pool size {len(pool)} exceeded LLM cutoff {pool_cap}; "
            f"routed to Unresolved Classifier (Discipline+EqptType fallback).",
        )

    batch_size = cfg.cleaning_agent.batch_size
    if len(pool) <= batch_size:
        r = cleaning_agent(row, pool, cfg)
        return _finalize_llm_result(row, r, pool, method="llm")

    batches = [pool.iloc[s:s + batch_size] for s in range(0, len(pool), batch_size)]
    print(f"  Running {len(batches)} CleaningAgent batches in parallel "
          f"(max {cfg.cleaning_agent.max_concurrent_llm})")
    picks = []
    with ThreadPoolExecutor(max_workers=cfg.cleaning_agent.max_concurrent_llm) as ex:
        future_to_idx = {ex.submit(cleaning_agent, row, b, cfg): i for i, b in enumerate(batches)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            r = fut.result()
            picked = r.get("suggested_equipment_no")
            print(f"    batch {i + 1}/{len(batches)} done -> {picked or 'no pick'}")
            if picked:
                picks.append(r)

    if not picks:
        return _no_match_result(row, f"None of {len(batches)} batches returned a convincing pick.")
    rank = {"High": 3, "Medium": 2, "Low": 1}
    picks.sort(key=lambda x: rank.get(x.get("equipment_no_confidence"), 0), reverse=True)
    return _finalize_llm_result(row, picks[0], pool, method="llm_batched")


def _finalize_llm_result(row, r, pool, method):
    orig_subloc1 = str(row.get("SubLoc1") or "").strip()
    suggested_subloc1 = r.get("suggested_subloc1") or orig_subloc1
    subloc_conf = r.get("subloc1_confidence", "Low")
    if suggested_subloc1 != orig_subloc1 and subloc_conf == "High":
        subloc_conf = "Medium"
    eq_conf = r.get("equipment_no_confidence", "Low")
    picked_eqp = r.get("suggested_equipment_no")

    suggested_tid = None
    if picked_eqp:
        cand = pool[pool["EquipmentNo"].astype(str) == str(picked_eqp)]
        if not cand.empty:
            suggested_tid = cand.iloc[0]["TechnicalIdentNo"]

    llm_needs_review = r.get("needs_review")
    if llm_needs_review is None:
        llm_needs_review = (not picked_eqp) or eq_conf == "Low"

    return {
        "Suggested TechnicalId":   suggested_tid,
        "Suggested EquipmentNo":   picked_eqp,
        "EquipmentNo Confidence":  eq_conf,
        "EquipmentNo Evidence":    r.get("equipment_no_evidence", ""),
        "Suggested SubLoc1":       suggested_subloc1,
        "SubLoc1 Confidence":      subloc_conf,
        "SubLoc1 Evidence":        r.get("subloc1_evidence", ""),
        "MatchMethod":             method,
        "NeedsReview":             bool(llm_needs_review),
    }


def _no_match_result(row, reason):
    return {
        "Suggested TechnicalId":   None,
        "Suggested EquipmentNo":   None,
        "EquipmentNo Confidence":  "Low",
        "EquipmentNo Evidence":    reason,
        "Suggested SubLoc1":       str(row.get("SubLoc1") or "").strip(),
        "SubLoc1 Confidence":      "Low",
        "SubLoc1 Evidence":        "No mapping.",
        "MatchMethod":             "no_match",
        "NeedsReview":             True,
    }


# =========================================================================
# Stage 1: run_cleaning
# =========================================================================
def run_cleaning(cfg: Config, master: pd.DataFrame, worklist: pd.DataFrame) -> pd.DataFrame:
    """Resolve every worklist row to its SAP master equipment. Writes the
    cleaned workbook and returns the cleaned DataFrame."""
    master = master.copy()
    parts = master["TechnicalIdentNo"].fillna("").astype(str).map(_tid_prefix_and_tail)
    master["_TIDPrefix"] = parts.map(lambda t: t[0])
    master["_TIDTail"]   = parts.map(lambda t: t[1])
    master["_TIDNorm"]   = master["TechnicalIdentNo"].fillna("").astype(str).map(_normalize)
    print(f"Master ready: {len(master)} rows.")

    results = []
    t0 = time.time()
    for idx, row in worklist.iterrows():
        print(f"\n[{idx + 1}/{len(worklist)}] WorklistID={row.get('WorklistID','')} | "
              f"EquipmentNo={row.get('EquipmentNo','')} | SubLoc1={row.get('SubLoc1','')}")
        pick = _resolve_deterministic(row, master) or _run_llm_path(row, master, cfg)
        results.append({
            "ID":                       row.get("ID"),
            "WorklistID":               row.get("WorklistID"),
            "EquipmentNo":              row.get("EquipmentNo"),
            "Suggested TechnicalId":    pick.get("Suggested TechnicalId"),
            "Suggested EquipmentNo":    pick.get("Suggested EquipmentNo"),
            "EquipmentNo Confidence":   pick.get("EquipmentNo Confidence"),
            "EquipmentNo Evidence":     pick.get("EquipmentNo Evidence"),
            "Loc":                      row.get("Loc"),
            "SubLoc1":                  row.get("SubLoc1"),
            "Suggested SubLoc1":        pick.get("Suggested SubLoc1"),
            "SubLoc1 Confidence":       pick.get("SubLoc1 Confidence"),
            "SubLoc1 Evidence":         pick.get("SubLoc1 Evidence"),
            "EqptType":                 row.get("EqptType"),
            "Discipline":               row.get("Discipline"),
            "Job Description":          row.get("Job Description"),
            "FuncLoc":                  row.get("FuncLoc"),
            "MatchMethod":              pick.get("MatchMethod"),
            "NeedsReview":              pick.get("NeedsReview", False),
        })

    out_df = pd.DataFrame(results)
    cfg.cleanly_mapped_worklist.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(cfg.cleanly_mapped_worklist, index=False)
    print(f"\nWrote {cfg.cleanly_mapped_worklist.name} ({len(out_df)} rows)")
    print("\n--- Cleaning summary ---")
    print(out_df["MatchMethod"].value_counts())
    print(f"NeedsReview : {int(out_df['NeedsReview'].sum())}")
    print(f"Elapsed     : {time.time() - t0:.0f}s")
    return out_df


# =========================================================================
# Orchestrator
# =========================================================================
def _stage_header(num: int, total: int, title: str) -> None:
    print("\n" + "-" * 70)
    print(f"  STAGE {num}/{total}  --  {title}")
    print("-" * 70)


def run_pipeline(cfg: Config | None = None) -> dict:
    """Run every enabled stage of the SD & TA Agent pipeline. Returns a
    summary dict with the paths of each produced artifact."""
    cfg = cfg or load_config()
    t0 = time.time()
    print("=" * 70)
    print(f"  SD & TA Agent -- year {cfg.year}")
    print(f"  Processed data dir: {cfg.paths.processed_data_dir}")
    print(f"  Flags: refresh_eq_master={cfg.flags.refresh_equipment_master}  "
          f"refresh_eq_class={cfg.flags.refresh_equipment_class}")
    print(f"         run_cleaning={cfg.flags.run_cleaning}  "
          f"run_job_library={cfg.flags.run_job_library}  "
          f"run_unresolved={cfg.flags.run_unresolved_classifier}")
    print(f"         run_merge={cfg.flags.run_merge}  "
          f"run_sap_export={cfg.flags.run_sap_export}")
    print("=" * 70)

    produced: dict[str, Path] = {}

    # -------- Prep: load ground-truth data (master + class) --------
    _stage_header(0, 5, "LOAD GROUND-TRUTH DATA (equipment master + equipment class)")
    master = fetch_equipment_master(cfg, force_refresh=cfg.flags.refresh_equipment_master)
    print(f"[main]   equipment master -> {len(master)} rows")
    equipment_class_df = fetch_equipment_class(cfg, force_refresh=cfg.flags.refresh_equipment_class)
    print(f"[main]   equipment class  -> {len(equipment_class_df)} rows")

    # -------- Stage 1: Cleaning --------
    _stage_header(1, 5, "CLEANING AGENT (map messy worklist equipment -> SAP master)")
    if cfg.flags.run_cleaning:
        worklist = load_worklist_xlsx(cfg)
        cleaned = run_cleaning(cfg, master, worklist)
        produced["cleanly_mapped_worklist"] = cfg.cleanly_mapped_worklist
    else:
        print(f"[main]   SKIP (run_cleaning=false); reading cached {cfg.cleanly_mapped_worklist.name}")
        cleaned = pd.read_excel(cfg.cleanly_mapped_worklist)
        print(f"[main]   cached cleaned worklist -> {len(cleaned)} rows")

    # -------- Stage 2: Unresolved classifier --------
    _stage_header(2, 5, "UNRESOLVED CLASSIFIER (coarser-class fallback for unmatched rows)")
    if cfg.flags.run_unresolved_classifier:
        _, unresolved_path = classify_unresolved(cfg, cleaned, equipment_class_df)
        produced["unresolved_items"] = unresolved_path
    else:
        print("[main]   SKIP (run_unresolved_classifier=false)")

    # # -------- Stage 3: Per-year Job Library --------
    # _stage_header(3, 5, "JOB LIBRARY AGENT (per-year clustering into Group Counters)")
    # if cfg.flags.run_job_library:
    #     print(f"[main]   loading tasklist: {cfg.tasklist_xlsx.name}")
    #     tasklist = pd.read_excel(cfg.tasklist_xlsx)
    #     print(f"[main]   tasklist -> {len(tasklist)} task rows")
    #     build_job_library_for_year(cfg, cleaned, master, tasklist, equipment_class_df)
    #     produced["job_library_new"] = cfg.job_library_new
    # else:
    #     print("[main]   SKIP (run_job_library=false)")

    # # -------- Stage 4: Merge --------
    # _stage_header(4, 5, "MERGE AGENT (fold new year into consolidated library)")
    # if cfg.flags.run_merge:
    #     run_merge(cfg)
    #     produced["job_library_consolidated"] = cfg.job_library_consolidated
    # else:
    #     print("[main]   SKIP (run_merge=false)")

    # # -------- Stage 5: SAP IA05 export --------
    # _stage_header(5, 5, "SAP IA05 EXPORT (convert consolidated library to SAP upload workbook)")
    # if cfg.flags.run_sap_export:
    #     export_sap(cfg)
    #     produced["sap_upload"] = cfg.sap_upload
    # else:
    #     print("[main]   SKIP (run_sap_export=false)")

    # elapsed = time.time() - t0
    # print("\n" + "=" * 70)
    # print(f"  Pipeline complete in {elapsed:.0f}s.")
    # print("=" * 70)
    # print("  Artifacts produced:")
    # for k, v in produced.items():
    #     print(f"    {k:30s} -> {v}")
    # if not produced:
    #     print("    (none -- all stages skipped by flags)")

    # return {"year": cfg.year, "elapsed_sec": round(elapsed, 1),
    #         "produced": {k: str(v) for k, v in produced.items()}}


if __name__ == "__main__":
    run_pipeline()
