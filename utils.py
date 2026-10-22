"""Shared helpers for the SD & TA Agent. Everything here was previously
duplicated across 2-3 of the agent files. One source of truth now.

Groups:
  * LLM + Langfuse factories      -- build_llm, build_langfuse, build_langfuse_config
  * Rate-limit retry              -- with_rate_limit_retry, is_rate_limit, wait_hint
  * Cell / dataframe normalisers  -- normalize_cell
  * Equipment-class resolver      -- load_equipment_class_lookup, resolve_class
  * Job Library IO                -- load_job_library, write_job_library
  * Static maps                   -- PREFIX_TO_SUBLOC1 / SUBLOC1_TO_PREFIX /
                                      DISCIPLINE_TO_WORKCENTER / DISCIPLINE_TO_LEVEL1
"""

import copy as _copy_mod
import json
import re
import time
from copy import copy as _copy
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from langchain_anthropic import ChatAnthropic
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from openpyxl.styles import Font, PatternFill

from config import Config


# =========================================================================
# Static maps (platform / discipline lookups used across agents)
# =========================================================================
PREFIX_TO_SUBLOC1 = {
    "BOCP": "BOCPP-A",
    "BOKA": "BOK-A",
    "BOPA": "BOP-A",
    "BOVA": "BOV-A",
    "BODA": "BODP-A",
    "BODB": "BODP-B",
    "BODC": "BODP-C",
    "BODD": "BODP-D",
    "BODE": "BODP-E",
    "BODF": "BODP-F",
}
SUBLOC1_TO_PREFIX = {v: k for k, v in PREFIX_TO_SUBLOC1.items()}

DISCIPLINE_TO_WORKCENTER = {
    "Electrical":              "PMO-ELE",
    "Instrument & Control":    "PMO-INS",
    "Instrument":              "PMO-INS",
    "Static":                  "PMO-MEC",
    "Rotating":                "PMO-MEC",
    "Piping":                  "PMO-MEC",
    "Mechanical":              "PMO-MEC",
    "Construction":            "PMO-CON",
}
DEFAULT_WORKCENTER = "PMO-ELE"

# Used by unresolved_classifier (Step A - deterministic Level-1 mapping).
DISCIPLINE_TO_LEVEL1 = {
    "Electrical":              "E",
    "Instrument & Control":    "I",
    "Instrument":              "I",
    "Static":                  "M",
    "Rotating":                "M",
    "Piping":                  "M",
    "Mechanical":              "M",
    "Construction":            "C",
}
UNKNOWN_LEVEL1 = "ZZ"


# =========================================================================
# LLM + Langfuse factories
# =========================================================================
def build_llm(cfg: Config, max_tokens: int | None = None) -> ChatAnthropic:
    """ChatAnthropic client wired to Azure Foundry; max_tokens overridable."""
    return ChatAnthropic(
        model=cfg.llm.model,
        api_key=cfg.secrets.azure_foundry_api_key or cfg.secrets.azure_ai_key,
        base_url=cfg.llm.foundry_base_url,
        temperature=cfg.llm.temperature,
        max_tokens=int(max_tokens or cfg.llm.max_tokens_default),
    )


def build_langfuse():
    """Return (langfuse_client, CallbackHandler). Reads LANGFUSE_* env vars
    which config.load_config() populates from .env."""
    return get_client(), CallbackHandler()


def build_langfuse_config(
    cfg: Config,
    run_name: str,
    session_id: str,
    callback_handler,
    tags: Iterable[str] = (),
    metadata: dict | None = None,
) -> dict:
    """Build the dict passed to agent.invoke(..., config=<this>)."""
    base_tags = [cfg.langfuse.agent_name, cfg.langfuse.project]
    all_tags = list(base_tags) + list(tags)
    md = {
        "langfuse_trace_name": run_name,
        "langfuse_user_id": "sd-ta-pipeline",
        "langfuse_session_id": session_id,
        "langfuse_tags": all_tags,
        "project": cfg.langfuse.project,
    }
    if metadata:
        md.update(metadata)
    return {"run_name": run_name, "callbacks": [callback_handler], "metadata": md}


# =========================================================================
# Rate-limit retry
# =========================================================================
def is_rate_limit(err) -> bool:
    msg = str(err)
    return "429" in msg or "RateLimit" in msg or "rate limit" in msg.lower()


def wait_hint(err, attempt: int, default_wait: int = 20) -> int:
    """Pull 'Please wait X seconds' from the error; otherwise exponential."""
    m = re.search(r"wait (\d+) seconds", str(err))
    if m:
        return int(m.group(1)) + 2
    return default_wait * attempt


def with_rate_limit_retry(fn: Callable, *args, cfg: Config | None = None,
                          max_retries: int | None = None, default_wait: int | None = None,
                          **kwargs):
    """Call fn(*args, **kwargs); on 429, parse wait hint and retry."""
    retries = max_retries if max_retries is not None else (cfg.rate_limit.max_retries if cfg else 5)
    wait    = default_wait if default_wait is not None else (cfg.rate_limit.default_wait_sec if cfg else 20)
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if is_rate_limit(e) and attempt < retries:
                w = wait_hint(e, attempt, default_wait=wait)
                print(f"    Rate limit hit on attempt {attempt}, sleeping {w}s...")
                time.sleep(w)
                continue
            raise


# =========================================================================
# Cell / DataFrame normaliser
# =========================================================================
def normalize_cell(v) -> str:
    """NaN / None / literal 'nan' -> ''. Everything else stripped."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


# =========================================================================
# Equipment-class resolver
# =========================================================================
def load_equipment_class_lookup(df: pd.DataFrame) -> dict:
    """df needs columns Level, Class, Class Description. Returns
    {class_code: {'level': int, 'description': str}}."""
    df = df[["Level", "Class", "Class Description"]].dropna(subset=["Class"]).copy()
    df.columns = ["Level", "Class", "Description"]
    df["Class"] = df["Class"].astype(str).str.strip().str.upper()
    return {
        r.Class: {"level": int(r.Level), "description": r.Description}
        for r in df.itertuples(index=False)
    }


def resolve_class(cp, lookup: dict, preferred_level: int) -> tuple:
    """CatalogProfile '1EJCL' -> walk prefixes; prefer `preferred_level`,
    roll up to coarser if not found. Returns (class_code, level, description)
    or (None, None, None)."""
    if pd.isna(cp):
        return None, None, None
    core = re.sub(r"^\d+", "", str(cp).strip().upper())
    for n in range(len(core), 0, -1):
        hit = lookup.get(core[:n])
        if hit and hit["level"] <= preferred_level:
            return core[:n], hit["level"], hit["description"]
    return None, None, None


def classes_at_level_under_root(lookup: dict, root_code: str, level: int) -> list[dict]:
    """Return all class codes starting with root_code (e.g. 'E') at a given
    level. Used by unresolved_classifier."""
    root = root_code.upper()
    out = []
    for code, info in lookup.items():
        if info["level"] == level and code.startswith(root):
            out.append({"class": code, "description": info["description"]})
    return sorted(out, key=lambda x: x["class"])


# =========================================================================
# Job Library IO  (the 3-sheet workbook used by both merge + exporter)
# =========================================================================
def load_job_library(path: Path) -> dict:
    """Read a 3-sheet Job Library workbook and return nested JSON:
        {
          "equipment_map":   DataFrame (Sheet 1),
          "counters":        {group: [counter_dict, ...]},
        }
    counter_dict = {group, counter, header, operations[],
                    total_hrs, equipments[], discipline}
    where discipline is the dominant discipline across the counter's equipments
    (used by sap_exporter for Work Center mapping)."""
    xl = pd.ExcelFile(path)
    sheet1 = pd.read_excel(xl, "EquipmentClassMap")
    sheet2 = pd.read_excel(xl, "JobLibrary")
    sheet3 = pd.read_excel(xl, "CounterEquipments")

    # Discipline per equipment (from Sheet 1).
    disc_by_eq: dict[str, str] = {}
    if "EquipmentNo" in sheet1.columns and "Discipline" in sheet1.columns:
        for _, r in sheet1.iterrows():
            eq = normalize_cell(r.get("EquipmentNo"))
            if eq and eq not in disc_by_eq:
                disc_by_eq[eq] = normalize_cell(r.get("Discipline"))

    # Walk Sheet 2 reconstructing counters.
    counters_by_group: dict[str, list] = {}
    current = None
    for _, r in sheet2.iterrows():
        group    = normalize_cell(r.get("TaskListGroup"))
        counter  = normalize_cell(r.get("GroupCounter"))
        header   = normalize_cell(r.get("Header"))
        item_no  = normalize_cell(r.get("ItemNo"))
        op_desc  = normalize_cell(r.get("OperationDescription"))
        dur      = r.get("Duration (Hrs)")

        if op_desc.upper() == "TOTAL":
            continue
        if header and not item_no:
            current = {"group": group, "counter": counter, "header": header,
                       "operations": [], "total_hrs": 0.0, "equipments": [],
                       "discipline": ""}
            counters_by_group.setdefault(group, []).append(current)
            continue
        if item_no and current is not None:
            try:
                dur_f = float(dur) if pd.notna(dur) and str(dur).strip() != "" else 0.0
            except Exception:
                dur_f = 0.0
            current["operations"].append({
                "item_no": item_no, "description": op_desc,
                "duration_hrs": round(dur_f, 2),
            })
            current["total_hrs"] = round(current["total_hrs"] + dur_f, 2)

    # Attach equipments from Sheet 3 and compute dominant discipline per counter.
    from collections import Counter as _C
    counter_disciplines: dict[tuple, list] = {}
    for _, r in sheet3.iterrows():
        g = normalize_cell(r.get("TaskListGroup"))
        c = normalize_cell(r.get("GroupCounter"))
        eq_no = normalize_cell(r.get("EquipmentNo"))
        for cnt in counters_by_group.get(g, []):
            if cnt["counter"] == c:
                cnt["equipments"].append({
                    "equipment_no": eq_no,
                    "description": normalize_cell(r.get("EquipmentDescription")),
                    "technical_ident_no": normalize_cell(r.get("TechnicalIdentNo")),
                    "functional_location": normalize_cell(r.get("FunctionalLocation")),
                })
                d = disc_by_eq.get(eq_no)
                if d:
                    counter_disciplines.setdefault((g, c), []).append(d)
                break

    for cnt_list in counters_by_group.values():
        for cnt in cnt_list:
            key = (cnt["group"], cnt["counter"])
            ds = counter_disciplines.get(key, [])
            if ds:
                cnt["discipline"] = _C(ds).most_common(1)[0][0]

    return {"equipment_map": sheet1, "counters": counters_by_group}


def write_job_library(path: Path, counters_by_group: dict, equipment_map: pd.DataFrame,
                      plant: str) -> None:
    """Write a 3-sheet Job Library workbook (EquipmentClassMap / JobLibrary /
    CounterEquipments) with HEADER + OPERATION + TOTAL styling. Overwrites
    `path`."""
    job_lib_rows = []
    counter_eq_rows = []
    for group in sorted(counters_by_group):
        for cnt in counters_by_group[group]:
            ctr = cnt["counter"]
            header = cnt["header"]
            ops = cnt["operations"]
            total = 0.0
            job_lib_rows.append({
                "TaskListGroup": group, "GroupCounter": ctr, "Header": header,
                "Plant": plant, "ItemNo": "", "OperationDescription": "",
                "Duration (Hrs)": "", "Unit": "", "RowType": "HEADER",
            })
            for i, op in enumerate(ops, start=1):
                dur = float(op.get("duration_hrs") or 0)
                total += dur
                job_lib_rows.append({
                    "TaskListGroup": group, "GroupCounter": ctr, "Header": "",
                    "Plant": plant, "ItemNo": f"{i * 10:04d}",
                    "OperationDescription": op["description"],
                    "Duration (Hrs)": round(dur, 2), "Unit": "HR",
                    "RowType": "OPERATION",
                })
            job_lib_rows.append({
                "TaskListGroup": group, "GroupCounter": ctr, "Header": "",
                "Plant": "", "ItemNo": "", "OperationDescription": "TOTAL",
                "Duration (Hrs)": round(total, 2), "Unit": "HR",
                "RowType": "TOTAL",
            })
            for e in cnt["equipments"]:
                counter_eq_rows.append({
                    "TaskListGroup": group, "GroupCounter": ctr, "Header": header,
                    "EquipmentNo": e.get("equipment_no", ""),
                    "EquipmentDescription": e.get("description", ""),
                    "TechnicalIdentNo": e.get("technical_ident_no", ""),
                    "FunctionalLocation": e.get("functional_location", ""),
                })

    sheet2 = pd.DataFrame(job_lib_rows)
    sheet3 = pd.DataFrame(counter_eq_rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        equipment_map.to_excel(xw, sheet_name="EquipmentClassMap", index=False)
        sheet2.drop(columns=["RowType"], errors="ignore").to_excel(
            xw, sheet_name="JobLibrary", index=False)
        sheet3.to_excel(xw, sheet_name="CounterEquipments", index=False)

        ws = xw.sheets["JobLibrary"]
        hdr_fill = PatternFill("solid", fgColor="305496")
        hdr_font = Font(bold=True, color="FFFFFF")
        tot_fill = PatternFill("solid", fgColor="FFF2CC")
        tot_font = Font(bold=True)
        for i, row_type in enumerate(sheet2["RowType"], start=2):
            if row_type == "HEADER":
                for c in range(1, ws.max_column + 1):
                    ws.cell(i, c).fill = hdr_fill
                    ws.cell(i, c).font = hdr_font
            elif row_type == "TOTAL":
                for c in range(1, ws.max_column + 1):
                    ws.cell(i, c).fill = tot_fill
                    ws.cell(i, c).font = tot_font
        for c in range(1, ws.max_column + 1):
            ws.column_dimensions[ws.cell(1, c).column_letter].width = 22
