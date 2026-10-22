# SD & TA Agent — SAP_Agent6

End-to-end pipeline that turns Petronas Bokor worklist + tasklist spreadsheets
into a SAP-ready **Job Library IA05 upload workbook**, with three tool-calling
LLM agents and full Langfuse observability.

```
Worklist + Tasklist  --->  Cleaning Agent  ---> Classify unresolved items
                                 |                        |
                                 v                        v
                        Cleaned worklist  --->  Job Library Agent (per class)
                                                          |
                                                          v
                                                Merge Agent (year -> consolidated)
                                                          |
                                                          v
                                             SAP IA05 Exporter (upload workbook)
```

Driven entirely by `config.ini`. **No CLI arguments.** The frontend / orchestrator
can also import `main.run_pipeline(cfg)` directly.

---

## Table of contents

1. [Quick start](#quick-start)
2. [What each file does](#what-each-file-does)
3. [Pipeline stages](#pipeline-stages)
4. [Configuration](#configuration)
5. [Flags & stage toggles](#flags--stage-toggles)
6. [Input files you need to provide](#input-files-you-need-to-provide)
7. [Output files produced](#output-files-produced)
8. [Equipment Class ingestion (one-time / on-demand)](#equipment-class-ingestion-one-time--on-demand)
9. [Observability (Langfuse)](#observability-langfuse)
10. [Common gotchas & troubleshooting](#common-gotchas--troubleshooting)
11. [Changing agent prompts](#changing-agent-prompts)

---

## Quick start

### 1. Install dependencies

Create / activate the `sap_petronas` conda env (or any env with Python 3.10+) and
install:

```powershell
pip install -r requirements.txt
pip install langchain>=1.0 langchain-anthropic>=0.3 langchain-core>=0.3 langfuse>=3.0
```

> **Note:** `requirements.txt` currently pins the base libs (pandas, openpyxl,
> psycopg, azure-identity, python-dotenv, anthropic). The three LangChain / Langfuse
> packages above are needed by the agent modules — add them manually or extend
> `requirements.txt`.

### 2. Make sure `az login` is current

The KG extractors and the Equipment-Class ingest use Azure CLI credentials to
get a Postgres AD token:

```powershell
az login
```

### 3. Drop your input files into `Raw Data New/`

```
SAP_Agent6/
├── Raw Data New/
│   ├── IA05_Planning Norms Electrical.xlsx    (SAP upload template, kept as-is)
│   ├── Worklist_<year>.xlsx                    (sheet "Export")
│   └── Tasklist_<year>.xlsx
```

### 4. Edit `config.ini`

Set the year and tweak flags:

```ini
[run]
year = 2024

[flags]
run_cleaning              = true
run_unresolved_classifier = true
run_job_library           = true
run_merge                 = true
run_sap_export            = true
```

### 5. Run

```powershell
cd "D:\Project Work\SAP PM Revitalization\Code Folder\SD-TA Agent\SAP_Agent6"
python main.py
```

Produced artifacts land in `Processed Data New/`.

---

## What each file does

| File | Purpose |
|---|---|
| `main.py` | Top-level `run_pipeline(cfg)` — orchestrates all 5 stages per the `[flags]` section. Also contains the deterministic matcher + batched LLM path for Stage 1 cleaning (reused from the proven backup). |
| `config.py` | Reads `config.ini` + `.env` and returns a strongly-typed `Config` dataclass. Also exposes year-stamped output paths (`cleanly_mapped_worklist`, `job_library_new`, etc.). |
| `config.ini` | **Single source of truth** for every non-secret setting. Edit this file, not Python code. |
| `.env` | Secrets only: Azure Foundry / Langfuse keys. **Never commit.** |
| `utils.py` | Shared helpers: `build_llm`, `build_langfuse`, `with_rate_limit_retry`, `normalize_cell`, `resolve_class`, `load_job_library`, `write_job_library`, discipline / platform maps. |
| `db_connection.py` | Azure-AD-authenticated psycopg connection to the SAP PM KG Postgres. |
| `kg_extractors.py` | `fetch_equipment_master(cfg)` and `fetch_equipment_class(cfg)` — cached per run.year under `Processed Data New/`. |
| `ingest_equipment_class.py` | Reads `Raw Data New/Equipment Class.xlsx` and upserts into Postgres table `equipment_class`. Runnable standalone (`python ingest_equipment_class.py`) or callable from a frontend API when a new file is uploaded. |
| `cleaning_agent.py` | **Stage 1 agent.** Picks the right SAP master equipment for ONE worklist row. LangChain `create_agent` with 2 tools (`lookup_subloc1_alias`, `submit_cleaning_result`). Thread-safe for parallel batches. Uses response-scan fallback for robust tool-call capture. |
| `unresolved_classifier.py` | **Stage 2 agent.** For rows the Cleaning Agent couldn't match, infers a Discipline root (keyword scan), then walks fallback_levels fine → coarse asking an LLM to pick the best Equipment Class. Writes `Unresolved_Worklist_Items_<year>.xlsx`. |
| `job_library_agent.py` | **Stage 3 agent.** Per equipment class: clusters equipments into Group Counters based on shared maintenance intent, enforcing a 40-char SAP KTEXT limit (with proportional-duration splits where needed) and the detail-first wording rule. |
| `merge_agent.py` | **Stage 4 agent.** Folds the per-year Job Library into `Job_Library_Consolidated.xlsx`. First-year base case = copy; subsequent years = one agent invocation per new-year counter (match-or-create against existing counters). Tracks + prints a per-run merge summary (upgrades, splits, inserts, new counters). |
| `sap_exporter.py` | **Stage 5 converter.** Reads the consolidated library + the IA05 template, writes `Job_Library_IA05_Upload.xlsx`. Preserves the template's row-0-to-4 metadata + per-cell styling (H / INTN / SBOP). Not an agent (deterministic). |
| `requirements.txt` | Base pip deps. **See the Install step** for the LangChain / Langfuse additions. |

---

## Pipeline stages

### Stage 0 — Load ground-truth data
- `fetch_equipment_master(cfg)` → KG Postgres → `Equipment_Master_Bokor_<year>.xlsx` (cache)
- `fetch_equipment_class(cfg)` → KG Postgres (`equipment_class` table) → `Equipment_Class_cache.xlsx` (cache)
- Set `[flags].refresh_equipment_master=true` to force a fresh KG pull.

### Stage 1 — Cleaning Agent
For each worklist row:
1. **Deterministic path** — token-extract the worklist EquipmentNo, prepend the SubLoc1-implied platform prefix, look up in the master's TID index. Single-hit → use it with High confidence.
2. **LLM path (if deterministic missed)** — filter the master to the SubLoc1 prefix (e.g. `BOCP` → 11,700 rows). If pool ≤ `max_pool_size_for_llm` (default 20000), batch into `batch_size` chunks (default 100) and run `max_concurrent_llm` (default 3) parallel LLM batches. Each batch returns a pick with confidence; the highest-confidence pick across batches wins.
3. **Over the pool cap?** The row is marked `no_match` with evidence explaining why — Stage 2 picks it up.

Output: `Cleanly_Mapped_Worklist_Bokor_<year>.xlsx`.

### Stage 2 — Unresolved Classifier
For every row from Stage 1 with no `Suggested EquipmentNo`:
1. **Step A (deterministic)** — infer a Level-1 discipline root from `Discipline` + `EqptType` + `Job Description` keywords.
2. **Step B (LLM walk)** — for each level in `[unresolved_classifier].fallback_levels` (default `4,3,2`), pass all Equipment Class codes under the Step-A root at that level to the LLM; it picks the best match or returns NONE.
3. **Fallback** — if every level returns NONE, use the Level-1 root itself (never fails).

Output: `Unresolved_Worklist_Items_<year>.xlsx` with the classified class + reviewer-override columns (`MANUAL_EquipmentNo`, `MANUAL_TechnicalIdentNo`, `REVIEWER_NOTES`).

### Stage 3 — Job Library Agent (per year)
1. Join cleaned worklist to Equipment Master to attach `CatalogProfile` + descriptors.
2. For each equipment, resolve its Equipment Class (walkup against the class hierarchy, preferring `[job_library_agent].class_level`).
3. For each class, invoke the Job Library Agent once:
   - `get_class_equipments(class_code)` → returns the equipments + their task sequences + durations
   - Agent clusters by shared maintenance intent (semantic, not wording-based), produces generalised operations ≤ 40 chars, splits long steps with proportional durations.
   - `save_job_counters(class_code, counters_json)` → persists the result.

Output: `Job_Library_Bokor_<year>_new.xlsx` (3 sheets: `EquipmentClassMap`, `JobLibrary`, `CounterEquipments`).

Also prints a 40-char SAP-limit audit — flags any operation that slipped the limit (should be zero).

### Stage 4 — Merge Agent
Folds the per-year library into `Job_Library_Consolidated.xlsx`:

- **First year** (no consolidated file yet) → simple copy, no agent calls.
- **Subsequent years** → for each new-year counter, one agent invocation:
  1. `get_counters_in_group(tasklist_group)` → existing consolidated counters in this group
  2. Agent decides **match existing or create new** based on shared procedure intent.
  3. If match: picks per-op decisions (`keep_existing` / `upgrade_wording` / `replace_with_split` / `insert_new`) and calls `assign_to_counter(...)` to apply them. Adds the new equipments (deduped by EquipmentNo).
  4. If no match: calls `create_new_counter(...)`.

End-of-run summary prints: new counters, matched existing, improved vs unchanged, operation-level changes, new equipments added.

### Stage 5 — SAP IA05 Export
Reads `Job_Library_Consolidated.xlsx` + the IA05 template. For each counter writes **three rows** in SAP IA05 format:

```
Row type      | TL Ind | Item Flag | VORNR | UVORN | STEUS | KTEXT  | LTXA1   | DAUNO
H  (header)   | A      | H         |       |       |       | header |         |
I  (INTN sum) |        | I         | 0010  |       | INTN  |        | header  | total
I  (SBOP x N) |        | I         | 0010  | 0001+ | SBOP  |        | op text | per-op
```

- Plant = `[plant].werks` (= `M014`)
- Work Center mapped from each counter's dominant Discipline (`Electrical → PMO-ELE`, `Instrument & Control → PMO-INS`, `Static/Rotating/Piping → PMO-MEC`, `Construction → PMO-CON`, default `PMO-ELE`)
- Rows 0–4 of the IA05 template (Mandatory / Optional / SAP codes / lengths / human headers) are preserved verbatim with styling. The template's own sample rows 6/7/8 are used as the H / INTN / SBOP style templates — their per-cell fill/font/border/alignment/format is cloned onto every written row.

Output: `Job_Library_IA05_Upload.xlsx`.

---

## Configuration

### `.env` — secrets only

```
AZURE_AI_KEY=...
AZURE_FOUNDRY_API_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
```

Never commit this file.

### `config.ini` — everything else

Grouped sections; the loader interpolates `${section:key}` references.

```ini
[paths]
raw_data_dir        = ./Raw Data New
processed_data_dir  = ./Processed Data New
ia05_template       = ${raw_data_dir}/IA05_Planning Norms Electrical.xlsx

[plant]
code             = SK          ; prefix for TaskListGroup, e.g. SKET, SKMSP
werks            = M014        ; SAP WERKS code
location_filter  = BOKOR       ; KG equipment-master location filter

[run]
year = 2024                    ; which Worklist_<year>.xlsx / Tasklist_<year>.xlsx to process

[llm]
model               = claude-sonnet-4-6
foundry_base_url    = https://tridiagonal-ai-eastus2-resource.services.ai.azure.com/anthropic
temperature         = 0
max_tokens_default  = 4096
max_tokens_cleaning = 2000     ; cleaning agent has a bigger prompt+candidates payload

[langfuse]
host                = http://65.1.79.48:3000
project             = SAP PM Data Revitalization
agent_name          = SD & TA Agent
session_cleaning    = bokor-cleaning
session_job_library = bokor-job-library
session_merge       = bokor-merge
session_unresolved  = bokor-unresolved

[kg]
host         = pg-db-petronas-sap-prd.postgres.database.azure.com
port         = 5432
database     = sap_pm_kg
user         = abhishek.mali@tridiagonal.ai
token_scope  = https://ossrdbms-aad.database.windows.net/.default

[cleaning_agent]
batch_size            = 100
max_concurrent_llm    = 3
max_pool_size_for_llm = 20000  ; skip LLM path if pool exceeds this (routed to Unresolved)

[job_library_agent]
class_level = 4                ; preferred Equipment Class level for TaskListGroup resolution (2|3|4)

[unresolved_classifier]
fallback_levels = 4,3,2        ; walk from fine to coarse

[flags]
refresh_equipment_master  = false
refresh_equipment_class   = false
run_cleaning              = true
run_unresolved_classifier = true
run_job_library           = true
run_merge                 = true
run_sap_export            = true

[rate_limit]
max_retries      = 5
default_wait_sec = 20
```

### Inline-comment gotcha ⚠

Python's `ConfigParser` does NOT strip inline comments by default. Keep comments
on their own lines prefixed with `;`:

```ini
; GOOD -- comment on its own line
run_cleaning = true

; BAD -- the value becomes "true   # comment..." and _bool() returns False
run_cleaning = true   # comment here
```

---

## Flags & stage toggles

Each `run_*` flag independently gates a stage. Common combinations:

| Scenario | Flags to set |
|---|---|
| Full pipeline on fresh data | all `run_*` = true, `refresh_equipment_master` = true, `refresh_equipment_class` = true |
| Re-export only (no agent calls) | only `run_sap_export` = true |
| Just test the Cleaning Agent | `run_cleaning` = true, everything else false |
| Re-run merge after a hand-edit to the per-year library | `run_merge` = true, others false |
| Rebuild Job Library after a prompt tweak, re-export SAP | `run_job_library`, `run_merge`, `run_sap_export` = true |

**Skip-flag + missing-cache**: skipping a stage requires its upstream output to exist on disk. If `run_cleaning=false` and `Cleanly_Mapped_Worklist_...xlsx` is missing, you'll get a `FileNotFoundError`.

---

## Input files you need to provide

Drop these into `Raw Data New/`:

| File | Shape | Notes |
|---|---|---|
| `IA05_Planning Norms Electrical.xlsx` | 32-col template, header on row 4 | Keep as the SAP-provided template; styling and metadata rows are preserved |
| `Worklist_<year>.xlsx` | Sheet named `Export` with columns `ID`, `WorklistID`, `EquipmentNo`, `Loc`, `SubLoc1`, `EqptType`, `Discipline`, `Job Description`, `FuncLoc` | `Loc` filtered to `[plant].location_filter` (default `BOKOR`); `WorklistID` must be unique |
| `Tasklist_<year>.xlsx` | Columns `WorklistID`, `EquipmentNo`, `TaskNo`, `Task`, `Dur (Hrs)`, `Discipline` | Multiple rows per `WorklistID` (one per operation); `Task` is free text |
| `Equipment Class.xlsx` *(optional)* | Columns `Level`, `Class`, `Class Description`, `Status` | Only needed if the Postgres `equipment_class` table isn't populated yet. See [Equipment Class ingestion](#equipment-class-ingestion-one-time--on-demand). |

---

## Output files produced

In `Processed Data New/`:

| File | Produced by | Content |
|---|---|---|
| `Equipment_Master_Bokor_<year>.xlsx` | KG fetch (cached) | 7 cols + `FunctionalLocation` for every BOKOR equipment |
| `Equipment_Class_cache.xlsx` | KG fetch (cached) | Class hierarchy from Postgres `equipment_class` |
| `Cleanly_Mapped_Worklist_Bokor_<year>.xlsx` | Stage 1 | Every worklist row + Suggested EquipmentNo + confidence + evidence + MatchMethod + NeedsReview |
| `Unresolved_Worklist_Items_<year>.xlsx` | Stage 2 | Rows with no SAP match + Classified_Class + MANUAL_* override columns |
| `Job_Library_Bokor_<year>_new.xlsx` | Stage 3 | 3-sheet workbook: EquipmentClassMap, JobLibrary (header + ops + total), CounterEquipments (with EquipmentDescription, TechnicalIdentNo, FunctionalLocation) |
| `Job_Library_Consolidated.xlsx` | Stage 4 | Rolling consolidated library across all years |
| `Job_Library_IA05_Upload.xlsx` | Stage 5 | SAP IA05 upload template format with H/INTN/SBOP rows |

---

## Equipment Class ingestion (one-time / on-demand)

The Equipment Class hierarchy lives in Postgres table `equipment_class`.

### First-time setup

Drop `Equipment Class.xlsx` into `Raw Data New/` and run:

```powershell
python ingest_equipment_class.py
```

Walks the file, validates required columns (`Level`, `Class`, `Class Description`, optional `Status`), and upserts into Postgres. The default is **REPLACE mode** (TRUNCATE then insert). The script also auto-creates the table on first run.

### Called from a frontend API

```python
from ingest_equipment_class import ingest_equipment_class
result = ingest_equipment_class("/path/to/new_file.xlsx", replace=True)
# {"rows_ingested": N, "rows_rejected": M, "errors": [...]}
```

Use `replace=False` to upsert in place (merge) rather than wipe + reinsert.

---

## Observability (Langfuse)

Every LLM call across all four agents is traced through Langfuse. On each
`agent.invoke(...)` the following metadata is attached:

```python
run_name            = "SD & TA Agent - <stage> - <id>"
session_id          = (session_cleaning | session_job_library | session_merge | session_unresolved)
tags                = ["SD & TA Agent", "SAP PM Data Revitalization", <stage>, <id>, ...]
metadata            = { worklist_id | equipment_class | tasklist_group, ... }
```

The host, project name, and session IDs come from the `[langfuse]` section of
`config.ini`.

### If traces don't appear in Langfuse

- The Langfuse server at `[langfuse].host` must be reachable from your machine.
- If the server is behind VPN, you'll see `Failed to export spans batch` warnings — these are **non-fatal** (the pipeline keeps running) but the traces aren't delivered.
- A server-side 500 (`Failed to upload JSON to S3`) means the Langfuse infra is misconfigured — fix on the server, not here.

---

## Common gotchas & troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `az login` prompt hangs forever | Stale AD token | Run `az login` in the same terminal before launching the pipeline |
| `No module named 'langchain_anthropic'` | LangChain not installed in your env | `pip install langchain langchain-anthropic langchain-core langfuse` |
| `Rate limit of 50000 per 60s exceeded` | Azure Foundry quota | The pipeline already retries with the server's hinted wait; be patient. Lower `max_concurrent_llm` to reduce burst. |
| `agent did NOT call submit_cleaning_result` on many batches | Model replied in text instead of calling the tool | Already mitigated — the agent uses forced-tool-call pattern + response-scan fallback. If you still see this often, bump `max_tokens_cleaning` in `[llm]`. |
| `Pool exceeds max_pool_size_for_llm` | Worklist row has a huge candidate pool (e.g. SubLoc1=BOCPP-A → 11,700 BOCP rows) | Expected behaviour — row routed to Unresolved Classifier. Bump `max_pool_size_for_llm` under `[cleaning_agent]` if you want to force the LLM to try. |
| `FileNotFoundError` on cached file | A `run_*` flag is false but the file doesn't exist | Either enable the stage or produce the file manually. |
| 40-char audit shows violations after Stage 3 | Agent produced an operation > 40 chars | Open `Job_Library_Bokor_<year>_new.xlsx`, inspect the offending op, iterate the system prompt in `job_library_agent.py` and re-run Stage 3. |
| Final SAP output lost styling | IA05 template was edited and the sample rows (6/7/8) no longer exist | Restore the template or point `[paths].ia05_template` at a clean copy. |

---

## Changing agent prompts

Each agent's `_SYSTEM_PROMPT` is a module-level string. To iterate:

1. Edit the prompt in `cleaning_agent.py` / `unresolved_classifier.py` / `job_library_agent.py` / `merge_agent.py`.
2. Run the single stage affected (e.g. `run_cleaning=true`, others false).
3. Check the output file.
4. Repeat.

All four agents use the **forced tool-call pattern** (LangChain `create_agent` + a MANDATORY output-contract clause in the prompt) + a **response-scan fallback** (so even if the `@tool` body doesn't physically execute, we still capture the args from the AIMessage). This is why the agents are reliable about producing structured output even when the model would otherwise reply in natural language.

---

## File-by-file dependency chain

```
config.ini + .env
      |
      v
   config.py
      |
      +---> utils.py --------> cleaning_agent.py
      |                        unresolved_classifier.py
      |                        job_library_agent.py
      |                        merge_agent.py
      |                        sap_exporter.py
      |
      +---> db_connection.py --> kg_extractors.py --+
      |                          ingest_equipment_class.py
      |
      v
   main.py  (orchestrator -- imports every agent)
```

Only `main.py` is the entry point. The frontend API would also use
`main.run_pipeline(cfg)` after loading / overriding the config.
