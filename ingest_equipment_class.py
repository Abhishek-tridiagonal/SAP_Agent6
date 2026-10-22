"""Ingest an Equipment Class workbook into the Postgres `equipment_class` table.

Invoked by:
  * The frontend API, after a user uploads a new Equipment Class file. The
    frontend saves the upload to a staging path and calls
    `ingest_equipment_class(path)`.
  * Locally, via `python ingest_equipment_class.py` which ingests the default
    `Raw data/Equipment Class.xlsx` -- useful for first-time setup and
    local re-runs without the frontend.

The template the frontend offers to users must match this schema exactly:
    required columns:  Level (int 1..4), Class (str), Class Description (str)
    optional columns:  Status (int: 1=Release, 2=Blocked)
"""

from pathlib import Path

import pandas as pd

from config import Config, load_config
from db_connection import get_connection


REQUIRED_COLS = ["Level", "Class", "Class Description"]
OPTIONAL_COLS = ["Status"]

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS equipment_class (
    class       TEXT PRIMARY KEY,
    level       INTEGER NOT NULL,
    description TEXT    NOT NULL,
    status      INTEGER
);
"""

TRUNCATE_SQL = "TRUNCATE TABLE equipment_class;"

UPSERT_SQL = """
INSERT INTO equipment_class (class, level, description, status)
VALUES (%s, %s, %s, %s)
ON CONFLICT (class) DO UPDATE SET
    level       = EXCLUDED.level,
    description = EXCLUDED.description,
    status      = EXCLUDED.status;
"""


def _validate(df: pd.DataFrame) -> tuple[list, list]:
    """Keep valid rows; collect rejection reasons for the rest.
    Returns (kept_rows: list[dict], rejected: list[dict])."""
    rejected = []
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'. "
                             f"Expected columns: {REQUIRED_COLS + OPTIONAL_COLS}")

    df = df[REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in df.columns]].copy()
    if "Status" not in df.columns:
        df["Status"] = None

    df["Class"] = df["Class"].astype(str).str.strip().str.upper()
    df["Class Description"] = df["Class Description"].astype(str).str.strip()
    df["Level"] = pd.to_numeric(df["Level"], errors="coerce")
    df["Status"] = pd.to_numeric(df["Status"], errors="coerce")

    kept_rows = []
    for i, r in df.iterrows():
        if not r["Class"] or r["Class"].lower() == "nan":
            rejected.append({"row": int(i), "reason": "empty Class"})
            continue
        if pd.isna(r["Level"]) or int(r["Level"]) not in (1, 2, 3, 4):
            rejected.append({"row": int(i), "reason": f"invalid Level '{r['Level']}'"})
            continue
        if not r["Class Description"] or r["Class Description"].lower() == "nan":
            rejected.append({"row": int(i), "reason": "empty Class Description"})
            continue
        kept_rows.append({
            "class": r["Class"],
            "level": int(r["Level"]),
            "description": r["Class Description"],
            "status": None if pd.isna(r["Status"]) else int(r["Status"]),
        })

    return kept_rows, rejected


def ingest_equipment_class(xlsx_path: Path, cfg: Config | None = None,
                           replace: bool = True) -> dict:
    """Read Equipment Class workbook -> upsert into Postgres `equipment_class`.

    Args:
      xlsx_path: path to the uploaded / local workbook.
      cfg:       optional pre-loaded Config (frontend may pass one).
      replace:   True = TRUNCATE then insert (fresh snapshot).
                 False = upsert in place (merge).

    Returns:
      {"rows_ingested": N, "rows_rejected": M, "errors": [{row, reason}, ...]}
    """
    print("=" * 70)
    print("  Equipment Class ingest -> Postgres table `equipment_class`")
    print("=" * 70)
    cfg = cfg or load_config()

    xlsx_path = Path(xlsx_path)
    print(f"[ingest] source file  : {xlsx_path}")
    print(f"[ingest] mode         : {'REPLACE (TRUNCATE + insert)' if replace else 'UPSERT (merge)'}")

    if not xlsx_path.exists():
        print(f"[ingest] ABORT -- file not found: {xlsx_path}")
        return {"rows_ingested": 0, "rows_rejected": 0,
                "errors": [{"row": -1, "reason": f"file not found: {xlsx_path}"}]}

    # ----- Step 1 : Read xlsx
    print(f"\n[ingest] Step 1/5 : reading {xlsx_path.name} ...")
    df = pd.read_excel(xlsx_path)
    print(f"[ingest]            {len(df)} rows x {len(df.columns)} cols loaded")

    # ----- Step 2 : Validate
    print(f"\n[ingest] Step 2/5 : validating rows against schema ...")
    print(f"[ingest]            required columns = {REQUIRED_COLS}")
    kept, rejected = _validate(df)
    print(f"[ingest]            {len(kept)} valid / {len(rejected)} rejected")
    if rejected:
        print(f"[ingest]            first few rejection reasons:")
        for r in rejected[:5]:
            print(f"[ingest]              row {r['row']}: {r['reason']}")

    # ----- Step 3 : Open DB connection (Azure AD token via `az login`)
    print(f"\n[ingest] Step 3/5 : opening Postgres connection ...")
    print(f"[ingest]            host     = {cfg.kg.host}")
    print(f"[ingest]            database = {cfg.kg.database}")
    print(f"[ingest]            user     = {cfg.kg.user}")

    with get_connection(cfg) as conn, conn.cursor() as cur:
        # ----- Step 4 : Ensure table + replace mode wipe
        print(f"\n[ingest] Step 4/5 : ensuring table `equipment_class` exists ...")
        cur.execute(CREATE_SQL)
        if replace:
            print("[ingest]            REPLACE mode -- truncating existing table")
            cur.execute(TRUNCATE_SQL)
        else:
            print("[ingest]            UPSERT mode -- existing rows kept, matched rows updated")

        # ----- Step 5 : Insert
        print(f"\n[ingest] Step 5/5 : upserting {len(kept)} rows ...")
        for i, row in enumerate(kept, 1):
            cur.execute(UPSERT_SQL,
                        (row["class"], row["level"], row["description"], row["status"]))
            if i % 50 == 0:
                print(f"[ingest]            {i}/{len(kept)} rows inserted")
        conn.commit()
        print(f"[ingest]            committed {len(kept)} rows")

    print("\n" + "=" * 70)
    print(f"  Done. {len(kept)} rows ingested, {len(rejected)} rejected.")
    print("=" * 70)

    return {
        "rows_ingested": len(kept),
        "rows_rejected": len(rejected),
        "errors": [{"row": r["row"], "reason": r["reason"]} for r in rejected],
    }


if __name__ == "__main__":
    cfg = load_config()
    result = ingest_equipment_class(cfg.paths.equipment_class_xlsx, cfg=cfg, replace=True)
    print()
    print(f"Summary -> Rows ingested : {result['rows_ingested']}")
    print(f"           Rows rejected : {result['rows_rejected']}")
    if result["errors"]:
        print("First 5 rejections:")
        for e in result["errors"][:5]:
            print(f"  row {e['row']}: {e['reason']}")
