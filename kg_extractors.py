"""Pulls ground-truth data from the SAP PM knowledge graph (Postgres).

Two live extractors + two stubs for the deferred Worklist/Tasklist migration.

  fetch_equipment_master(cfg, force_refresh=False) -> pd.DataFrame
  fetch_equipment_class (cfg, force_refresh=False) -> pd.DataFrame
  fetch_worklist(cfg)   -> NotImplementedError  (deferred)
  fetch_tasklist(cfg)   -> NotImplementedError  (deferred)

Both live extractors cache to xlsx under paths.processed_data_dir. If the
cache exists and `force_refresh=False`, the cache is returned without hitting
the DB (safer, faster, no `az login` prompt for repeat runs).
"""

import pandas as pd

from config import Config
from db_connection import get_connection


# =========================================================================
# Equipment Master  (Bokor equipment, 7 cols + FunctionalLocation join)
# =========================================================================
_EQUIP_SQL = """
    SELECT
        n.node_id,
        n.natural_key AS "EquipmentNo",
        MAX(np.value_text) FILTER (WHERE np.property_name = 'object_description') AS "ObjectDescription",
        MAX(np.value_text) FILTER (WHERE np.property_name = 'catalog_profile')    AS "CatalogProfile",
        MAX(np.value_text) FILTER (WHERE np.property_name = 'technical_identno')  AS "TechnicalIdentNo",
        MAX(np.value_text) FILTER (WHERE np.property_name = 'functlocdescrip')    AS "FunctLocDescrip",
        MAX(np.value_text) FILTER (WHERE np.property_name = 'system_status')      AS "SystemStatus"
    FROM nodes n
    JOIN node_properties np ON np.node_id = n.node_id
    WHERE n.node_type = 'Equipment'
      AND n.node_id IN (
          SELECT node_id FROM node_properties
          WHERE property_name = 'location' AND value_text = %s
      )
    GROUP BY n.node_id, n.natural_key
"""

_FLOC_SQL = """
    SELECT e.target_id AS node_id, fl.natural_key AS "FunctionalLocation"
    FROM edges e
    JOIN nodes fl ON fl.node_id = e.source_id
    WHERE e.relationship = 'HAS_EQUIPMENT'
      AND e.target_id = ANY(%s)
"""


def fetch_equipment_master(cfg: Config, force_refresh: bool = False) -> pd.DataFrame:
    """Pull the location's equipment master from the KG. Cached per run.year."""
    cache = cfg.equipment_master_cache
    print(f"[kg] fetch_equipment_master  force_refresh={force_refresh}  cache={cache.name}")
    if not force_refresh and cache.exists():
        print(f"[kg]   cache hit -- reading {cache} ({cache.stat().st_size // 1024} KB)")
        df = pd.read_excel(cache)
        print(f"[kg]   returned {len(df)} rows from cache")
        return df

    print(f"[kg]   no cache (or force_refresh) -- querying KG for location={cfg.plant.location_filter}")
    with get_connection(cfg) as conn:
        print("[kg]   SQL #1: equipment nodes + properties")
        equip = pd.read_sql(_EQUIP_SQL, conn, params=(cfg.plant.location_filter,))
        print(f"[kg]           -> {len(equip)} equipment rows")
        if equip.empty:
            df = equip.drop(columns=["node_id"], errors="ignore")
        else:
            print("[kg]   SQL #2: HAS_EQUIPMENT edges to FunctionalLocation")
            floc = pd.read_sql(_FLOC_SQL, conn, params=(equip["node_id"].tolist(),))
            print(f"[kg]           -> {len(floc)} equipment->FL edges")
            df = equip.merge(floc, on="node_id", how="left").drop(columns=["node_id"])

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(cache, index=False)
    print(f"[kg]   cached -> {cache.name} ({len(df)} rows)")
    return df


# =========================================================================
# Equipment Class  (Postgres table populated by ingest_equipment_class.py)
# =========================================================================
_EQCLASS_SQL = """
    SELECT level AS "Level", class AS "Class",
           description AS "Class Description", status AS "Status"
    FROM equipment_class
    ORDER BY level, class
"""


def fetch_equipment_class(cfg: Config, force_refresh: bool = False) -> pd.DataFrame:
    """Pull the equipment class hierarchy from the Postgres `equipment_class`
    table. Falls back to the Raw-data xlsx if the table doesn't exist yet."""
    cache = cfg.equipment_class_cache
    print(f"[kg] fetch_equipment_class  force_refresh={force_refresh}  cache={cache.name}")
    if not force_refresh and cache.exists():
        print(f"[kg]   cache hit -- reading {cache}")
        df = pd.read_excel(cache)
        print(f"[kg]   returned {len(df)} rows from cache")
        return df

    print("[kg]   no cache (or force_refresh) -- querying Postgres table `equipment_class`")
    try:
        with get_connection(cfg) as conn:
            df = pd.read_sql(_EQCLASS_SQL, conn)
            print(f"[kg]   returned {len(df)} class rows from Postgres")
    except Exception as e:
        print(f"[kg]   WARN: Postgres fetch failed ({e})")
        print(f"[kg]   FALLBACK: reading from xlsx {cfg.paths.equipment_class_xlsx.name}")
        df = pd.read_excel(cfg.paths.equipment_class_xlsx)[
            ["Level", "Class", "Class Description", "Status"]
        ].dropna(subset=["Class"])
        print(f"[kg]   {len(df)} class rows from fallback xlsx")

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(cache, index=False)
    print(f"[kg]   cached -> {cache.name}")
    return df


# =========================================================================
# Worklist / Tasklist  (deferred migration to KG)
# =========================================================================
def fetch_worklist(cfg: Config) -> pd.DataFrame:
    raise NotImplementedError(
        "Worklist KG extractor not yet implemented. "
        f"Read from {cfg.worklist_xlsx.name} instead."
    )


def fetch_tasklist(cfg: Config) -> pd.DataFrame:
    raise NotImplementedError(
        "Tasklist KG extractor not yet implemented. "
        f"Read from {cfg.tasklist_xlsx.name} instead."
    )


# =========================================================================
# Worklist loader (xlsx, filtered to location)   -- kept here for parity
# =========================================================================
def load_worklist_xlsx(cfg: Config, sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load worklist xlsx, filter to the configured location, optionally sample."""
    print(f"[io] load_worklist_xlsx  file={cfg.worklist_xlsx.name}")
    try:
        df = pd.read_excel(cfg.worklist_xlsx, sheet_name="Export")
        print(f"[io]   loaded sheet 'Export' -- {len(df)} rows")
    except ValueError:
        df = pd.read_excel(cfg.worklist_xlsx)
        print(f"[io]   loaded first sheet -- {len(df)} rows")

    before = len(df)
    loc = cfg.plant.location_filter.lower()
    df = df[df["Loc"].fillna("").astype(str).str.strip().str.lower() == loc]
    df = df.dropna(subset=["EquipmentNo"]).reset_index(drop=True)
    print(f"[io]   filtered to Loc={cfg.plant.location_filter} + non-null EquipmentNo: "
          f"{len(df)}/{before} rows kept")

    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
        print(f"[io]   sampled -> {len(df)} rows")
    return df
