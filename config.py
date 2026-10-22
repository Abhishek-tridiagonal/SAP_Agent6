"""Config loader for the SD & TA Agent.

Reads config.ini (non-secret settings) + .env (secrets). Resolves relative
paths against this file's directory. Exposes a single `load_config()` entry
point that returns a flat `Config` object other modules read from.

Usage:
    from config import load_config
    cfg = load_config()
    cfg.year                                  # 2026
    cfg.paths.processed_data_dir              # Path(...)
    cfg.flags.run_cleaning                    # True / False
    cfg.langfuse.agent_name                   # "SD & TA Agent"
    cfg.secrets.azure_foundry_api_key         # from .env
"""

from configparser import ConfigParser, ExtendedInterpolation
from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.ini"
ENV_PATH = BASE_DIR / ".env"


# =========================================================================
# Typed section wrappers (so other modules can autocomplete + fail fast
# if config keys are missing)
# =========================================================================
@dataclass
class Paths:
    raw_data_dir: Path
    processed_data_dir: Path
    ia05_template: Path
    equipment_class_xlsx: Path


@dataclass
class Plant:
    code: str          # "SK"
    werks: str         # "M014"
    location_filter: str  # "BOKOR"


@dataclass
class LLM:
    model: str
    foundry_base_url: str
    temperature: float
    max_tokens_default: int
    max_tokens_cleaning: int


@dataclass
class Langfuse:
    host: str
    project: str
    agent_name: str
    session_cleaning: str
    session_job_library: str
    session_merge: str
    session_unresolved: str


@dataclass
class KG:
    host: str
    port: int
    database: str
    user: str
    token_scope: str


@dataclass
class CleaningAgent:
    batch_size: int
    max_concurrent_llm: int
    max_pool_size_for_llm: int


@dataclass
class JobLibraryAgent:
    class_level: int


@dataclass
class UnresolvedClassifier:
    fallback_levels: list    # e.g. [4, 3, 2]


@dataclass
class Flags:
    refresh_equipment_master: bool
    refresh_equipment_class: bool
    run_cleaning: bool
    run_job_library: bool
    run_unresolved_classifier: bool
    run_merge: bool
    run_sap_export: bool


@dataclass
class RateLimit:
    max_retries: int
    default_wait_sec: int


@dataclass
class Secrets:
    azure_ai_key: str
    azure_foundry_api_key: str
    langfuse_secret_key: str
    langfuse_public_key: str


@dataclass
class Config:
    year: int
    paths: Paths
    plant: Plant
    llm: LLM
    langfuse: Langfuse
    kg: KG
    cleaning_agent: CleaningAgent
    job_library_agent: JobLibraryAgent
    unresolved_classifier: UnresolvedClassifier
    flags: Flags
    rate_limit: RateLimit
    secrets: Secrets

    # Convenience year-stamped output paths.
    @property
    def worklist_xlsx(self) -> Path:
        return self.paths.raw_data_dir / f"Worklist_{self.year}.xlsx"

    @property
    def tasklist_xlsx(self) -> Path:
        return self.paths.raw_data_dir / f"Tasklist_{self.year}.xlsx"

    @property
    def equipment_master_cache(self) -> Path:
        return self.paths.processed_data_dir / f"Equipment_Master_Bokor_{self.year}.xlsx"

    @property
    def equipment_class_cache(self) -> Path:
        return self.paths.processed_data_dir / "Equipment_Class_cache.xlsx"

    @property
    def cleanly_mapped_worklist(self) -> Path:
        return self.paths.processed_data_dir / f"Cleanly_Mapped_Worklist_Bokor_{self.year}.xlsx"

    @property
    def unresolved_items(self) -> Path:
        return self.paths.processed_data_dir / f"Unresolved_Worklist_Items_{self.year}.xlsx"

    @property
    def job_library_new(self) -> Path:
        return self.paths.processed_data_dir / f"Job_Library_Bokor_{self.year}_new.xlsx"

    @property
    def job_library_consolidated(self) -> Path:
        return self.paths.processed_data_dir / "Job_Library_Consolidated.xlsx"

    @property
    def sap_upload(self) -> Path:
        return self.paths.processed_data_dir / "Job_Library_IA05_Upload.xlsx"


# =========================================================================
# Loader
# =========================================================================
def _resolve(p: str) -> Path:
    """Resolve a path from config.ini. Absolute paths pass through; relative
    paths are resolved against the Agent Code folder (BASE_DIR)."""
    pp = Path(p)
    return pp if pp.is_absolute() else (BASE_DIR / pp).resolve()


def _bool(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "on"}


def _int_list(v: str) -> list:
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def load_config(config_path: Path = CONFIG_PATH, env_path: Path = ENV_PATH) -> Config:
    load_dotenv(env_path)

    cp = ConfigParser(interpolation=ExtendedInterpolation())
    cp.read(config_path, encoding="utf-8")

    paths = Paths(
        raw_data_dir       = _resolve(cp["paths"]["raw_data_dir"]),
        processed_data_dir = _resolve(cp["paths"]["processed_data_dir"]),
        ia05_template      = _resolve(cp["paths"]["ia05_template"]),
        equipment_class_xlsx = _resolve(cp["paths"]["equipment_class_xlsx"]),
    )

    plant = Plant(
        code = cp["plant"]["code"],
        werks = cp["plant"]["werks"],
        location_filter = cp["plant"]["location_filter"],
    )

    llm = LLM(
        model = cp["llm"]["model"],
        foundry_base_url = cp["llm"]["foundry_base_url"],
        temperature = cp["llm"].getfloat("temperature"),
        max_tokens_default = cp["llm"].getint("max_tokens_default"),
        max_tokens_cleaning = cp["llm"].getint("max_tokens_cleaning"),
    )

    langfuse = Langfuse(
        host = cp["langfuse"]["host"],
        project = cp["langfuse"]["project"],
        agent_name = cp["langfuse"]["agent_name"],
        session_cleaning = cp["langfuse"]["session_cleaning"],
        session_job_library = cp["langfuse"]["session_job_library"],
        session_merge = cp["langfuse"]["session_merge"],
        session_unresolved = cp["langfuse"]["session_unresolved"],
    )

    kg = KG(
        host = cp["kg"]["host"],
        port = cp["kg"].getint("port"),
        database = cp["kg"]["database"],
        user = cp["kg"]["user"],
        token_scope = cp["kg"]["token_scope"],
    )

    cleaning = CleaningAgent(
        batch_size = cp["cleaning_agent"].getint("batch_size"),
        max_concurrent_llm = cp["cleaning_agent"].getint("max_concurrent_llm"),
        max_pool_size_for_llm = cp["cleaning_agent"].getint("max_pool_size_for_llm", fallback=500),
    )

    jlib = JobLibraryAgent(
        class_level = cp["job_library_agent"].getint("class_level"),
    )

    unresolved = UnresolvedClassifier(
        fallback_levels = _int_list(cp["unresolved_classifier"]["fallback_levels"]),
    )

    flags = Flags(
        refresh_equipment_master  = _bool(cp["flags"]["refresh_equipment_master"]),
        refresh_equipment_class   = _bool(cp["flags"]["refresh_equipment_class"]),
        run_cleaning              = _bool(cp["flags"]["run_cleaning"]),
        run_job_library           = _bool(cp["flags"]["run_job_library"]),
        run_unresolved_classifier = _bool(cp["flags"]["run_unresolved_classifier"]),
        run_merge                 = _bool(cp["flags"]["run_merge"]),
        run_sap_export            = _bool(cp["flags"]["run_sap_export"]),
    )

    rate = RateLimit(
        max_retries = cp["rate_limit"].getint("max_retries"),
        default_wait_sec = cp["rate_limit"].getint("default_wait_sec"),
    )

    secrets = Secrets(
        azure_ai_key          = os.getenv("AZURE_AI_KEY", ""),
        azure_foundry_api_key = os.getenv("AZURE_FOUNDRY_API_KEY", ""),
        langfuse_secret_key   = os.getenv("LANGFUSE_SECRET_KEY", ""),
        langfuse_public_key   = os.getenv("LANGFUSE_PUBLIC_KEY", ""),
    )

    # Export Langfuse keys to env so the Langfuse SDK's get_client() finds them.
    os.environ.setdefault("LANGFUSE_HOST", langfuse.host)
    os.environ.setdefault("LANGFUSE_BASE_URL", langfuse.host)

    return Config(
        year=cp["run"].getint("year"),
        paths=paths, plant=plant, llm=llm, langfuse=langfuse, kg=kg,
        cleaning_agent=cleaning, job_library_agent=jlib,
        unresolved_classifier=unresolved, flags=flags, rate_limit=rate,
        secrets=secrets,
    )


if __name__ == "__main__":
    # Smoke test.
    cfg = load_config()
    print(f"Year:               {cfg.year}")
    print(f"Plant:              {cfg.plant.code} / WERKS={cfg.plant.werks}")
    print(f"Class level:        {cfg.job_library_agent.class_level}")
    print(f"Processed data dir: {cfg.paths.processed_data_dir}")
    print(f"IA05 template:      {cfg.paths.ia05_template}")
    print(f"Worklist xlsx:      {cfg.worklist_xlsx}")
    print(f"Fallback levels:    {cfg.unresolved_classifier.fallback_levels}")
    print(f"Flags: {cfg.flags}")
