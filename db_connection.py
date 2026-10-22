"""Postgres connection to the SAP PM knowledge graph via Azure AD."""

import psycopg
from azure.identity import AzureCliCredential

from config import Config, load_config


def get_connection(cfg: Config | None = None):
    """Open a Postgres connection using Azure AD token (needs `az login`)."""
    cfg = cfg or load_config()
    print(f"[db] requesting Azure AD token (needs 'az login') for scope {cfg.kg.token_scope}")
    token = AzureCliCredential(process_timeout=20).get_token(cfg.kg.token_scope)
    print(f"[db] token acquired; connecting to {cfg.kg.database}@{cfg.kg.host}:{cfg.kg.port} ...")
    conn = psycopg.connect(
        host=cfg.kg.host, port=cfg.kg.port, dbname=cfg.kg.database,
        user=cfg.kg.user, password=token.token,
        sslmode="require", connect_timeout=20,
    )
    print("[db] connection open")
    return conn


if __name__ == "__main__":
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT current_database(), current_user;")
        print("Connected:", cur.fetchone())
