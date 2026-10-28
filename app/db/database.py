import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.core.config import get_settings
from app.core.ids import normalize_id


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(get_settings().db_path, timeout=10, factory=ClosingConnection)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    db = connect()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def customer_exists(customer_id: str) -> bool:
    with connect() as db:
        return db.execute("SELECT 1 FROM customers WHERE customer_id=?", (customer_id,)).fetchone() is not None


def owned_order(db: sqlite3.Connection, order_id: str, customer_id: str):
    return db.execute("SELECT * FROM orders WHERE order_id=? AND customer_id=?", (normalize_id(order_id), customer_id)).fetchone()


def policy_parameter(db: sqlite3.Connection, key: str) -> str | None:
    row = db.execute("SELECT value FROM policy_parameters WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None
