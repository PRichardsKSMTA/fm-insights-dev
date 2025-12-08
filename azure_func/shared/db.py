"""Database helpers for invoking SQL stored procedures with rich diagnostics."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

try:
    import pyodbc  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    pyodbc = None  # type: ignore
    _import_error = exc
else:
    _import_error = None

from .logging_utils import get_json_logger

class DatabaseError(RuntimeError):
    """Raised when database access fails."""

@dataclass
class _CursorResult:
    columns: Sequence[str]
    rows: List[Sequence[Any]]

class DatabaseClient:
    """pyodbc-based helper with robust logging, introspection, and multi-result handling."""

    def __init__(self, connection_string: Optional[str] = None) -> None:
        self._logger = get_json_logger("mmqb.db")
        if pyodbc is None:
            self._logger.error("pyodbc not installed", extra={"event": "db_init_error"})
            raise DatabaseError("pyodbc is required but is not installed") from _import_error

        env_connection = os.environ.get("SQL_CONN_STR") or os.environ.get("SQL_CONNECTION_STRING")
        self.connection_string = connection_string or env_connection
        if not self.connection_string:
            self._logger.error("SQL connection string missing", extra={"event": "db_init_error"})
            raise DatabaseError("SQL_CONN_STR environment variable is not configured (SQL_CONNECTION_STRING is deprecated)")

        self._logger.info("DatabaseClient initialized", extra={"event": "db_init_ok"})

    def _get_connection(self) -> "pyodbc.Connection":  # type: ignore[name-defined]
        self._logger.info("Opening DB connection", extra={"event": "db_connect_start"})
        conn = pyodbc.connect(self.connection_string, autocommit=False)  # type: ignore[attr-defined]
        # Emit server/db identity so we know exactly where we are
        try:
            cur = conn.cursor()
            cur.execute("SELECT @@SERVERNAME AS server_name, DB_NAME() AS database_name;")
            row = cur.fetchone()
            if row:
                self._logger.info(
                    "DB identity",
                    extra={"event": "db_identity", "server": row.server_name, "database": row.database_name},
                )
        except Exception:
            pass
        self._logger.info("DB connection opened", extra={"event": "db_connect_ok"})
        return conn

    @staticmethod
    def _materialize(cursor: "pyodbc.Cursor") -> _CursorResult:  # type: ignore[name-defined]
        columns = [col[0] for col in (cursor.description or [])]
        rows = cursor.fetchall() if columns else []
        return _CursorResult(columns=columns, rows=rows)

    @staticmethod
    def _coerce_rows(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[dict]:
        return [dict(zip(columns, row)) for row in rows]

    def _log_params_preview(self, params: Sequence[Any] | None) -> List[dict]:
        preview = []
        for idx, p in enumerate(list(params or [])):
            pv = str(p)
            if len(pv) > 200:
                pv = pv[:200] + "…"
            preview.append({"idx": idx, "type": type(p).__name__, "value_preview": pv})
        return preview

    # --- New: generic query executor for diagnostics ---------------------------------

    def execute_query(self, sql: str, params: Sequence[Any] | None = None) -> Tuple[List[str], List[Tuple[Any, ...]]]:
        """Run an arbitrary SELECT for diagnostics."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params or [])
            cols = [c[0] for c in (cur.description or [])]
            rows = cur.fetchall() if cols else []
            return cols, rows

    def introspect_procedure(self, proc_name: str) -> None:
        """Log the canonical schema, object_id, and parameters of a procedure."""
        try:
            # Accept inputs like "DEV.MyProc" or "MyProc"
            if "." in proc_name:
                schema_name, object_name = proc_name.split(".", 1)
            else:
                schema_name, object_name = "DEV", proc_name

            sql = """
                SELECT
                    QUOTENAME(OBJECT_SCHEMA_NAME(o.object_id)) + '.' + QUOTENAME(o.name) AS proc_full_name,
                    o.object_id,
                    p.parameter_id,
                    p.name AS param_name,
                    TYPE_NAME(p.user_type_id) AS param_type,
                    p.max_length,
                    p.is_output
                FROM sys.objects o
                LEFT JOIN sys.parameters p ON p.object_id = o.object_id
                WHERE o.type IN ('P', 'PC') AND o.name = ? AND OBJECT_SCHEMA_NAME(o.object_id) = ?
                ORDER BY p.parameter_id;
            """
            cols, rows = self.execute_query(sql, [object_name, schema_name])
            params_desc = [
                {
                    "parameter_id": r[2],
                    "name": r[3],
                    "type": r[4],
                    "max_length": r[5],
                    "is_output": bool(r[6]),
                }
                for r in rows
            ]
            self._logger.info(
                f"Proc introspection {schema_name}.{object_name}",
                extra={
                    "event": "proc_introspect",
                    "proc": f"{schema_name}.{object_name}",
                    "object_id": rows[0][1] if rows else None,
                    "params": params_desc,
                },
            )
        except Exception as exc:
            self._logger.exception("Proc introspection failed", extra={"event": "proc_introspect_error", "proc": proc_name})
            # Do not raise; purely diagnostic

    # --- Core: call stored procedure and collect the FIRST non-empty result set -------

    def call_procedure(self, proc_name: str, params: Sequence[Any] | None = None) -> List[dict]:
        """Invoke a stored procedure and return the first data-bearing result set as list[dict]."""
        self._logger.info(
            f"Calling stored procedure {proc_name}",
            extra={"event": "proc_call_start", "proc": proc_name, "params": self._log_params_preview(params)},
        )
        params = list(params or [])
        sql = f"EXEC {proc_name}"
        if params:
            placeholders = ", ".join(["?"] * len(params))
            sql = f"{sql} {placeholders}"

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)

                # Walk result sets until we find one with columns (actual result set)
                rows_dicts: List[dict] = []
                result_cols: List[str] = []
                result_rows_count = 0
                set_index = 0

                while True:
                    desc = cursor.description
                    if desc:  # found a data-bearing result set
                        cols = [c[0] for c in desc]
                        data_rows = cursor.fetchall()
                        result_cols = cols
                        rows_dicts = self._coerce_rows(cols, data_rows)
                        result_rows_count = len(rows_dicts)
                        self._logger.info(
                            f"Result set #{set_index} rows={result_rows_count}",
                            extra={
                                "event": "proc_result_set",
                                "proc": proc_name,
                                "set_index": set_index,
                                "columns": cols[:100],
                                "row_count": result_rows_count,
                                "sample": rows_dicts[0] if rows_dicts else None,
                            },
                        )
                        break

                    # No columns in this set; advance
                    has_more = cursor.nextset()
                    if not has_more:
                        break
                    set_index += 1

                # Drain any additional sets (to keep drivers happy)
                while cursor.nextset():
                    set_index += 1

                conn.commit()

                if not result_cols:
                    # No data-bearing result sets at all
                    self._logger.warning(
                        "Procedure returned no result sets",
                        extra={"event": "proc_no_resultset", "proc": proc_name},
                    )
                    return []

                self._logger.info(
                    f"Stored procedure {proc_name} completed rows={result_rows_count}",
                    extra={"event": "proc_call_ok", "proc": proc_name, "row_count": result_rows_count, "columns": result_cols[:100]},
                )
                return rows_dicts

        except Exception as exc:  # pragma: no cover
            self._logger.exception(
                f"Stored procedure {proc_name} failed",
                extra={"event": "proc_call_error", "proc": proc_name},
            )
            raise DatabaseError(f"Failed to execute stored procedure {proc_name}") from exc

    def call_procedure_no_results(self, proc_name: str, params: Sequence[Any] | None = None) -> None:
        """Invoke a stored procedure where no result set is expected."""
        self._logger.info(
            f"Calling stored procedure (no results) {proc_name}",
            extra={"event": "proc_call_nr_start", "proc": proc_name, "params": self._log_params_preview(params)},
        )
        params = list(params or [])
        sql = f"EXEC {proc_name}"
        if params:
            placeholders = ", ".join(["?"] * len(params))
            sql = f"{sql} {placeholders}"
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                while cursor.nextset():
                    pass
                conn.commit()
                self._logger.info(
                    f"Stored procedure (no results) {proc_name} completed",
                    extra={"event": "proc_call_nr_ok", "proc": proc_name},
                )
        except Exception as exc:  # pragma: no cover
            self._logger.exception(
                f"Stored procedure (no results) {proc_name} failed",
                extra={"event": "proc_call_nr_error", "proc": proc_name},
            )
            raise DatabaseError(f"Failed to execute stored procedure {proc_name}") from exc

    def fetch_one(self, proc_name: str, params: Sequence[Any] | None = None) -> Optional[dict]:
        rows = self.call_procedure(proc_name, params)
        return rows[0] if rows else None
