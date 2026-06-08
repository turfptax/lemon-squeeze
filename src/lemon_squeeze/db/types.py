"""Custom SQLAlchemy column types.

`UTCDateTime` is a thin TypeDecorator over `DateTime(timezone=True)` that:
  - Coerces naive datetimes to UTC on the way IN (writes).
  - Re-applies tz=UTC on the way OUT (reads) — SQLite ignores `timezone=True`
    and hands back naive datetimes, which would then trip aware-vs-naive math.

Putting this in the data layer is the right altitude: every Run/Eval/Prompt
timestamp comes back aware automatically, so the consuming code (report,
dashboard) doesn't have to keep redoing the same `if tzinfo is None: ...`
dance.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator):
    """Stored as `DateTime(timezone=True)`; presented as tz-aware UTC."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        # Convert to UTC if it came in as another zone — keeps DB storage uniform.
        return value.astimezone(timezone.utc)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
