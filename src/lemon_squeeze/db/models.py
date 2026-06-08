"""SQLAlchemy ORM models for the prompt/run/eval store.

Design notes:
- `prompts.content_hash` is a SHA-256 of the normalized content and is unique;
  ingestion dedupes against it across sources so the same prompt isn't double-counted.
- `prompt_tags` carries (tag, classifier, confidence) so multiple classifiers can
  vote on the same prompt without overwriting each other.
- `runs` is keyed by (prompt, model) but allows many rows — every execution is kept
  so we can compare regressions across model versions or sampling params.
- `evaluations` is per-run, per-rubric. `scored_by` records whether a human, an
  LLM judge, or an automated check produced the score.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from lemon_squeeze.db.types import UTCDateTime


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Prompt(Base):
    __tablename__ = "prompts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)

    token_count: Mapped[int | None] = mapped_column(Integer)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)

    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_ref: Mapped[str | None] = mapped_column(String(512))
    source_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    created_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    ingested_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=_utcnow, nullable=False
    )

    tags: Mapped[list[PromptTag]] = relationship(
        back_populates="prompt", cascade="all, delete-orphan"
    )
    runs: Mapped[list[Run]] = relationship(back_populates="prompt", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_prompts_source_created", "source", "created_at"),)


class TagTaxonomy(Base):
    """Optional taxonomy of supported tags. Used by the heuristic classifier and for UI grouping."""

    __tablename__ = "tag_taxonomy"

    tag: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[str | None] = mapped_column(Text)
    parent: Mapped[str | None] = mapped_column(String(64), ForeignKey("tag_taxonomy.tag"))


class PromptTag(Base):
    __tablename__ = "prompt_tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prompt_id: Mapped[int] = mapped_column(
        ForeignKey("prompts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tag: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    classifier: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=_utcnow, nullable=False
    )

    prompt: Mapped[Prompt] = relationship(back_populates="tags")

    __table_args__ = (
        UniqueConstraint("prompt_id", "tag", "classifier", name="uq_prompt_tag_classifier"),
    )


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    family: Mapped[str | None] = mapped_column(String(64))
    size_params_b: Mapped[float | None] = mapped_column(Float)
    context_window: Mapped[int | None] = mapped_column(Integer)
    local: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    cost_in_per_mtok: Mapped[float | None] = mapped_column(Float)
    cost_out_per_mtok: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)

    runs: Mapped[list[Run]] = relationship(back_populates="model")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prompt_id: Mapped[int] = mapped_column(
        ForeignKey("prompts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_id: Mapped[int] = mapped_column(
        ForeignKey("models.id", ondelete="CASCADE"), nullable=False, index=True
    )

    response: Mapped[str | None] = mapped_column(Text)
    tokens_in: Mapped[int | None] = mapped_column(Integer)
    tokens_out: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)

    temperature: Mapped[float | None] = mapped_column(Float)
    top_p: Mapped[float | None] = mapped_column(Float)
    seed: Mapped[int | None] = mapped_column(Integer)
    run_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=_utcnow, nullable=False
    )

    prompt: Mapped[Prompt] = relationship(back_populates="runs")
    model: Mapped[Model] = relationship(back_populates="runs")
    evaluations: Mapped[list[Evaluation]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rubric: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # SHA-256 of (judge_kind + judge_config + applies_to_tags). Used to detect
    # when a rubric YAML has been edited since the eval was written, so
    # `evaluate_runs` can auto-replace stale rows instead of silently skipping.
    # Nullable for backward compatibility with rows written before this column
    # existed — those are treated as "unknown" and not flagged as stale.
    rubric_hash: Mapped[str | None] = mapped_column(String(64))
    score: Mapped[float] = mapped_column(Float, nullable=False)
    passed: Mapped[bool | None] = mapped_column(Boolean)
    scored_by: Mapped[str] = mapped_column(String(32), nullable=False)  # human | llm | auto
    judge_model: Mapped[str | None] = mapped_column(String(128))
    notes: Mapped[str | None] = mapped_column(Text)
    eval_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=_utcnow, nullable=False
    )

    run: Mapped[Run] = relationship(back_populates="evaluations")
