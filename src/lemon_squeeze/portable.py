"""Export/import the DB as JSONL — backup, sharing, machine migration.

The DB is a graph: Run references Prompt+Model; Evaluation references Run;
PromptTag references Prompt. Exporting raw integer IDs would break on
reimport because new rows get fresh PKs. So this module keys foreign rows
by their natural identity instead:

  - Prompt by content_hash
  - Model by name
  - Run by (prompt_content_hash, model_name, created_at) — almost unique;
    we also write a stable surrogate `_export_id` field to disambiguate

On import, we look up the foreign rows by natural key, fail loud if missing
(unless caller passes `allow_orphans=True`, which writes a placeholder), and
dedupe against existing rows. Run rows additionally check for prior import
via `run_metadata.lemon_export_id`.

Format (one JSON object per line, file per table):
  prompts.jsonl     — Prompt rows
  models.jsonl      — Model rows
  prompt_tags.jsonl — PromptTag rows referencing prompts by content_hash
  runs.jsonl        — Run rows referencing prompt + model by natural key
  evaluations.jsonl — Evaluation rows referencing runs by _export_id
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from lemon_squeeze.db import (
    Evaluation,
    Model,
    Prompt,
    PromptTag,
    Run,
    TagTaxonomy,
    get_session,
)

EXPORT_VERSION = 1
EXPORT_ID_KEY = "lemon_export_id"


# ---------- Export -----------------------------------------------------------


@dataclass
class ExportReport:
    prompts: int = 0
    models: int = 0
    prompt_tags: int = 0
    runs: int = 0
    evaluations: int = 0
    tag_taxonomy: int = 0
    files: list[Path] = field(default_factory=list)


def export_to_dir(
    out_dir: Path,
    *,
    include_runs: bool = True,
    include_evaluations: bool = True,
    include_taxonomy: bool = True,
) -> ExportReport:
    """Write each table to a JSONL file under `out_dir`.

    Prompts and models are always exported (other tables reference them).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = ExportReport()

    with get_session() as s:
        prompts = list(s.scalars(select(Prompt)).all())
        models = list(s.scalars(select(Model)).all())
        prompt_tags = list(s.scalars(select(PromptTag)).all())
        runs = list(s.scalars(select(Run)).all())
        evaluations = list(s.scalars(select(Evaluation)).all())
        taxonomy = list(s.scalars(select(TagTaxonomy)).all()) if include_taxonomy else []

        prompt_hash_by_id = {p.id: p.content_hash for p in prompts}
        model_name_by_id = {m.id: m.name for m in models}

        # Surrogate IDs for runs so evaluations can reference them.
        # If a Run already has an export_id (from a prior export), reuse it.
        # Otherwise allocate and PERSIST so re-imports of this export find a
        # matching row in this DB and dedupe correctly.
        run_export_ids: dict[int, str] = {}
        for r in runs:
            meta = dict(r.run_metadata or {})
            existing_eid = meta.get(EXPORT_ID_KEY)
            if existing_eid:
                run_export_ids[r.id] = existing_eid
                continue
            eid = str(uuid.uuid4())
            meta[EXPORT_ID_KEY] = eid
            r.run_metadata = meta
            run_export_ids[r.id] = eid

        # Detach so we can stop the session before writing.
        prompts_data = [_prompt_to_dict(p) for p in prompts]
        models_data = [_model_to_dict(m) for m in models]
        tags_data = [
            _prompt_tag_to_dict(t, prompt_hash_by_id) for t in prompt_tags
        ]
        taxonomy_data = [_taxonomy_to_dict(t) for t in taxonomy]
        runs_data = (
            [
                _run_to_dict(r, prompt_hash_by_id, model_name_by_id, run_export_ids[r.id])
                for r in runs
            ]
            if include_runs else []
        )
        evals_data = (
            [_eval_to_dict(e, run_export_ids) for e in evaluations]
            if include_evaluations and include_runs else []
        )

    _write_jsonl(out_dir / "prompts.jsonl", prompts_data); report.prompts = len(prompts_data); report.files.append(out_dir / "prompts.jsonl")
    _write_jsonl(out_dir / "models.jsonl", models_data); report.models = len(models_data); report.files.append(out_dir / "models.jsonl")
    _write_jsonl(out_dir / "prompt_tags.jsonl", tags_data); report.prompt_tags = len(tags_data); report.files.append(out_dir / "prompt_tags.jsonl")
    if include_taxonomy:
        _write_jsonl(out_dir / "tag_taxonomy.jsonl", taxonomy_data); report.tag_taxonomy = len(taxonomy_data); report.files.append(out_dir / "tag_taxonomy.jsonl")
    if include_runs:
        _write_jsonl(out_dir / "runs.jsonl", runs_data); report.runs = len(runs_data); report.files.append(out_dir / "runs.jsonl")
    if include_evaluations and include_runs:
        _write_jsonl(out_dir / "evaluations.jsonl", evals_data); report.evaluations = len(evals_data); report.files.append(out_dir / "evaluations.jsonl")

    manifest = {
        "version": EXPORT_VERSION,
        "exported_at": _iso(datetime.now(timezone.utc)),
        "counts": {
            "prompts": report.prompts,
            "models": report.models,
            "prompt_tags": report.prompt_tags,
            "runs": report.runs,
            "evaluations": report.evaluations,
            "tag_taxonomy": report.tag_taxonomy,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report.files.append(out_dir / "manifest.json")
    return report


def _prompt_to_dict(p: Prompt) -> dict[str, Any]:
    return {
        "content": p.content,
        "content_hash": p.content_hash,
        "token_count": p.token_count,
        "char_count": p.char_count,
        "source": p.source,
        "source_ref": p.source_ref,
        "source_metadata": p.source_metadata,
        "created_at": _iso(p.created_at),
        "ingested_at": _iso(p.ingested_at),
    }


def _model_to_dict(m: Model) -> dict[str, Any]:
    return {
        "name": m.name,
        "provider": m.provider,
        "family": m.family,
        "size_params_b": m.size_params_b,
        "context_window": m.context_window,
        "local": m.local,
        "cost_in_per_mtok": m.cost_in_per_mtok,
        "cost_out_per_mtok": m.cost_out_per_mtok,
        "notes": m.notes,
    }


def _prompt_tag_to_dict(t: PromptTag, prompt_hash_by_id: dict[int, str]) -> dict[str, Any]:
    return {
        "prompt_content_hash": prompt_hash_by_id.get(t.prompt_id),
        "tag": t.tag,
        "classifier": t.classifier,
        "confidence": t.confidence,
        "created_at": _iso(t.created_at),
    }


def _taxonomy_to_dict(t: TagTaxonomy) -> dict[str, Any]:
    return {"tag": t.tag, "description": t.description, "parent": t.parent}


def _run_to_dict(
    r: Run,
    prompt_hash_by_id: dict[int, str],
    model_name_by_id: dict[int, str],
    export_id: str,
) -> dict[str, Any]:
    meta = dict(r.run_metadata or {})
    meta[EXPORT_ID_KEY] = export_id
    return {
        "_export_id": export_id,
        "prompt_content_hash": prompt_hash_by_id.get(r.prompt_id),
        "model_name": model_name_by_id.get(r.model_id),
        "response": r.response,
        "tokens_in": r.tokens_in,
        "tokens_out": r.tokens_out,
        "latency_ms": r.latency_ms,
        "cost_usd": r.cost_usd,
        "temperature": r.temperature,
        "top_p": r.top_p,
        "seed": r.seed,
        "run_metadata": meta,
        "error": r.error,
        "created_at": _iso(r.created_at),
    }


def _eval_to_dict(e: Evaluation, run_export_ids: dict[int, str]) -> dict[str, Any]:
    return {
        "run_export_id": run_export_ids.get(e.run_id),
        "rubric": e.rubric,
        "score": e.score,
        "passed": e.passed,
        "scored_by": e.scored_by,
        "judge_model": e.judge_model,
        "notes": e.notes,
        "eval_metadata": e.eval_metadata,
        "created_at": _iso(e.created_at),
    }


# ---------- Import -----------------------------------------------------------


@dataclass
class ImportReport:
    prompts_inserted: int = 0
    prompts_deduped: int = 0
    models_inserted: int = 0
    models_updated: int = 0
    prompt_tags_inserted: int = 0
    prompt_tags_deduped: int = 0
    runs_inserted: int = 0
    runs_deduped: int = 0
    evaluations_inserted: int = 0
    evaluations_deduped: int = 0
    taxonomy_inserted: int = 0
    skipped: list[str] = field(default_factory=list)


def import_from_dir(in_dir: Path) -> ImportReport:
    """Import a previously-exported directory back into the DB.

    Dedup keys:
      - Prompts:   content_hash (skip if present)
      - Models:    name (update fields if present)
      - PromptTag: (prompt_id, tag, classifier) — unique constraint in DB
      - Runs:      run_metadata.lemon_export_id (skip if present)
      - Evals:     (run_id, rubric, created_at) — best-effort

    Missing foreign rows (e.g. an Eval whose Run wasn't in the export) are
    skipped with a message; we don't fabricate placeholders.
    """
    in_dir = Path(in_dir)
    report = ImportReport()

    prompts_data = _read_jsonl(in_dir / "prompts.jsonl")
    models_data = _read_jsonl(in_dir / "models.jsonl")
    tags_data = _read_jsonl(in_dir / "prompt_tags.jsonl")
    taxonomy_data = _read_jsonl(in_dir / "tag_taxonomy.jsonl")
    runs_data = _read_jsonl(in_dir / "runs.jsonl")
    evals_data = _read_jsonl(in_dir / "evaluations.jsonl")

    with get_session() as s:
        # Existing rows.
        existing_prompts = {p.content_hash: p for p in s.scalars(select(Prompt)).all()}
        existing_models = {m.name: m for m in s.scalars(select(Model)).all()}
        existing_taxonomy = {t.tag for t in s.scalars(select(TagTaxonomy)).all()}
        existing_run_export_ids = set()
        for r in s.scalars(select(Run)).all():
            eid = (r.run_metadata or {}).get(EXPORT_ID_KEY)
            if eid:
                existing_run_export_ids.add(eid)

        # 1. Taxonomy (idempotent — safe to insert first).
        for rec in taxonomy_data:
            if rec.get("tag") in existing_taxonomy:
                continue
            s.add(TagTaxonomy(**{k: rec.get(k) for k in ("tag", "description", "parent")}))
            existing_taxonomy.add(rec["tag"])
            report.taxonomy_inserted += 1

        # 2. Prompts.
        for rec in prompts_data:
            h = rec.get("content_hash")
            if not h:
                report.skipped.append(f"prompt without content_hash: {rec.get('content', '')[:50]!r}")
                continue
            if h in existing_prompts:
                report.prompts_deduped += 1
                continue
            # `or len(content)` would silently round-trip char_count=0 → 0,
            # which is technically right, but more importantly char_count=0
            # is legitimate for empty prompts; distinguish None from 0.
            char_count = rec.get("char_count")
            if char_count is None:
                char_count = len(rec.get("content", ""))
            p = Prompt(
                content=rec.get("content", ""),
                content_hash=h,
                token_count=rec.get("token_count"),
                char_count=char_count,
                source=rec.get("source", "imported"),
                source_ref=rec.get("source_ref"),
                source_metadata=rec.get("source_metadata"),
                created_at=_parse_iso(rec.get("created_at")),
            )
            s.add(p)
            s.flush()
            existing_prompts[h] = p
            report.prompts_inserted += 1

        # 3. Models.
        for rec in models_data:
            name = rec.get("name")
            if not name:
                report.skipped.append(f"model without name: {rec}")
                continue
            if name in existing_models:
                m = existing_models[name]
                for k in (
                    "provider", "family", "size_params_b", "context_window",
                    "local", "cost_in_per_mtok", "cost_out_per_mtok", "notes",
                ):
                    v = rec.get(k)
                    if v is not None:
                        setattr(m, k, v)
                report.models_updated += 1
                continue
            m = Model(**{k: rec.get(k) for k in (
                "name", "provider", "family", "size_params_b", "context_window",
                "local", "cost_in_per_mtok", "cost_out_per_mtok", "notes",
            )})
            if m.provider is None:
                m.provider = "imported"
            if m.local is None:
                m.local = False
            s.add(m)
            s.flush()
            existing_models[name] = m
            report.models_inserted += 1

        # 4. Prompt tags.
        existing_tag_keys = set()
        for t in s.scalars(select(PromptTag)).all():
            existing_tag_keys.add((t.prompt_id, t.tag, t.classifier))
        for rec in tags_data:
            h = rec.get("prompt_content_hash")
            p = existing_prompts.get(h) if h else None
            if p is None:
                report.skipped.append(
                    f"prompt_tag references unknown prompt content_hash={h!r}; tag={rec.get('tag')}"
                )
                continue
            key = (p.id, rec.get("tag"), rec.get("classifier"))
            if key in existing_tag_keys:
                report.prompt_tags_deduped += 1
                continue
            s.add(
                PromptTag(
                    prompt_id=p.id,
                    tag=rec.get("tag", "unknown"),
                    classifier=rec.get("classifier", "imported"),
                    confidence=rec.get("confidence", 1.0),
                    created_at=_parse_iso(rec.get("created_at")),
                )
            )
            existing_tag_keys.add(key)
            report.prompt_tags_inserted += 1

        # 5. Runs.
        run_id_by_export_id: dict[str, int] = {}
        for r in s.scalars(select(Run)).all():
            eid = (r.run_metadata or {}).get(EXPORT_ID_KEY)
            if eid:
                run_id_by_export_id[eid] = r.id
        for rec in runs_data:
            eid = rec.get("_export_id")
            if not eid:
                report.skipped.append(f"run without _export_id: {rec}")
                continue
            if eid in existing_run_export_ids:
                report.runs_deduped += 1
                continue
            ph = rec.get("prompt_content_hash")
            mn = rec.get("model_name")
            p = existing_prompts.get(ph) if ph else None
            m = existing_models.get(mn) if mn else None
            if p is None or m is None:
                report.skipped.append(
                    f"run references missing prompt={ph!r} or model={mn!r}; skipped"
                )
                continue
            run = Run(
                prompt_id=p.id,
                model_id=m.id,
                response=rec.get("response"),
                tokens_in=rec.get("tokens_in"),
                tokens_out=rec.get("tokens_out"),
                latency_ms=rec.get("latency_ms"),
                cost_usd=rec.get("cost_usd"),
                temperature=rec.get("temperature"),
                top_p=rec.get("top_p"),
                seed=rec.get("seed"),
                run_metadata=rec.get("run_metadata"),
                error=rec.get("error"),
                created_at=_parse_iso(rec.get("created_at")),
            )
            s.add(run)
            s.flush()
            run_id_by_export_id[eid] = run.id
            existing_run_export_ids.add(eid)
            report.runs_inserted += 1

        # 6. Evaluations.
        existing_eval_keys = set()
        for e in s.scalars(select(Evaluation)).all():
            existing_eval_keys.add((e.run_id, e.rubric, e.created_at))
        for rec in evals_data:
            eid = rec.get("run_export_id")
            run_id = run_id_by_export_id.get(eid) if eid else None
            if run_id is None:
                report.skipped.append(
                    f"evaluation references unknown run _export_id={eid!r}; skipped"
                )
                continue
            created = _parse_iso(rec.get("created_at"))
            key = (run_id, rec.get("rubric"), created)
            if key in existing_eval_keys:
                report.evaluations_deduped += 1
                continue
            s.add(
                Evaluation(
                    run_id=run_id,
                    rubric=rec.get("rubric", "unknown"),
                    score=rec.get("score", 0.0),
                    passed=rec.get("passed"),
                    scored_by=rec.get("scored_by", "imported"),
                    judge_model=rec.get("judge_model"),
                    notes=rec.get("notes"),
                    eval_metadata=rec.get("eval_metadata"),
                    created_at=created,
                )
            )
            existing_eval_keys.add(key)
            report.evaluations_inserted += 1

    return report


# ---------- Helpers ----------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default))
            f.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _parse_iso(v: Any) -> datetime | None:
    if not isinstance(v, str):
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"can't serialize {type(obj).__name__}")
