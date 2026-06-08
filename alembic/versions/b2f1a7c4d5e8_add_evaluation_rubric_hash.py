"""Add Evaluation.rubric_hash for staleness detection

Revision ID: b2f1a7c4d5e8
Revises: dd2bf37a86ee
Create Date: 2026-06-07 22:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2f1a7c4d5e8"
down_revision: Union[str, None] = "dd2bf37a86ee"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite-safe: nullable column add via batch mode.
    with op.batch_alter_table("evaluations") as batch_op:
        batch_op.add_column(sa.Column("rubric_hash", sa.String(length=64), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("evaluations") as batch_op:
        batch_op.drop_column("rubric_hash")
