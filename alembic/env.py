"""Alembic environment.

Reads the DB URL from `lemon_squeeze.config.settings.db_url` so it always
matches whatever `LEMON_DB_PATH` is set to. The Base metadata comes from
the project's ORM so autogenerate works.
"""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from lemon_squeeze.config import settings
from lemon_squeeze.db import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Wire the URL from project settings every time, ignoring any value in
# alembic.ini. That way `LEMON_DB_PATH=./test.db alembic upgrade head` works.
config.set_main_option("sqlalchemy.url", settings.db_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL without DB connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # SQLite needs batch mode for ALTER TABLE
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # SQLite-safe
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
